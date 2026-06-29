#!/usr/bin/env python3
"""
Agentic workload generator for the KV-energy experiment.

Emits a DETERMINISTIC, SEEDED replay of ReAct-style agentic traffic whose shape
is parametrised on the distributions of arXiv:2605.26297 (turns, context growth,
output composition, cache/append ratio). NOT a real agent: it replays the FORM
of the load so the energy measurement is reproducible (see PROTOCOL).

Context = shared cacheable prefix (build_prefix.py artifact, ~15K tok) + grown
history. Hit-rate regime (H0/H1/H2) is induced by how much of the history is
shared across sessions vs unique. The `failure` condition appends unique, bloated
error observations that erode effective hit-rate.

inferscope is NOT called here. On the node, the orchestrator brackets this
generator with inferscope --sample-only (energy via ADR-010/012, hit-rate via
ADR-011). For SIM validation (CPU, no NVML), --measure-hitrate scrapes the
engine's Prometheus /metrics directly to confirm the generator realises the
target regime.
"""
import argparse
import json
import random
import time
import urllib.request
from dataclasses import dataclass, asdict, field
from pathlib import Path

PREFIX_DIR = Path(__file__).parent / "prefixes"

# --- Workload shape parameters, sourced from arXiv:2605.26297 (Qwen) ----------
# Each value is anchored to a figure in PROTOCOL.md. Distributions internal to
# min/max/mean are an explicit ASSUMPTION (sampled within range), declared there.
SHADE = {
    # turns per task: Fig.3 (Qwen, mean range across benchmarks)
    "turns_mean": 31, "turns_std": 20, "turns_min": 6, "turns_max": 65,
    # per-turn append (new, non-cached tokens): §5 Append/Output ~3.6-6.1x output
    "append_out_ratio_mean": 4.8,
    # per-turn output tokens (decode): modest; tool-call dominated
    "output_tokens_mean": 180, "output_tokens_std": 90,
    # output composition: Fig.5 (Qwen Thinking ~35% thinking, tool-call heavy)
    # (composition affects token volume only here, not semantic content)
    # failure context inflation: Fig.6 (up to 1.8x)
    "failure_inflation": 1.8,
}

REGIMES = {
    # fraction of history SHARED across sessions -> induces hit-rate
    "H0": {"history_shared_frac": 0.0, "prefix_shared": False},
    "H1": {"history_shared_frac": 0.5, "prefix_shared": True},
    "H2": {"history_shared_frac": 0.95, "prefix_shared": True},
}


@dataclass
class RunManifest:
    regime: str
    condition: str           # nominal | failure
    rep: int
    seed: int
    prefix_version: str
    prefix_tokens: int
    target_context_tokens: int
    n_sessions: int
    # filled after run:
    turns_generated: int = 0
    history_tokens_final: int = 0
    requests_sent: int = 0
    hitrate_target: str = ""
    hitrate_realized: float | None = None      # from Prometheus delta (sim)
    prefix_cache_hits_delta: int | None = None
    prefix_cache_queries_delta: int | None = None
    sampling_params: dict = field(default_factory=dict)
    error: str | None = None


def load_prefix(version: str) -> tuple[str, int]:
    txt = (PREFIX_DIR / f"agentic_system_{version}.txt").read_text()
    meta = json.loads((PREFIX_DIR / f"agentic_system_{version}.meta.json").read_text())
    return txt, meta["token_count"]


def approx_tokens(text: str) -> int:
    # cheap proxy for history-growth control DURING generation (not for the
    # authoritative prefix count, which build_prefix.py does with the tokenizer).
    # ~3.5 chars/token for english+json; only used to decide when to stop growing.
    return len(text) // 4


def make_history_block(rng: random.Random, shade: dict, shared: bool,
                       failure: bool, block_idx: int) -> str:
    """One turn's appended block: message + tool-call + observation."""
    out_toks = max(16, int(rng.gauss(shade["output_tokens_mean"],
                                     shade["output_tokens_std"])))
    append_toks = int(out_toks * shade["append_out_ratio_mean"])
    if failure:
        append_toks = int(append_toks * shade["failure_inflation"])
    # Shared blocks reuse a fixed tag (cacheable across sessions); unique blocks
    # carry a session-specific nonce so they cannot be prefix-cache-shared.
    tag = "SHARED" if shared else f"UNIQ{block_idx}"
    filler_units = max(1, append_toks // 6)
    body = " ".join(f"{tag}_obs_token_{i}" for i in range(filler_units))
    return (f"\n## Turn {block_idx}\n"
            f"Thinking: analyzing state at step {block_idx}.\n"
            f"Tool call: read_file(path=module_{block_idx}.py)\n"
            f"Observation: {body}\n")


def build_session_prompt(prefix: str, rng: random.Random, shade: dict,
                         regime_cfg: dict, condition: str,
                         target_ctx_tokens: int) -> tuple[str, int]:
    """Grow history until prefix+history reaches target context. Returns
    (full_prompt, turns)."""
    n_turns = int(rng.gauss(shade["turns_mean"], shade["turns_std"]))
    n_turns = max(shade["turns_min"], min(shade["turns_max"], n_turns))
    history = []
    turn = 0
    base = prefix if regime_cfg["prefix_shared"] else (
        f"SESSION_UNIQUE_PREFIX_{rng.random()}\n" + prefix)
    while turn < n_turns:
        shared = rng.random() < regime_cfg["history_shared_frac"]
        history.append(make_history_block(rng, shade, shared,
                                          condition == "failure", turn))
        turn += 1
        if approx_tokens(base + "".join(history)) >= target_ctx_tokens:
            break
    return base + "".join(history), turn


def scrape_prefix_metrics(metrics_url: str) -> tuple[int, int]:
    """Return (prefix_cache_hits, prefix_cache_queries) from vLLM /metrics."""
    with urllib.request.urlopen(metrics_url, timeout=5) as r:
        text = r.read().decode()
    hits = queries = 0
    for line in text.splitlines():
        if line.startswith("vllm:prefix_cache_hits"):
            hits = int(float(line.rsplit(" ", 1)[1]))
        elif line.startswith("vllm:prefix_cache_queries"):
            queries = int(float(line.rsplit(" ", 1)[1]))
    return hits, queries


def send_completion(endpoint: str, model: str, prompt: str, max_tokens: int):
    body = json.dumps({"model": model, "prompt": prompt,
                       "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(endpoint + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def run(args) -> RunManifest:
    prefix, prefix_tokens = load_prefix(args.prefix_version)
    regime_cfg = REGIMES[args.regime]
    rng = random.Random(args.seed)
    man = RunManifest(
        regime=args.regime, condition=args.condition, rep=args.rep,
        seed=args.seed, prefix_version=args.prefix_version,
        prefix_tokens=prefix_tokens, target_context_tokens=args.target_context,
        n_sessions=args.n_sessions, hitrate_target=args.regime,
        sampling_params=dict(SHADE),
    )

    hits0 = queries0 = None
    if args.measure_hitrate:
        try:
            hits0, queries0 = scrape_prefix_metrics(args.metrics_url)
        except Exception as e:
            man.error = f"metrics scrape (pre) failed: {e}"

    total_turns = sent = 0
    final_hist_tokens = 0
    try:
        for sess in range(args.n_sessions):
            prompt, turns = build_session_prompt(
                prefix, rng, SHADE, regime_cfg, args.condition, args.target_context)
            total_turns += turns
            final_hist_tokens = approx_tokens(prompt) - prefix_tokens
            # Replay each turn as a re-send of prefix+history-so-far. Here we send
            # the full grown prompt once per session as the representative request;
            # --per-turn expands to one request per turn (heavier, optional).
            if args.per_turn:
                # rebuild incrementally would re-tokenize; approximate by sending
                # the final prompt `turns` times (same prefix -> cache hits)
                for _ in range(turns):
                    send_completion(args.endpoint, args.model, prompt, args.max_tokens)
                    sent += 1
            else:
                send_completion(args.endpoint, args.model, prompt, args.max_tokens)
                sent += 1
    except Exception as e:
        man.error = (man.error + " | " if man.error else "") + f"send failed: {e}"

    man.turns_generated = total_turns
    man.history_tokens_final = final_hist_tokens
    man.requests_sent = sent

    if args.measure_hitrate and hits0 is not None:
        try:
            time.sleep(1.0)  # let metrics settle
            hits1, queries1 = scrape_prefix_metrics(args.metrics_url)
            dh, dq = hits1 - hits0, queries1 - queries0
            man.prefix_cache_hits_delta = dh
            man.prefix_cache_queries_delta = dq
            man.hitrate_realized = (dh / dq) if dq > 0 else None
        except Exception as e:
            man.error = (man.error + " | " if man.error else "") + \
                        f"metrics scrape (post) failed: {e}"
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8000")
    ap.add_argument("--model", default="facebook/opt-125m")  # sim default
    ap.add_argument("--regime", choices=list(REGIMES), required=True)
    ap.add_argument("--condition", choices=["nominal", "failure"], default="nominal")
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefix-version", default="v1")
    ap.add_argument("--target-context", type=int, default=40000)
    ap.add_argument("--n-sessions", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--per-turn", action="store_true",
                    help="send one request per turn (heavier; default: one/session)")
    ap.add_argument("--measure-hitrate", action="store_true",
                    help="scrape Prometheus /metrics for realized hit-rate (SIM)")
    ap.add_argument("--metrics-url", default="http://localhost:8000/metrics")
    ap.add_argument("--out", default=None, help="manifest output path")
    args = ap.parse_args()

    man = run(args)
    out = Path(args.out) if args.out else Path(
        f"sim-results/manifest_{args.regime}_{args.condition}_rep{args.rep}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(man), indent=2))
    print(json.dumps(asdict(man), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
