#!/usr/bin/env python3
"""Orchestrator for the agentic-kv energy experiment (PROTOCOL.md).

Iterates the experimental matrix (regime x condition x rep), invokes the
deterministic workload generator per cell, and collects manifests into one
directory per run with a summary index. Resume-safe: cells whose manifest
already exists are skipped (matters on a paid GPU node).

Modes:
  --sim  validate generator calibration against llm-d-inference-sim
         (CPU-only: no inferscope, realized hit-rate scraped by the
         generator itself via --measure-hitrate).
  GPU    (default) the orchestrator OWNS the engine: launches bare vLLM
         (--enforce-eager, matrix invariant), resolves the EngineCore PID
         (pgid-scoped), brackets each cell with inferscope --sample-only
         (ADR-012) and scrapes the local /metrics (_total anchors).
         --sample-secs is REQUIRED and must exceed the slowest cell.
"""
import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent

# In a disaggregated P/D topology, per-pod counters MUST be scraped per pod
# and summed; scraping through the router samples a random backend and the
# window-delta comes out empty or dirty (observed on llmd-sim, 2026-07-05).
SIM_PODS = [
    ("llmd-sim", "llm-d.ai/role=prefill", 8000),
    ("llmd-sim", "llm-d.ai/role=decode", 8200),
]


def scrape_pods_prefix_counters(context):
    """Sum vllm:prefix_cache_{hits,queries} across all sim pods via the
    API-server pod proxy (no extra port-forwards)."""
    hits = queries = 0
    for ns, selector, port in SIM_PODS:
        name = subprocess.run(
            ["kubectl", "--context", context, "-n", ns, "get", "pod",
             "-l", selector, "-o", "jsonpath={.items[0].metadata.name}"],
            capture_output=True, text=True, check=True).stdout.strip()
        raw = subprocess.run(
            ["kubectl", "--context", context, "get", "--raw",
             f"/api/v1/namespaces/{ns}/pods/{name}:{port}/proxy/metrics"],
            capture_output=True, text=True, check=True).stdout
        for line in raw.splitlines():
            if line.startswith("vllm:prefix_cache_hits{"):
                hits += int(float(line.rsplit(" ", 1)[1]))
            elif line.startswith("vllm:prefix_cache_queries{"):
                queries += int(float(line.rsplit(" ", 1)[1]))
    return hits, queries


GPU_READY_PATH = "/health"


def scrape_local_prefix_counters(metrics_url):
    """Read vllm:prefix_cache_{hits,queries}_total from the local engine
    /metrics (GPU mode: single engine, no multi-pod sum).

    Anchors carry the exact `_total{` suffix: vLLM v0.23.0 defines these
    counters as prometheus_client Counters (v1/metrics/loggers.py:542-559,
    _counter_cls = Counter at :406) and prometheus_client exposes Counter
    samples as `<name>_total` (verified empirically via generate_latest,
    2026-07-05). A loose startswith("vllm:prefix_cache_hits") would ALSO
    match `vllm:prefix_cache_hits_created{` — a unix-timestamp gauge
    (~1.78e9) — and silently corrupt the sum. The sim fixture exposes the
    names WITHOUT `_total`; the two scrapers are intentionally separate.
    """
    hits = queries = 0
    with urllib.request.urlopen(metrics_url, timeout=10) as resp:
        raw = resp.read().decode()
    for line in raw.splitlines():
        if line.startswith("vllm:prefix_cache_hits_total{"):
            hits += int(float(line.rsplit(" ", 1)[1]))
        elif line.startswith("vllm:prefix_cache_queries_total{"):
            queries += int(float(line.rsplit(" ", 1)[1]))
    return hits, queries


def launch_engine(args):
    """Launch vLLM as a bare host process in its own session (decision
    2026-07-05: bare `pip vllm==0.23.0` over the docker digest pin — the
    pgid-scoped PID discovery and the per-PID NVML attach require a host
    process; reproducibility pin = pip version + `pip freeze` recorded in
    provenance). The orchestrator owns the engine for the whole matrix; no
    restart between cells (H0 isolation via seed-disjoint prefixes).
    --enforce-eager is hardcoded: matrix invariant per PROTOCOL."""
    port = urllib.parse.urlsplit(args.endpoint).port or 8000
    vllm_bin = shutil.which("vllm") or str(Path(sys.executable).parent / "vllm")
    if not Path(vllm_bin).exists():
        sys.exit(f"vllm binary not found (PATH and {Path(sys.executable).parent}); "
                 "activate the venv or fix PATH")
    cmd = [vllm_bin, "serve", args.model,
           "--port", str(port),
           "--enforce-eager"]
    cmd += shlex.split(args.engine_args)
    log_path = Path(args.out_dir) / "engine.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    return proc, cmd


def wait_ready(endpoint, timeout_s, expect_model=None):
    """Ready when /health is 200 AND, if expect_model is given, /v1/models
    lists it. F11 (rehearsal 2026-07-10): a stale listener on the port
    (zombie engine, forgotten port-forward) answered /health for a process
    that was NOT ours — cells would then talk to an engine inferscope is
    not monitoring (energy attributed to an idle PID)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(endpoint + GPU_READY_PATH,
                                        timeout=5) as r:
                if r.status != 200:
                    raise OSError("health != 200")
            if expect_model is not None:
                with urllib.request.urlopen(endpoint + "/v1/models",
                                            timeout=5) as r:
                    ids = [m.get("id") for m in
                           json.loads(r.read()).get("data", [])]
                if expect_model not in ids:
                    raise OSError(f"identity mismatch: {ids}")
            return True
        except Exception:
            pass
        time.sleep(2)
    return False


def find_enginecore_pid(api_proc, retries=30, delay_s=1.0):
    """Resolve the EngineCore PID for THIS run, scoped to its process group
    (faithful copy of cuda-graphs run_experiment_tight.py — attaching to the
    APIServer parent yields near-idle GPU samples: no CUDA context, the exact
    artefact the cuda-graphs energy re-run fixed)."""
    try:
        pgid = os.getpgid(api_proc.pid)
    except Exception:
        return None
    for _ in range(retries):
        out = subprocess.run(
            ["pgrep", "-g", str(pgid), "-f", "VLLM::EngineCore"],
            capture_output=True, text=True)
        pids = [int(x) for x in out.stdout.split()]
        if pids:
            return pids[0]
        time.sleep(delay_s)
    return None


def stop_engine(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


# Index maps for multi-axis seed derivation (prime-weighted, so no
# (nonce, regime, condition, rep) tuples collide; the old additive form
# seed_base+rep+nonce collided across campaigns: 20260709+3 == 20260710+2).
REGIME_IDX = {"H0": 0, "H1": 1, "H2": 2}
COND_IDX = {"nominal": 0, "failure": 1}


ATTACH_OFFSET_S = 0.5  # NVML window opens before the first token (cuda-graphs pattern);
                       # recorded in manifest for analysis alignment.


def run_cell(args, regime, condition, rep, scrape, gpu_ctx=None):
    cell = f"{regime}_{condition}_rep{rep}"
    cell_dir = Path(args.out_dir) / cell
    manifest = cell_dir / "manifest.json"
    if manifest.exists():
        print(f"[skip] {cell}: manifest exists")
        return json.loads(manifest.read_text())
    cell_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(REPO / "agentic_workload.py"),
        "--endpoint", args.endpoint,
        "--model", args.model,
        "--regime", regime,
        "--condition", condition,
        "--rep", str(rep),
        "--seed", str(args.seed_base
                      + args.run_nonce * 1000003
                      + REGIME_IDX[regime] * 10007
                      + COND_IDX[condition] * 101
                      + rep),
        "--prefix-version", args.prefix_version,
        "--target-context", str(args.target_context),
        "--n-sessions", str(args.n_sessions),
        "--out", str(manifest),
    ]
    if gpu_ctx is not None:
        # GPU mode: history growth counted with the real tokenizer of the
        # served model; the engine enforces prompt+max_tokens <= max_model_len
        # (smoke 2026-07-09: word-proxy undercounted hex filler ~3.37x -> 400).
        cmd.append("--bpe-counter")
    # hit-rate is measured by the orchestrator (sim: per-pod sum; GPU:
    # local single-engine /metrics); generator --measure-hitrate NOT used.
    print(f"[run ] {cell}")
    t0 = time.time()
    h0, q0 = scrape()
    infer_proc = None
    infer_json_path = cell_dir / "inferscope.json"
    if gpu_ctx is not None:
        # inferscope --sample-only is a fixed-duration timer with no signal
        # handling: launched BEFORE the generator, brackets it, exits on its
        # own timer; JSON collected from stdout via communicate().
        infer_cmd = [
            gpu_ctx["inferscope_bin"],
            "--sample-only",
            "--pid", str(gpu_ctx["engine_pid"]),
            "--duration-secs", str(gpu_ctx["sample_secs"]),
            "--gpu",
            # ADR-012: Prometheus scrape on the same window as the NVML
            # attach -> per-phase energy alongside whole-window energy.
            "--metrics-endpoint", args.metrics_url,
            "--model", args.model,
            "--json",
        ]
        infer_proc = subprocess.Popen(infer_cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True)
        time.sleep(ATTACH_OFFSET_S)
    gen_t0 = time.time()
    gen_env = dict(os.environ)
    if gpu_ctx is not None:
        # F2: tokenizer strictly from local cache (warmed by the engine
        # download); no hub round-trips mid-matrix (anonymous rate limits).
        gen_env["HF_HUB_OFFLINE"] = "1"
    proc = subprocess.run(cmd, capture_output=True, text=True, env=gen_env)
    gen_wall = time.time() - gen_t0
    (cell_dir / "generator.log").write_text(proc.stdout + proc.stderr)
    infer_ok = False
    if infer_proc is not None:
        try:
            infer_out, infer_err = infer_proc.communicate(
                timeout=gpu_ctx["sample_secs"] + 60)
            if infer_out and infer_out.strip():
                infer_json_path.write_text(infer_out)
                infer_ok = True
            else:
                (cell_dir / "inferscope.stderr").write_text(infer_err or "")
        except Exception:
            try:
                infer_proc.kill()
            except Exception:
                pass
    if proc.returncode != 0:
        print(f"[FAIL] {cell}: rc={proc.returncode}, see {cell_dir}/generator.log")
        return {"cell": cell, "error": f"generator rc={proc.returncode}"}
    m = json.loads(manifest.read_text())
    h1, q1 = scrape()
    dq = q1 - q0
    m["prefix_cache_hits_delta"] = h1 - h0
    m["prefix_cache_queries_delta"] = dq
    m["hitrate_realized"] = (h1 - h0) / dq if dq > 0 else None
    m["hitrate_scrape"] = ("per-pod sum (prefill+decode), orchestrator"
                           if args.sim else
                           "local vLLM /metrics (_total anchors), single engine")
    if gpu_ctx is not None:
        m["gpu"] = {
            "sample_secs": gpu_ctx["sample_secs"],
            "attach_offset_s": ATTACH_OFFSET_S,
            "generator_wall_s": round(gen_wall, 1),
            "engine_pid": gpu_ctx["engine_pid"],
            "inferscope_json": str(infer_json_path) if infer_ok else None,
        }
        if gen_wall > 0.9 * gpu_ctx["sample_secs"]:
            m["gpu"]["window_warning"] = (
                "generator filled >90% of the sample window; widen "
                "--sample-secs (a truncated window undercounts energy "
                "while tokens come from the full run -> tok/J inflated)")
            print(f"[WARN] {cell}: generator wall {gen_wall:.0f}s > 90% "
                  f"of window {gpu_ctx['sample_secs']}s")
    manifest.write_text(json.dumps(m, indent=2))
    print(f"[done] {cell} in {time.time() - t0:.0f}s — "
          f"hit target={m.get('hitrate_target')} realized={m.get('hitrate_realized')}")
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000")
    ap.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics")
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--regimes", default="H0,H1,H2")
    ap.add_argument("--conditions", default="nominal")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--prefix-version", default="v1")
    ap.add_argument("--target-context", type=int, default=32768)
    ap.add_argument("--n-sessions", type=int, default=20)
    ap.add_argument("--out-dir", default=str(REPO / "sim-results" / "calibration"))
    ap.add_argument("--kube-context", default="kind-llmd-sim")
    ap.add_argument("--sample-secs", type=int, default=None,
                    help="inferscope --sample-only window per cell (GPU mode, "
                         "REQUIRED there): fixed-duration timer, must exceed "
                         "the slowest cell or tok/J comes out inflated")
    ap.add_argument("--engine-args", default="",
                    help="extra vllm serve args, shlex-split verbatim "
                         "(--enforce-eager is hardcoded: matrix invariant)")
    ap.add_argument("--inferscope-bin", default="inferscope")
    ap.add_argument("--ready-timeout", type=int, default=900,
                    help="engine readiness timeout (32B load from cold cache)")
    ap.add_argument("--run-nonce", type=int,
                    default=int(time.time()) % 1000000 * 100,
                    help="offsets the seed so prompts are disjoint across "
                         "campaigns (warm sim/engine cache from a previous "
                         "campaign otherwise pollutes realized hit-rate)")
    args = ap.parse_args()

    # F14: an external SIGTERM (timeout(1), tmux kill-session, node agent)
    # kills CPython WITHOUT running finally -> stop_engine never fires and
    # the engine (own session, killpg-only reachable) leaks: on a GPU node
    # that is ~60GB of VRAM held and the next bind failing. Convert TERM
    # into SystemExit so the existing finally tears the engine down.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    engine_proc = None
    gpu_ctx = None
    if args.sim:
        scrape = lambda: scrape_pods_prefix_counters(args.kube_context)
    else:
        if args.sample_secs is None:
            sys.exit("GPU mode: --sample-secs is required (no default on "
                     "purpose — size it from the H1 calibration runs)")
        scrape = lambda: scrape_local_prefix_counters(args.metrics_url)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # F9: contract check BEFORE paying engine readiness. The binary
        # identity is version+commit+FEATURES: a build without gpu-nvidia
        # hides --gpu and NVML never runs (silent zero-energy campaign,
        # root-caused 2026-07-10).
        hp = subprocess.run([args.inferscope_bin, "--help"],
                            capture_output=True, text=True)
        if hp.returncode != 0 or "--gpu" not in hp.stdout:
            sys.exit("inferscope contract check FAILED: --gpu absent from "
                     "--help (binary built without gpu-nvidia feature?). "
                     "Rebuild: cargo build --release --features gpu-nvidia")
        probe = subprocess.run(
            [args.inferscope_bin, "--sample-only", "--pid", str(os.getpid()),
             "--duration-secs", "1", "--json"],
            capture_output=True, text=True)
        try:
            json.loads(probe.stdout)
        except Exception:
            sys.exit("inferscope contract check FAILED: dummy --sample-only "
                     "did not emit parseable JSON on stdout")
        engine_proc, engine_cmd = launch_engine(args)
        print(f"[gpu ] engine launched pid={engine_proc.pid}, waiting ready "
              f"(timeout {args.ready_timeout}s)")
        if not wait_ready(args.endpoint, args.ready_timeout,
                          expect_model=args.model):
            stop_engine(engine_proc)
            sys.exit(f"engine not ready in {args.ready_timeout}s, "
                     f"see {out_dir}/engine.log")
        engine_pid = find_enginecore_pid(engine_proc)
        if engine_pid is None:
            stop_engine(engine_proc)
            sys.exit("could not resolve EngineCore PID (attaching to the "
                     "APIServer parent would sample an idle GPU)")
        pip_freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True).stdout
        (out_dir / "pip-freeze.txt").write_text(pip_freeze)
        (out_dir / "engine.json").write_text(json.dumps({
            "engine_cmd": engine_cmd,
            "api_pid": engine_proc.pid,
            "engine_core_pid": engine_pid,
            "sample_secs": args.sample_secs,
            "seed_base": args.seed_base,
            "run_nonce": args.run_nonce,
            "pip_freeze": "pip-freeze.txt",
        }, indent=2))
        gpu_ctx = {
            "inferscope_bin": args.inferscope_bin,
            "engine_pid": engine_pid,
            "sample_secs": args.sample_secs,
        }
        print(f"[gpu ] EngineCore pid={engine_pid}, window={args.sample_secs}s")
        # Warm-up cell (root cause 2026-07-10): the FIRST request after
        # engine start misses the whole system prefix (14,785 tok for v1)
        # exactly once; the engine persists across cells, so the first
        # measured cell is depressed by ~prefix/queries (~0.06 on H1
        # calibration, enough to trip any sane divergence gate). Warm the
        # prefix with a short discarded cell BEFORE anything measured.
        # Reserved nonce offset keeps its prompts seed-disjoint from every
        # real cell of this and other campaigns.
        warm_ns = argparse.Namespace(**vars(args))
        warm_ns.run_nonce = args.run_nonce + 900001
        warm_ns.n_sessions = 2
        # target below the prefix size -> history_budget = 0 -> the warm-up
        # sends the bare prefix (verbatim, counter-independent). Growing
        # history here with the approx counter against the real engine
        # would rebuild the 2026-07-09 overflow (approx undercounts the
        # filler ~3.37x -> prompt past max_model_len -> HTTP 400).
        warm_ns.target_context = 1000
        warm_ns.out_dir = str(Path(args.out_dir) / "warmup")
        # F12: the warm-up must NOT be resume-safe. Its purpose is tied to
        # THIS engine start (cold prefix); a resumed campaign restarts the
        # engine, so a skipped warm-up re-creates the first-cell artefact
        # on the first resumed cell (verified in rehearsal 2026-07-10).
        import shutil as _sh
        _sh.rmtree(warm_ns.out_dir, ignore_errors=True)
        print("[warm] warm-up cell (discarded, prefix cache warming — "
              "re-run at EVERY engine start, never resumed)")
        wm = run_cell(warm_ns, "H1", "nominal", 1, scrape, gpu_ctx=None)
        wman = Path(warm_ns.out_dir) / "H1_nominal_rep1" / "manifest.json"
        if wman.exists():
            wj = json.loads(wman.read_text())
            wj["discarded"] = True
            wj["purpose"] = "prefix cache warm-up, excluded from all statistics"
            wman.write_text(json.dumps(wj, indent=2))
        if wm.get("error"):
            stop_engine(engine_proc)
            sys.exit(f"warm-up cell failed: {wm['error']} — aborting before matrix")

    try:
        summary = []
        for regime in args.regimes.split(","):
            for condition in args.conditions.split(","):
                for rep in range(1, args.reps + 1):
                    summary.append(run_cell(args, regime, condition, rep,
                                            scrape, gpu_ctx))
        idx = Path(args.out_dir) / "index.json"
        idx.write_text(json.dumps(summary, indent=2))
        print(f"index -> {idx}")
    finally:
        if engine_proc is not None:
            stop_engine(engine_proc)


if __name__ == "__main__":
    main()
