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
  GPU mode (energy capture via inferscope --sample-only, ADR-012) lands
  in the next block; invoking without --sim exits explicitly.
"""
import argparse
import json
import subprocess
import sys
import time
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


# Index maps for multi-axis seed derivation (prime-weighted, so no
# (nonce, regime, condition, rep) tuples collide; the old additive form
# seed_base+rep+nonce collided across campaigns: 20260709+3 == 20260710+2).
REGIME_IDX = {"H0": 0, "H1": 1, "H2": 2}
COND_IDX = {"nominal": 0, "failure": 1}


def run_cell(args, regime, condition, rep):
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
    # hit-rate is measured by the orchestrator per-pod (see above); the
    # generator's --measure-hitrate (router scrape) is NOT used.
    print(f"[run ] {cell}")
    t0 = time.time()
    h0 = q0 = None
    if args.sim:
        h0, q0 = scrape_pods_prefix_counters(args.kube_context)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (cell_dir / "generator.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        print(f"[FAIL] {cell}: rc={proc.returncode}, see {cell_dir}/generator.log")
        return {"cell": cell, "error": f"generator rc={proc.returncode}"}
    m = json.loads(manifest.read_text())
    if args.sim and h0 is not None:
        h1, q1 = scrape_pods_prefix_counters(args.kube_context)
        dq = q1 - q0
        m["prefix_cache_hits_delta"] = h1 - h0
        m["prefix_cache_queries_delta"] = dq
        m["hitrate_realized"] = (h1 - h0) / dq if dq > 0 else None
        m["hitrate_scrape"] = "per-pod sum (prefill+decode), orchestrator"
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
    ap.add_argument("--run-nonce", type=int,
                    default=int(time.time()) % 1000000 * 100,
                    help="offsets the seed so prompts are disjoint across "
                         "campaigns (warm sim/engine cache from a previous "
                         "campaign otherwise pollutes realized hit-rate)")
    args = ap.parse_args()

    if not args.sim:
        sys.exit("GPU mode not implemented yet — run with --sim "
                 "(energy capture lands in the next block)")

    summary = []
    for regime in args.regimes.split(","):
        for condition in args.conditions.split(","):
            for rep in range(1, args.reps + 1):
                summary.append(run_cell(args, regime, condition, rep))
    idx = Path(args.out_dir) / "index.json"
    idx.write_text(json.dumps(summary, indent=2))
    print(f"index -> {idx}")


if __name__ == "__main__":
    main()
