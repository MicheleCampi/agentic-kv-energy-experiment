#!/usr/bin/env python3
"""One cell of the cost campaign: bracket a replay trajectory with
inferscope and derive its price on the node.

Phase 2 of the experiment. Phase 1 (the hit-rate matrix) is
run_experiment.py; the two orchestrators do not merge, because ADR-013
wants one trajectory in flight and the matrix wants N concurrent
sessions.

Two parameters are irreversible per cell and neither is correctable
afterwards: the sampling window, which must be sized from a measured
span, and the steps-file, which inferscope reads once after the run and
never re-joins. Both are printed by --dry-run before anything is spent.

The cell directory is the archivable unit: steps-file, its .meta.json,
the inferscope report, the derived cost and the argv that produced them
all live together. Anything less and the report cannot be re-analysed --
the trajectory inside it is already joined, so a join defect is no longer
diagnosable without its input.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inferscope_contract import check_inferscope, ContractError  # noqa: E402

# Identical to run_experiment.py: a cell that attaches with a different
# offset is not comparable with the matrix. NVML's window opens before
# the first token (cuda-graphs pattern).
ATTACH_OFFSET_S = 0.5

NS = 1_000_000_000
SENTINEL_ID_BASE = 900001


def write_sentinel(path):
    """A readable steps-file placed before inferscope starts.

    inferscope reads --steps-file after the run and treats an absent file
    as fatal: exit 1, empty stdout, and the sampled energy lost with it.
    If the replay dies, this file is what the cell falls back to, so the
    report still carries the window's resource and GPU sections.

    Its anchors are one hour in the past and its ids start at 900001, so
    it cannot be mistaken for a measurement: inferscope abstains on steps
    outside the sampling window (verified 2026-08-04, exit 0 with
    trajectory: None), and the gates never write a .meta.json for it.
    """
    t0 = time.time_ns() - 3600 * NS
    shape = [("llm_call", 0, 1), ("tool", 1, 2), ("llm_call", 2, 3), ("tool", 3, 4)]
    with open(path, "w", encoding="utf-8") as f:
        for i, (kind, a, b) in enumerate(shape, start=SENTINEL_ID_BASE):
            f.write(json.dumps({
                "step_id": i,
                "kind": kind,
                "t_start_unix_ns": t0 + a * NS,
                "t_end_unix_ns": t0 + b * NS,
            }) + "\n")


def replay_argv(a, cell, steps_path):
    """argv of the measurement arm. Fixed structure, tool latency swept."""
    return [
        str(a.python), str(Path(a.driver) / "run_replay.py"),
        "--base-url", a.base_url,
        "--model", a.model,
        "--steps-file", str(steps_path),
        "--tool-latency-s", str(a.tool_latency_s),
        "--tool-latency-cv", str(a.tool_latency_cv),
        "--n-llm", str(a.n_llm),
        "--n-tool", str(a.n_tool),
        "--max-tokens", str(a.max_tokens),
        "--seed", str(a.seed),
    ]


def inferscope_argv(a, steps_path):
    """argv of the measurement. --duration-secs and --steps-file are the
    two irreversible parameters: the window cannot be resized afterwards
    and the join happens once, in flight."""
    argv = [
        a.inferscope_bin, "--sample-only",
        "--pid", str(a.engine_pid),
        "--duration-secs", str(a.window_secs),
        "--steps-file", str(steps_path),
        "--gpu",
        "--json",
    ]
    if a.metrics_url:
        # ADR-014 D6: the vocabulary is declared, never inferred.
        argv += ["--metrics-endpoint", a.metrics_url,
                 "--engine", a.engine, "--model", a.model]
    return argv


def cost_argv(a, report_path):
    """argv of the price derivation. Run per cell ON THE NODE, never at
    the end of the campaign: an abstention by `cost` is only diagnosable
    while the GPU that produced the report is still there."""
    argv = [a.inferscope_bin, "cost", "--report", str(report_path)]
    if a.usd_per_hour is not None:
        argv += ["--usd-per-hour", str(a.usd_per_hour)]
    else:
        argv += ["--usd-per-kwh", str(a.usd_per_kwh)]
    return argv


def decision_argv(a, report_path, out_path):
    """argv of the decision arm. Dimensionless, so independent of the
    declared rate: P1 releases on tool segments longer than the measured
    re-entry price, P2 is the packing bound 1/(1-f_nongen)."""
    return [
        str(a.python), str(Path(a.driver) / "analyze_cost_decision.py"),
        str(report_path),
        "--reentry-secs", str(a.reentry_secs),
        "--json-out", str(out_path),
    ]


def run_cell(a, cell_dir):
    """Execute one cell. Returns 0 on a usable cell, 1 otherwise."""
    cell_dir.mkdir(parents=True, exist_ok=True)
    steps_path = cell_dir / "steps.jsonl"
    report_path = cell_dir / "inferscope.json"
    argv = {
        "replay": replay_argv(a, cell_dir, steps_path),
        "inferscope": inferscope_argv(a, steps_path),
        "cost": cost_argv(a, report_path),
        "decision": decision_argv(a, report_path, cell_dir / "decision.json"),
    }
    (cell_dir / "argv.json").write_text(json.dumps(argv, indent=1))

    write_sentinel(steps_path)
    infer = subprocess.Popen(argv["inferscope"], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    time.sleep(ATTACH_OFFSET_S)

    t0 = time.time()
    replay = subprocess.run(argv["replay"], capture_output=True, text=True)
    replay_wall = time.time() - t0
    (cell_dir / "replay.log").write_text(replay.stdout + replay.stderr)
    # The gates write the .meta.json only past all of them, so its
    # existence IS the assertion that the trajectory is valid. A failed
    # replay leaves the sentinel in place and the cell stays diagnosable.
    meta_ok = Path(str(steps_path) + ".meta.json").exists()
    if replay.returncode != 0 or not meta_ok:
        print(f"[cell] replay FAILED rc={replay.returncode} meta={meta_ok} "
              f"— sentinel steps-file left in place, see replay.log")

    try:
        out, err = infer.communicate(timeout=a.window_secs + 60)
    except subprocess.TimeoutExpired:
        infer.kill()
        out, err = infer.communicate()
        print("[cell] inferscope did not exit on its own timer")
    if not (out or "").strip():
        (cell_dir / "inferscope.stderr").write_text(err or "")
        print(f"[cell] ABORT: no report on stdout (rc={infer.returncode}), "
              f"stderr in {cell_dir}/inferscope.stderr")
        return 1
    report_path.write_text(out)

    rep = json.loads(out)
    # Both are abort criteria on the node and both are EXPECTED to be
    # absent off-node: gpu is null without NVML, and the trajectory
    # abstains on absence when there is no GPU timeline to attribute.
    if rep.get("gpu") is None:
        print("[cell] ABORT: \"gpu\": null on a GPU node — NVML did not attach")
        return 1
    if rep.get("trajectory") is None:
        print("[cell] ABORT: no trajectory section — the join found nothing "
              "in the window (steps outside it, or the replay never ran)")
        return 1

    for name in ("cost", "decision"):
        r = subprocess.run(argv[name], capture_output=True, text=True)
        (cell_dir / f"{name}.log").write_text(r.stdout + r.stderr)
        if r.returncode != 0:
            print(f"[cell] {name} exited {r.returncode}, see {name}.log")
    print(f"[cell] ok — replay {replay_wall:.1f}s in a {a.window_secs}s window "
          f"({100 * replay_wall / a.window_secs:.0f}% filled) -> {cell_dir}")
    return 0


def main():
    p = argparse.ArgumentParser(
        description="One cell of the cost campaign (phase 2).")
    p.add_argument("--out-dir", required=True,
                   help="cell directory: the archivable unit")
    p.add_argument("--engine-pid", type=int,
                   help="EngineCore PID to attach to. Required unless --dry-run")
    p.add_argument("--window-secs", type=int,
                   help="sampling window, WHOLE seconds: inferscope's "
                        "--duration-secs rejects a decimal point, and a "
                        "float here would print as an integer in --dry-run "
                        "only when it happened to be whole (found "
                        "2026-08-04, node off). Size it from a MEASURED "
                        "span, never guess it: it is not correctable after "
                        "the fact")
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--metrics-url", default="")
    p.add_argument("--engine", default="vllm", choices=["vllm", "sglang"])
    p.add_argument("--inferscope-bin",
                   default=os.environ.get("EXP_INFERSCOPE_BIN", "inferscope"))
    p.add_argument("--driver", default=str(Path(__file__).resolve().parent))
    p.add_argument("--python",
                   default=str(Path(__file__).resolve().parent / ".venv/bin/python"))
    p.add_argument("--tool-latency-s", type=float, required=True)
    p.add_argument("--tool-latency-cv", type=float, default=0.0)
    p.add_argument("--n-llm", type=int, default=4)
    p.add_argument("--n-tool", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reentry-secs", type=float, required=True,
                   help="re-entry price C in seconds: a MEASURED cold start "
                        "(vllm-coldstart-probe: ~18s)")
    p.add_argument("--usd-per-hour", type=float)
    p.add_argument("--usd-per-kwh", type=float)
    p.add_argument("--dry-run", action="store_true",
                   help="print the argv of every stage and exit, spending "
                        "nothing")
    a = p.parse_args()

    if (a.usd_per_hour is None) == (a.usd_per_kwh is None):
        sys.exit("exactly one of --usd-per-hour / --usd-per-kwh (ADR-015 D2: "
                 "one basis per derivation, summing them double-counts)")
    cell_dir = Path(a.out_dir)

    if a.dry_run:
        a.engine_pid = a.engine_pid or 0
        a.window_secs = a.window_secs if a.window_secs is not None else 0
        steps_path = cell_dir / "steps.jsonl"
        for name, argv in (
                ("inferscope", inferscope_argv(a, steps_path)),
                ("replay", replay_argv(a, cell_dir, steps_path)),
                ("cost", cost_argv(a, cell_dir / "inferscope.json")),
                ("decision", decision_argv(a, cell_dir / "inferscope.json",
                                           cell_dir / "decision.json"))):
            print(f"--- {name}\n{' '.join(argv)}\n")
        if not a.window_secs:
            print("NOTE: --window-secs unset, shown as 0. On the node it is "
                  "the calibration span x the margin, and it is irreversible.")
        return 0

    if a.engine_pid is None or a.window_secs is None:
        sys.exit("--engine-pid and --window-secs are required unless --dry-run")
    try:
        check_inferscope(a.inferscope_bin, a.model)
    except ContractError as e:
        sys.exit(f"inferscope contract check FAILED: {e}")
    return run_cell(a, cell_dir)


if __name__ == "__main__":
    sys.exit(main())
