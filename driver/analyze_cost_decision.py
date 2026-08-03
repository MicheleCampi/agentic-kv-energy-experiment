#!/usr/bin/env python3
"""Decision arm for the A10 cost campaign.

Derives dimensionless occupancy figures from the `trajectory` section of
one or more inferscope reports. Two policies are evaluated on the same
measured trajectory:

  P1 -- per-segment release. Free the GPU during each tool segment
        longer than the re-entry price C (a measured cold start).
        Saving is sum over tool segments of max(0, d - C). With a cold
        start of ~18s and tool latencies in the seconds range this is
        expected to be ZERO on every cell. That is a result -- it
        falsifies the obvious policy -- not a failure of the script.

  P2 -- packing. Non-generating time is not freed, it is filled. The
        overlap bound 1/(1 - f_nongen) states how many trajectories one
        GPU could host before generating segments contend. It is an
        UPPER BOUND under declared non-interference: real continuous
        batching changes throughput, and the bound must never be quoted
        without that limit stated alongside it.

DENOMINATOR (load-bearing decision). The span used here is the
TRAJECTORY span -- last step end minus first step start -- NOT
`run_duration_ns`, which is the span of the sampling window. On the
2026-07-21 A10 evidence the two differ by roughly 8x because sampling
began 3.9s before the first step: with the window as denominator the
non-generating fraction becomes an artefact of window sizing rather
than a property of the workload. `run_duration_ns` is read when present
and reported only as a diagnostic of window excess.

No dollar rate enters this script. Every figure is dimensionless and so
independent of the occupancy rate, which is what makes cells
comparable. The priced figure comes from `inferscope cost` on the node.
"""

import argparse
import json
import sys
from pathlib import Path

NS = 1e9


def load_trajectory(path):
    """Return (trajectory_dict, None) or (None, reason)."""
    try:
        report = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        return None, f"unreadable: {exc}"
    traj = report.get("trajectory")
    if traj is None:
        return None, "no trajectory section (withheld: no GPU basis?)"
    steps = traj.get("steps") or []
    if len(steps) < 2:
        return None, f"only {len(steps)} step(s): no span to measure"
    return traj, None


def analyze(traj, reentry_ns):
    steps = sorted(traj["steps"], key=lambda s: s["start_elapsed_ns"])
    span = steps[-1]["end_elapsed_ns"] - steps[0]["start_elapsed_ns"]
    if span <= 0:
        raise ValueError("non-positive trajectory span")

    def dur(s):
        return s["end_elapsed_ns"] - s["start_elapsed_ns"]

    tool_durs = [dur(s) for s in steps if s["kind"] == "tool"]
    llm_durs = [dur(s) for s in steps if s["kind"] == "llm_call"]
    gaps = [b["start_elapsed_ns"] - a["end_elapsed_ns"]
            for a, b in zip(steps, steps[1:])]

    tool_ns, llm_ns, gap_ns = sum(tool_durs), sum(llm_durs), sum(gaps)

    # Reconciliation: the three parts must tile the span exactly. A
    # mismatch means overlapping steps, which ADR-013 is supposed to
    # have dropped -- surface it, never absorb it.
    residual = span - (tool_ns + llm_ns + gap_ns)
    overlaps = sum(1 for g in gaps if g < 0)

    nongen_ns = tool_ns + gap_ns
    f_nongen = nongen_ns / span
    packing = 1.0 / (1.0 - f_nongen) if f_nongen < 1.0 else float("inf")

    # P1: per-segment release against the declared re-entry price.
    p1_eligible = [d for d in tool_durs if d > reentry_ns]
    p1_saving_ns = sum(d - reentry_ns for d in p1_eligible)

    run_ns = traj.get("run_duration_ns")
    window_excess = (run_ns / span) if run_ns else None

    return {
        "n_steps": len(steps),
        "n_llm": len(llm_durs),
        "n_tool": len(tool_durs),
        "span_s": span / NS,
        "llm_s": llm_ns / NS,
        "tool_s": tool_ns / NS,
        "gap_s": gap_ns / NS,
        "gap_share": gap_ns / span,
        "f_nongen": f_nongen,
        "packing_bound": packing,
        "p1_segments_eligible": len(p1_eligible),
        "p1_saving_s": p1_saving_ns / NS,
        "p1_saving_share": p1_saving_ns / span,
        "reconciliation_residual_ns": residual,
        "overlapping_pairs": overlaps,
        "dropped_steps": len(traj.get("dropped_steps") or []),
        "run_duration_ns": run_ns,
        "window_excess_factor": window_excess,
        "mean_tool_s": (tool_ns / len(tool_durs) / NS) if tool_durs else None,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Occupancy decision arm over inferscope trajectories.")
    ap.add_argument("reports", nargs="+", help="inferscope report JSON files")
    ap.add_argument(
        "--reentry-secs", type=float, required=True,
        help="re-entry price C in seconds: a MEASURED cold start. No "
             "default on purpose -- the value must be visible in the "
             "command that produced the numbers (vllm-coldstart-probe: "
             "~18s; see also the 27s/96s variance finding).")
    ap.add_argument("--json-out", help="write per-report results as JSON")
    args = ap.parse_args()

    reentry_ns = int(args.reentry_secs * NS)
    results, failures = [], []

    for path in args.reports:
        traj, reason = load_trajectory(path)
        if traj is None:
            failures.append((path, reason))
            continue
        row = analyze(traj, reentry_ns)
        row["report"] = Path(path).name
        results.append(row)

    if not results:
        print("no usable report", file=sys.stderr)
        for path, reason in failures:
            print(f"  {Path(path).name}: {reason}", file=sys.stderr)
        return 1

    print(f"re-entry price C = {args.reentry_secs:.1f}s")
    print()
    hdr = (f"{'report':<34} {'span':>7} {'llm':>7} {'tool':>7} "
           f"{'f_nongen':>9} {'packing':>8} {'P1 save':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['report'][:34]:<34} {r['span_s']:>6.3f}s "
              f"{r['llm_s']:>6.3f}s {r['tool_s']:>6.3f}s "
              f"{100 * r['f_nongen']:>8.2f}% {r['packing_bound']:>8.2f} "
              f"{r['p1_saving_s']:>8.3f}s")

    print()
    print("integrity:")
    for r in results:
        flags = []
        if r["reconciliation_residual_ns"] != 0:
            flags.append(f"residual={r['reconciliation_residual_ns']}ns")
        if r["overlapping_pairs"]:
            flags.append(f"overlaps={r['overlapping_pairs']}")
        if r["dropped_steps"]:
            flags.append(f"dropped={r['dropped_steps']}")
        if r["window_excess_factor"]:
            flags.append(f"window/span={r['window_excess_factor']:.2f}x")
        else:
            flags.append("run_duration_ns absent (pre-ADR-015 report)")
        flags.append(f"gaps={100 * r['gap_share']:.2f}% of span")
        print(f"  {r['report'][:34]:<34} {', '.join(flags)}")

    if failures:
        print()
        print("skipped:")
        for path, reason in failures:
            print(f"  {Path(path).name}: {reason}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print()
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
