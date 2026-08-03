"""Shared gates and meta-file writing for both trajectory arms.

Two entry points emit the same ADR-013 steps-file through the same
StepsFileCallback: run_trajectory.py drives a real Deep Agents loop,
run_replay.py drives a declared structure. They must be judged by
identical criteria — a gate that differs between the arms would make the
anchoring cells unusable as evidence that the replay is representative.
Hence one implementation, imported by both.
"""
import json
import sys


def check_and_write_meta(steps_file, open_runs, extra_meta, final_text=None):
    """Validate a completed steps-file and write its meta sidecar.

    Returns a process exit code: 0 past all gates, 1 on any failure. The
    meta file is written only past the gates, so its existence is itself
    the assertion that the trajectory it describes is valid.
    """
    if open_runs != 0:
        print(f"FAIL: {open_runs} unclosed run(s) — boundary anomaly",
              file=sys.stderr)
        return 1
    with open(steps_file, encoding="utf-8") as f:
        steps = [json.loads(line) for line in f]
    if not steps:
        print("FAIL: empty steps-file", file=sys.stderr)
        return 1
    kinds = [s["kind"] for s in steps]
    n_llm, n_tool = kinds.count("llm_call"), kinds.count("tool")
    overlaps = sum(
        1 for a, b in zip(steps, steps[1:])
        if b["t_start_unix_ns"] < a["t_end_unix_ns"]
    )
    bad_span = sum(1 for s in steps if s["t_end_unix_ns"] <= s["t_start_unix_ns"])
    print(f"steps: {len(steps)} (llm_call={n_llm}, tool={n_tool})")
    print(f"overlapping consecutive segments: {overlaps}")
    print(f"non-positive spans: {bad_span}")
    if final_text is not None:
        print(f"final message: {final_text[:200]!r}")
    if n_llm < 2 or n_tool < 2:
        print("FAIL: trajectory too short — need >=2 llm_call and >=2 tool steps",
              file=sys.stderr)
        return 1
    if bad_span:
        print("FAIL: non-positive span(s)", file=sys.stderr)
        return 1
    if overlaps:
        print("FAIL: overlapping segments — serialization broken", file=sys.stderr)
        return 1
    # observed_span_s is what sizes the sampling window of a cell, and
    # the window is not correctable after the fact — see ADR-013.
    meta = dict(extra_meta)
    meta.update({
        "steps_file": steps_file,
        "n_steps": len(steps),
        "n_llm_call": n_llm,
        "n_tool": n_tool,
        "t_start_unix_ns": steps[0]["t_start_unix_ns"],
        "t_end_unix_ns": steps[-1]["t_end_unix_ns"],
        "observed_span_s": (steps[-1]["t_end_unix_ns"]
                            - steps[0]["t_start_unix_ns"]) / 1e9,
    })
    if "tool_latency_s" in meta:
        meta["tool_wall_s"] = n_tool * meta["tool_latency_s"]
    meta_path = steps_file + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    span = meta["observed_span_s"]
    if "tool_wall_s" in meta:
        print(f"span: {span:.1f}s (tool wall {meta['tool_wall_s']:.1f}s at "
              f"{meta['tool_latency_s']}s/tool, "
              f"{100 * meta['tool_wall_s'] / span:.1f}% of span)")
    else:
        print(f"span: {span:.1f}s")
    print(f"meta: {meta_path}")
    return 0
