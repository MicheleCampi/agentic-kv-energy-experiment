#!/usr/bin/env python3
"""Drive N replay trajectories concurrently and sample the engine's
running-request count (ADR-0010 in vllm-coldstart-operator).

The prediction under test: N trajectories each idle for f_nongen of their
span should hold the engine at a time-averaged `num_requests_running` of
N * (1 - f_nongen), not N. Both sides are measured here — the left from
the engine's own Prometheus endpoint, the right from the steps-files the
replays write.

WINDOW, declared before the first run: the measurement window is the
interval in which ALL N trajectories are simultaneously in flight, i.e.
[max(start_i), min(end_i)] over the N steps-files, resolved offline once
they exist. Not the union. In the tail where one trajectory has finished
and the others have not, true concurrency is N-1, and averaging over it
would depress the running count for a reason that belongs to the choice
of window rather than to the engine. That is the same defect class as
dividing by the sampling window instead of the trajectory span, which is
what the cost campaign had to correct for.

The sampler therefore runs for the whole wall-clock and timestamps every
sample; the trim happens in analysis. The full series stays in the
evidence, and the gap between the trimmed and untrimmed reading is itself
reported rather than hidden.

This experiment CANNOT use ADR-013 per-step attribution: with N
trajectories overlapping, a sampled instant has no unique owner. Observables
are engine-side only.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

NS = 1_000_000_000


def scrape_running(metrics_url, timeout=2.0):
    """Sum `vllm:num_requests_running` across the endpoint's series.

    Returns None on any failure: absence is not zero, exactly as the
    operator's NodeState contract states. A failed scrape that read as
    0.0 would look like an idle engine and pull the mean down.
    """
    try:
        with urllib.request.urlopen(metrics_url, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
    except Exception:
        return None
    # The metric name must end where the sample does: vLLM emits
    # `vllm:num_requests_running{engine="0",model_name="..."} 2.0`, and a
    # bare startswith would also match a hypothetical
    # `vllm:num_requests_running_total` and silently sum it in. The next
    # character after the name has to be `{` or whitespace.
    total, seen = 0.0, False
    prefix = "vllm:num_requests_running"
    for line in body.splitlines():
        if line.startswith("#") or not line.startswith(prefix):
            continue
        rest = line[len(prefix):]
        if rest and rest[0] not in "{ \t":
            continue
        try:
            total += float(line.rsplit(None, 1)[1])
            seen = True
        except (IndexError, ValueError):
            continue
    return total if seen else None


def replay_argv(a, idx, steps_path):
    """argv of one replay arm. Seeds differ per trajectory so the N are
    not bit-identical requests the engine could collapse; everything
    else is the fixed structure the cost campaign pinned.

    With --heterogeneous, the shape varies per trajectory too. Real agents
    are not clones of each other: they run different numbers of steps and
    generate different amounts. Identical trajectories are the strongest
    case for staggering and the weakest for lockstep, because clones cannot
    drift apart on their own — so measuring only them overstates what an
    admission policy would buy.

    The variation is deterministic in idx, not random: the same idx always
    produces the same shape, so an arm is reproducible from its argv alone.
    """
    n_llm, n_tool, max_tokens = a.n_llm, a.n_tool, a.max_tokens
    if a.heterogeneous:
        # Spread around the homogeneous baseline (4 LLM calls, 3 tools, 192
        # tokens) so the mean workload per trajectory is unchanged and only
        # its dispersion differs. Cycling on idx keeps totals stable across
        # N, which is what makes arms at different N comparable at all.
        n_llm = (3, 4, 5, 4)[idx % 4]
        n_tool = n_llm - 1
        max_tokens = (128, 192, 256, 192)[idx % 4]
    return [
        str(a.python), str(Path(a.driver) / "run_replay.py"),
        "--base-url", a.base_url,
        "--model", a.model,
        "--steps-file", str(steps_path),
        "--tool-latency-s", str(a.tool_latency_s),
        "--tool-latency-cv", str(a.tool_latency_cv),
        "--n-llm", str(n_llm),
        "--n-tool", str(n_tool),
        "--max-tokens", str(max_tokens),
        "--seed", str(a.seed_base + idx),
        "--request-timeout-s", str(a.request_timeout_s),
        "--max-retries", str(a.max_retries),
        "--retry-backoff-s", str(a.retry_backoff_s),
    ]


def run_arm(a, out_dir):
    """Launch N replays, sample the engine while they run, write evidence.

    Returns 0 when every replay passed its gates, 1 otherwise. A failed
    replay is not silently averaged over: with N-1 trajectories the arm
    is a different experiment.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = [out_dir / f"steps-{i}.jsonl" for i in range(a.n)]
    argv = {f"replay-{i}": replay_argv(a, i, steps[i]) for i in range(a.n)}
    (out_dir / "argv.json").write_text(json.dumps(argv, indent=1))

    samples = []
    t_start = time.time_ns()
    # Start offset (staggered-start experiment). Replica i starts at
    # i * start_offset_s. Zero reproduces the lockstep condition ADR-0010
    # measured, where the raw series never showed running=1 because the
    # trajectories generated together and waited together.
    #
    # The sleep sits between the Popen calls rather than inside the replay
    # driver so that nothing about the trajectory itself changes: same
    # script, same seeds, same request pattern. The only difference between
    # the arms is when each process begins, which is what the experiment is
    # about.
    # Arrival process. `fixed` spaces starts evenly, which is a policy a
    # scheduler could implement. `poisson` draws each gap from an exponential
    # with the same mean, which is what unmanaged arrival looks like — the
    # comparison between them is the whole point: it separates "spacing helps"
    # from "any decorrelation helps".
    rng = random.Random(a.seed_base)
    procs = []
    for i in range(a.n):
        if i and a.start_offset_s > 0:
            gap = (rng.expovariate(1.0 / a.start_offset_s)
                   if a.arrival == "poisson" else a.start_offset_s)
            time.sleep(gap)
        procs.append(
            subprocess.Popen(argv[f"replay-{i}"], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        )
    # Sample until every replay has exited. The loop owns the cadence, so
    # a slow scrape shortens the next sleep rather than drifting the
    # series: a drifting series would weight late samples differently
    # from early ones and the mean would stop being a time average.
    period = a.sample_period_ms / 1000.0
    next_at = time.time()
    while any(p.poll() is None for p in procs):
        v = scrape_running(a.metrics_url)
        samples.append({"t_unix_ns": time.time_ns(), "running": v})
        next_at += period
        time.sleep(max(0.0, next_at - time.time()))

    rcs = []
    for i, p in enumerate(procs):
        out, _ = p.communicate()
        (out_dir / f"replay-{i}.log").write_text(out or "")
        rcs.append(p.returncode)

    (out_dir / "running-series.json").write_text(json.dumps({
        "t_start_unix_ns": t_start,
        "t_end_unix_ns": time.time_ns(),
        "sample_period_ms": a.sample_period_ms,
        "metrics_url": a.metrics_url,
        "n_concurrent": a.n,
        "samples": samples,
    }, indent=1))

    missing = sum(1 for s in samples if s["running"] is None)
    print(f"[arm n={a.n}] {len(samples)} samples, {missing} failed scrapes, "
          f"replay rcs={rcs}")
    metas = [Path(str(s) + ".meta.json").exists() for s in steps]
    if any(rc != 0 for rc in rcs) or not all(metas):
        print(f"[arm n={a.n}] ABORT: not every replay passed its gates "
              f"(meta present: {metas}) — with N-1 trajectories this is a "
              f"different experiment, not a noisier one")
        return 1
    return 0


def analyse(out_dir, start_offset_s=0.0, arrival="fixed", heterogeneous=False):
    """Apply the declared window and report both readings.

    Returns a dict, also written to analysis.json. The untrimmed mean is
    reported beside the trimmed one because the difference between them
    is the size of the tail artefact, and a reader is entitled to see it
    rather than take the trim on trust.
    """
    d = json.loads((out_dir / "running-series.json").read_text())
    n = d["n_concurrent"]
    spans = []
    for p in sorted(out_dir.glob("steps-*.jsonl")):
        steps = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if not steps:
            return {"error": f"{p.name} is empty"}
        spans.append((steps[0]["t_start_unix_ns"], steps[-1]["t_end_unix_ns"]))
    if len(spans) != n:
        return {"error": f"{len(spans)} steps-files for n={n}"}

    lo, hi = max(s for s, _ in spans), min(e for _, e in spans)
    if hi <= lo:
        return {"error": "no interval with all N in flight — the "
                         "trajectories did not overlap, so nothing here "
                         "measures concurrency"}

    vals = [s["running"] for s in d["samples"] if s["running"] is not None]
    inwin = [s["running"] for s in d["samples"]
             if s["running"] is not None and lo <= s["t_unix_ns"] <= hi]
    if not inwin:
        return {"error": "no successful scrape inside the declared window"}

    # f_nongen from the trajectories themselves, same definition the cost
    # campaign publishes: tool wall plus inter-step gaps over the span.
    fracs = []
    for p in sorted(out_dir.glob("steps-*.jsonl")):
        steps = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        span = steps[-1]["t_end_unix_ns"] - steps[0]["t_start_unix_ns"]
        tool = sum(s["t_end_unix_ns"] - s["t_start_unix_ns"]
                   for s in steps if s["kind"] == "tool")
        gaps = sum(b["t_start_unix_ns"] - a_["t_end_unix_ns"]
                   for a_, b in zip(steps, steps[1:]))
        fracs.append((tool + gaps) / span)
    f = sum(fracs) / len(fracs)

    mean_in = sum(inwin) / len(inwin)
    predicted = n * (1.0 - f)
    res = {
        "n_concurrent": n,
        "f_nongen_mean": f,
        "predicted_running": predicted,
        "observed_running_windowed": mean_in,
        "observed_running_untrimmed": sum(vals) / len(vals) if vals else None,
        "window_ns": [lo, hi],
        "window_secs": (hi - lo) / NS,
        "samples_in_window": len(inwin),
        # Staggered-start experiment. The offset is echoed so a result
        # file states which arm produced it rather than relying on the
        # directory name. running_eq_one_fraction is the primary metric:
        # it is the direct measure of interleaving, and it was exactly
        # zero across every sample of the ADR-0010 lockstep run.
        "start_offset_s": start_offset_s,
        "arrival": arrival,
        "heterogeneous": heterogeneous,
        "running_eq_one_fraction": (
            sum(1 for v in inwin if v == 1) / len(inwin) if inwin else None
        ),
        "samples_total": len(d["samples"]),
        "failed_scrapes": sum(1 for s in d["samples"] if s["running"] is None),
    }
    # ADR-0010 D3, thresholds fixed before any run.
    if abs(mean_in - n) <= 0.10 * n:
        res["verdict"] = "BOUND RULED OUT AS CAPACITY: observed within 10% of N"
    elif abs(mean_in - predicted) <= 0.15 * predicted:
        res["verdict"] = "BOUND SUPPORTED: observed within 15% of N*(1-f_nongen)"
    else:
        res["verdict"] = "INCONCLUSIVE: between the two criteria, closes nothing"
    (out_dir / "analysis.json").write_text(json.dumps(res, indent=1))
    return res


def main():
    p = argparse.ArgumentParser(
        description="ADR-0010: drive N concurrent trajectories, sample the "
                    "engine's running count, apply the declared window.")
    p.add_argument("--out-dir", required=True, help="arm directory: the archivable unit")
    p.add_argument("--n", type=int, required=True,
                   help="concurrent trajectories. ADR-0010 D4 runs two arms: "
                        "n=1 as the anchor, n=ceil(bound) as the test.")
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics")
    p.add_argument("--driver", default=str(Path(__file__).resolve().parent))
    p.add_argument("--python",
                   default=str(Path(__file__).resolve().parent / ".venv/bin/python"))
    p.add_argument("--tool-latency-s", type=float, required=True,
                   help="must match the cost-campaign cell whose f_nongen "
                        "the prediction is taken from")
    p.add_argument("--tool-latency-cv", type=float, default=0.0)
    p.add_argument("--n-llm", type=int, default=4)
    p.add_argument("--n-tool", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--request-timeout-s", type=float, default=600.0,
                   help="passed through to each replay.")
    p.add_argument("--max-retries", type=int, default=0,
                   help="passed through to each replay. 0 keeps the "
                        "previous behaviour: a failed call ends the "
                        "trajectory. The preemption campaign raises it, "
                        "because a trajectory that dies measures nothing "
                        "about what recovery costs.")
    p.add_argument("--retry-backoff-s", type=float, default=2.0,
                   help="passed through to each replay.")
    p.add_argument("--heterogeneous", action="store_true",
                   help="vary steps and generation size per trajectory "
                        "around the same mean, so the N are not clones. "
                        "Clones cannot drift apart on their own, which "
                        "makes lockstep artificially persistent.")
    p.add_argument("--arrival", choices=["fixed", "poisson"], default="fixed",
                   help="fixed applies --start-offset-s between starts. "
                        "poisson draws each gap from an exponential with "
                        "that mean, which is how requests actually arrive: "
                        "neither synchronised nor evenly spaced.")
    p.add_argument("--start-offset-s", type=float, default=0.0,
                   help="delay between successive replica starts. 0 "
                        "reproduces the lockstep condition ADR-0010 "
                        "measured; a positive value staggers the "
                        "trajectories so one may generate while another "
                        "waits on a tool. The value is an input to be "
                        "declared, not tuned between runs.")
    p.add_argument("--sample-period-ms", type=int, default=250,
                   help="scrape cadence. Deliberately slower than the "
                        "operator reporter's: this is an HTTP round-trip "
                        "per sample and the quantity is a gauge, not a "
                        "counter that would lose events between reads.")
    p.add_argument("--dry-run", action="store_true",
                   help="print the argv of every arm and exit, spending nothing")
    p.add_argument("--analyse-only", action="store_true",
                   help="re-run the analysis over an existing arm directory")
    a = p.parse_args()

    if a.n < 1:
        sys.exit("--n must be >= 1")
    out_dir = Path(a.out_dir)

    if a.analyse_only:
        res = analyse(out_dir, a.start_offset_s, a.arrival, a.heterogeneous)
        print(json.dumps(res, indent=1))
        return 0 if "error" not in res else 1

    if a.dry_run:
        for i in range(a.n):
            print(f"--- replay-{i}\n{' '.join(replay_argv(a, i, out_dir / f'steps-{i}.jsonl'))}\n")
        print(f"--- sampler\nGET {a.metrics_url} every {a.sample_period_ms}ms "
              f"until all {a.n} replays exit")
        return 0

    rc = run_arm(a, out_dir)
    res = analyse(out_dir, a.start_offset_s, a.arrival, a.heterogeneous)
    print(json.dumps(res, indent=1))
    if "error" in res:
        return 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
