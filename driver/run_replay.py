"""Deterministic ReAct-shape replay emitting an inferscope ADR-013 steps-file.

Why this exists alongside run_trajectory.py
-------------------------------------------
run_trajectory.py drives a real Deep Agents loop. Measured on the same
prompt, model and temperature=0.0, its span moved 7.4s -> 34.7s across
runs and its step count 4 -> 8: the model decides the trajectory's shape,
so both the numerator (tool wall) and the denominator (span) of the
quantity under study move for reasons unrelated to the parameter being
swept. A curve built on that arm would not be readable.

This is the same conclusion PROTOCOL.md reached for the hit-rate matrix
("un agente reale e' non-deterministico ... quel non-determinismo
distruggerebbe la riproducibilita' della misura energetica"), reached
again for a different measurement. The replay fixes the structure —
number of LLM calls, number of tool steps, tokens generated per call —
and leaves tool latency as the only swept variable.

What it is NOT: a claim about agent behaviour. It replays the SHAPE of an
agentic trajectory. The agentic arm anchors it: a few cells of
run_trajectory.py at the same latency, on the same node, showing real
trajectories landing in the region the replay describes.

Both arms write the same steps-file format through the same
StepsFileCallback and pass the same gates in trajectory_gates.py.
"""
import argparse
import math
import random
import sys
import time
import uuid

from openai import OpenAI

from steps_callback import StepsFileCallback
from trajectory_gates import check_and_write_meta


def sample_latency(rng: random.Random, mean_s: float, cv: float) -> float:
    """One tool latency in seconds, lognormal with the stated mean and CV.

    Parametrised by ARITHMETIC mean, not by the underlying normal's mean.
    For a lognormal, sigma_log**2 = ln(1 + cv**2) and mu_log = ln(mean) -
    sigma_log**2 / 2; without that second term the realised mean would be
    mean * exp(sigma_log**2 / 2), so at cv=0.5 every cell would run 11%
    slower than the axis says it did. The correction is what makes
    --tool-latency-s keep meaning what it claims once cv > 0.

    Lognormal rather than truncated normal because real tool calls are
    right-skewed: a low median with a long tail of slow ones. A normal
    would also need truncation at zero, which silently shifts the mean
    it was chosen to preserve.

    cv == 0 returns the mean exactly, so existing cells reproduce
    bit-for-bit rather than approximately.
    """
    if cv == 0.0:
        return mean_s
    sigma_log = math.sqrt(math.log(1.0 + cv * cv))
    mu_log = math.log(mean_s) - sigma_log * sigma_log / 2.0
    return rng.lognormvariate(mu_log, sigma_log)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--steps-file", required=True)
    p.add_argument("--tool-latency-s", type=float, default=0.2,
                   help="mean seconds each tool step sleeps. The swept "
                        "variable.")
    p.add_argument("--tool-latency-cv", type=float, default=0.0,
                   help="coefficient of variation (sigma/mean) of tool "
                        "latency. 0.0 reproduces the fixed-latency "
                        "behaviour exactly. A CV is comparable across "
                        "cells of different mean, which an absolute sigma "
                        "is not: 0.5s of spread is absurd at a 0.2s mean "
                        "and modest at a 5.0s one. Sampled lognormal, "
                        "which is right-skewed like real tool calls: "
                        "median low, mean pulled up by the rare slow one.")
    p.add_argument("--n-llm", type=int, default=4,
                   help="LLM calls in the trajectory. Fixed by design.")
    p.add_argument("--n-tool", type=int, default=3,
                   help="tool steps, interleaved between LLM calls. Fixed.")
    p.add_argument("--max-tokens", type=int, default=192,
                   help="tokens generated per LLM call. Fixed so that span "
                        "varies with engine throughput, not with how much "
                        "the model felt like saying.")
    p.add_argument("--request-timeout-s", type=float, default=600.0,
                   help="per-request timeout. The default matches the "
                        "OpenAI client default, so behaviour is unchanged "
                        "unless set. The preemption campaign lowers it, "
                        "because a request hanging on a dead engine never "
                        "reaches the retry path.")
    p.add_argument("--max-retries", type=int, default=0,
                   help="retries per LLM call when the endpoint fails. "
                        "0 reproduces the previous behaviour exactly: an "
                        "exception ends the trajectory. The preemption "
                        "campaign raises it, because a trajectory that "
                        "dies measures nothing about recovery cost.")
    p.add_argument("--retry-backoff-s", type=float, default=2.0,
                   help="wait between retries. Fixed rather than "
                        "exponential so the recovery cost measured is the "
                        "engine's, not the backoff policy's.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.n_llm < 2 or args.n_tool < 2:
        sys.exit("--n-llm and --n-tool must both be >= 2 (ADR-013 gates)")
    if args.tool_latency_cv < 0.0:
        sys.exit("--tool-latency-cv must be >= 0")
    if args.tool_latency_s <= 0.0:
        sys.exit("--tool-latency-s must be > 0")

    # Own RNG instance, not the module-global state: any library that
    # samples in this process would otherwise shift the sequence and
    # make the seed recorded in the .meta.json a false claim.
    rng = random.Random(args.seed)
    latencies = []

    # An explicit timeout, because the default is ten minutes and a request
    # in flight when the engine dies does not fail — it hangs. Measured
    # during the preemption session: a turn interrupted mid-generation sat
    # at 607s with zero retries, because the client was still waiting for a
    # reply from a process that no longer existed. That is worth knowing in
    # its own right: an agent with default settings does not notice a lost
    # replica, it waits. The timeout here turns that hang into the failure
    # the retry path is built to absorb.
    client = OpenAI(base_url=args.base_url, api_key="dummy",
                    timeout=args.request_timeout_s)
    handler = StepsFileCallback(args.steps_file)

    # A growing conversation, as in a real ReAct loop: each turn carries
    # the history, so prefix reuse behaves the way the cache regimes
    # expect. Content is fixed text, not model-authored, so the token
    # budget per turn is a design parameter and not an outcome.
    messages = [
        {"role": "system",
         "content": "You are an infrastructure planning assistant. "
                    "Answer with concrete technical detail."},
        {"role": "user",
         "content": "Plan a small inference deployment. Describe the GPU node, "
                    "the scheduler, and the cache layer in turn."},
    ]

    for i in range(args.n_llm):
        run_id = uuid.uuid4()
        handler.begin_step(run_id, "llm_call")
        # Retry on failure, resending the whole conversation. That is what a
        # real agent does: the state lives in the client, so when the endpoint
        # it was talking to disappears the accumulated context has to go back
        # over the wire and be reprocessed. The cost of losing a replica
        # mid-trajectory is exactly that context, and measuring it is the point
        # of the preemption campaign — a retry that resent only the last turn
        # would measure something cheaper than what actually happens.
        resp = None
        attempts = 0
        first_failure_ns = None
        while resp is None:
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=0.0,
                    seed=args.seed,
                    max_tokens=args.max_tokens,
                )
            except Exception as exc:
                attempts += 1
                if first_failure_ns is None:
                    first_failure_ns = time.time_ns()
                if attempts > args.max_retries:
                    print(f"step {i}: giving up after {attempts} attempts: {exc}",
                          file=sys.stderr)
                    raise
                time.sleep(args.retry_backoff_s)
        handler.end_step(
            run_id,
            prompt_tokens=getattr(resp.usage, "prompt_tokens", None),
            completion_tokens=getattr(resp.usage, "completion_tokens", None),
            retries=attempts,
            first_failure_ns=first_failure_ns,
        )
        text = resp.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": text})

        # Interleave a tool step after every LLM call but the last, so
        # the trajectory ends on generation as a real loop does.
        if i < args.n_tool:
            tool_id = uuid.uuid4()
            handler.begin_step(tool_id, "tool")
            latency = sample_latency(rng, args.tool_latency_s,
                                     args.tool_latency_cv)
            latencies.append(latency)
            time.sleep(latency)
            handler.end_step(tool_id)
            messages.append({
                "role": "user",
                "content": f"Tool result {i + 1}: component specification "
                           f"retrieved. Continue the plan.",
            })

    return check_and_write_meta(
        args.steps_file,
        handler.open_runs,
        {
            "arm": "replay",
            "tool_latency_s": args.tool_latency_s,
            "tool_latency_cv": args.tool_latency_cv,
            "tool_latency_realised_s": latencies,
            "model": args.model,
            "base_url": args.base_url,
            "n_llm_planned": args.n_llm,
            "n_tool_planned": args.n_tool,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
    )


if __name__ == "__main__":
    sys.exit(main())
