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
import sys
import time
import uuid

from openai import OpenAI

from steps_callback import StepsFileCallback
from trajectory_gates import check_and_write_meta


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--steps-file", required=True)
    p.add_argument("--tool-latency-s", type=float, default=0.2,
                   help="seconds each tool step sleeps. The swept variable.")
    p.add_argument("--n-llm", type=int, default=4,
                   help="LLM calls in the trajectory. Fixed by design.")
    p.add_argument("--n-tool", type=int, default=3,
                   help="tool steps, interleaved between LLM calls. Fixed.")
    p.add_argument("--max-tokens", type=int, default=192,
                   help="tokens generated per LLM call. Fixed so that span "
                        "varies with engine throughput, not with how much "
                        "the model felt like saying.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.n_llm < 2 or args.n_tool < 2:
        sys.exit("--n-llm and --n-tool must both be >= 2 (ADR-013 gates)")

    client = OpenAI(base_url=args.base_url, api_key="dummy")
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
        resp = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=0.0,
            seed=args.seed,
            max_tokens=args.max_tokens,
        )
        handler.end_step(run_id)
        text = resp.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": text})

        # Interleave a tool step after every LLM call but the last, so
        # the trajectory ends on generation as a real loop does.
        if i < args.n_tool:
            tool_id = uuid.uuid4()
            handler.begin_step(tool_id, "tool")
            time.sleep(args.tool_latency_s)
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
