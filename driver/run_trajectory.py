"""ADR-013 rehearsal driver: one agentic trajectory against an OpenAI-compatible
endpoint, emitting an inferscope steps-file via StepsFileCallback.

One trajectory in flight (ADR-013 wall). No subagents: keeps run boundaries
flat, so start/end pairs map 1:1 onto whole segments for reconciliation.
"""

import argparse
import json
import sys
import time

from deepagents import create_deep_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from steps_callback import StepsFileCallback
from serialize_tools import SerializeToolCalls


@tool
def lookup_component(name: str) -> str:
    """Return specs for a named infrastructure component."""
    time.sleep(0.2)  # deterministic, visible tool segment
    specs = {
        "gpu-node": "8x H100 SXM5, 2TB RAM, NVLink",
        "scheduler": "EPP-based, KV-cache aware scoring",
        "cache": "prefix cache, LRU eviction, per-pod",
    }
    return specs.get(name, f"no spec found for '{name}'")


@tool
def estimate_cost(gpu_hours: float, rate_per_hour: float) -> str:
    """Estimate cost in USD for a number of GPU hours at a given hourly rate."""
    time.sleep(0.2)
    return f"${gpu_hours * rate_per_hour:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--steps-file", required=True)
    parser.add_argument("--max-steps", type=int, default=20,
                        help="recursion limit for the graph (runaway guard)")
    args = parser.parse_args()

    model = ChatOpenAI(
        base_url=args.base_url,
        api_key="dummy",
        model=args.model,
        temperature=0.0,
    )

    handler = StepsFileCallback(args.steps_file)

    agent = create_deep_agent(
        model=model,
        tools=[lookup_component, estimate_cost],
        middleware=[SerializeToolCalls()],
        system_prompt=(
            "You are an infrastructure planning assistant. "
            "Use the available tools to answer. Call one tool at a time."
        ),
    )

    prompt = (
        "Plan a small inference deployment: look up the specs for 'gpu-node' "
        "and 'scheduler', then estimate the cost of 24 GPU hours at $2.50/hour. "
        "Summarize your findings."
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"callbacks": [handler], "recursion_limit": args.max_steps},
    )

    # sanity gates
    if handler.open_runs != 0:
        print(f"FAIL: {handler.open_runs} unclosed run(s) — boundary anomaly",
              file=sys.stderr)
        return 1

    with open(args.steps_file, encoding="utf-8") as f:
        steps = [json.loads(line) for line in f]

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
    print(f"final message: {result['messages'][-1].content[:200]!r}")

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
