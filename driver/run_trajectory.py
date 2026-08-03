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
from trajectory_gates import check_and_write_meta
from serialize_tools import SerializeToolCalls


# Tool latency is a DESIGN PARAMETER, not a measurement. It is swept
# across cells so that the published figure is a curve with a stated
# crossover, never a single point that a reader could mistake for a
# property of the workload. Set via --tool-latency-s; the value in
# force is recorded next to the steps-file.
TOOL_LATENCY_S = 0.2

@tool
def lookup_component(name: str) -> str:
    """Return specs for a named infrastructure component."""
    time.sleep(TOOL_LATENCY_S)  # deterministic, visible tool segment
    specs = {
        "gpu-node": "8x H100 SXM5, 2TB RAM, NVLink",
        "scheduler": "EPP-based, KV-cache aware scoring",
        "cache": "prefix cache, LRU eviction, per-pod",
    }
    return specs.get(name, f"no spec found for '{name}'")


@tool
def estimate_cost(gpu_hours: float, rate_per_hour: float) -> str:
    """Estimate cost in USD for a number of GPU hours at a given hourly rate."""
    time.sleep(TOOL_LATENCY_S)
    return f"${gpu_hours * rate_per_hour:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--steps-file", required=True)
    parser.add_argument("--tool-latency-s", type=float, default=0.2,
                        help="seconds each tool sleeps. Design parameter, "
                             "swept across cells; recorded in the meta file.")
    parser.add_argument("--max-steps", type=int, default=20,
                        help="recursion limit for the graph (runaway guard)")
    parser.add_argument(
        "--prompt",
        default=(
            "Plan a small inference deployment: look up the specs for 'gpu-node' "
            "and 'scheduler', then estimate the cost of 24 GPU hours at $2.50/hour. "
            "Summarize your findings."
        ),
        help=(
            "task given to the agent. The default is the one that produced the "
            "published evidence; override it to drive a longer trajectory."
        ),
    )
    args = parser.parse_args()
    global TOOL_LATENCY_S
    TOOL_LATENCY_S = args.tool_latency_s

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

    prompt = args.prompt

    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"callbacks": [handler], "recursion_limit": args.max_steps},
    )

    return check_and_write_meta(
        args.steps_file,
        handler.open_runs,
        {
            "arm": "agentic",
            "tool_latency_s": args.tool_latency_s,
            "model": args.model,
            "base_url": args.base_url,
            "max_steps": args.max_steps,
            "prompt": args.prompt,
        },
        final_text=result["messages"][-1].content,
    )


if __name__ == "__main__":
    sys.exit(main())
