#!/usr/bin/env python3
"""
Deterministic shared-prefix builder for the agentic KV-energy experiment.

Produces a versioned prefix artifact (system prompt + tool definitions in
Claude-Code-style JSON-schema) and reports its EXACT token count under the
Qwen2.5 tokenizer (shared across all Qwen2.5 sizes, so the count matches the
32B target model on the node).

The prefix is the shared context that induces KV-cache hits. Deterministic:
fixed seed -> identical prefix -> identical token count.
inferscope is NOT involved here; this only builds the input artifact.
"""
import argparse
import json
import random
from pathlib import Path

TOKENIZER_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # tokenizer shared with Qwen2.5-32B
PREFIX_DIR = Path(__file__).parent / "prefixes"

SYSTEM_PREAMBLE = """You are an autonomous software engineering agent operating in a ReAct loop.
At each step you reason about the current state, decide on an action, invoke one
or more tools, observe the results, and continue until the task is complete.
You have access to a filesystem, a shell, and a set of tools defined below.
Always think step by step before acting. Prefer reading and exploring before
writing or executing. Report progress concisely and stop when the goal is met.
You must follow the project's conventions and never fabricate file contents you
have not read. When a tool returns an error, diagnose before retrying.
"""

TOOL_NAMES = [
    "read_file", "write_file", "edit_file", "bash", "grep_search",
    "glob_search", "list_directory", "run_tests", "git_diff", "git_commit",
    "web_fetch", "web_search", "launch_subagent", "task_output", "apply_patch",
    "delete_file", "move_file", "create_directory", "read_logs", "lint_check",
]
FIELD_TYPES = ["string", "integer", "boolean", "array", "object"]


def make_tool_def(name, rng):
    n_params = rng.randint(2, 6)
    props, required = {}, []
    for i in range(n_params):
        pname = f"{name}_arg{i}"
        ptype = rng.choice(FIELD_TYPES)
        prop = {"type": ptype, "description": (
            f"Parameter {i} of the {name} tool. Provides the {ptype} value used "
            f"to control how {name} operates on the target. Must be supplied "
            f"when the agent invokes this tool in a step.")}
        if ptype == "array":
            prop["items"] = {"type": "string"}
        if ptype == "object":
            prop["properties"] = {"key": {"type": "string"}}
        props[pname] = prop
        if rng.random() < 0.6:
            required.append(pname)
    return {"type": "function", "function": {
        "name": name,
        "description": (
            f"The {name} tool performs the {name.replace('_', ' ')} operation "
            f"within the agent's execution environment. Use it during the act "
            f"phase of the ReAct loop. It returns an observation appended to "
            f"the conversation context."),
        "parameters": {"type": "object", "properties": props, "required": required}}}


def build_prefix(n_tools, seed):
    rng = random.Random(seed)
    names, i = [], 0
    while len(names) < n_tools:
        base = TOOL_NAMES[i % len(TOOL_NAMES)]
        suffix = i // len(TOOL_NAMES)
        names.append(base if suffix == 0 else f"{base}_v{suffix}")
        i += 1
    tools = [make_tool_def(nm, rng) for nm in names]
    parts = [SYSTEM_PREAMBLE, "\n\n## Available tools\n\n"]
    for t in tools:
        parts.append(json.dumps(t, indent=2))
        parts.append("\n\n")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tools", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--target-tokens", type=int, default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    n_tools = args.n_tools
    if args.target_tokens is not None:
        n_tools = 1
        while True:
            ntok = len(tok.encode(build_prefix(n_tools, args.seed)))
            if ntok >= args.target_tokens or n_tools > 2000:
                break
            n_tools += 1
        print(f"auto-tuned n_tools={n_tools} for target ~{args.target_tokens}")

    text = build_prefix(n_tools, args.seed)
    ntok = len(tok.encode(text))
    PREFIX_DIR.mkdir(exist_ok=True)
    out = PREFIX_DIR / f"agentic_system_{args.version}.txt"
    out.write_text(text)
    meta = PREFIX_DIR / f"agentic_system_{args.version}.meta.json"
    meta.write_text(json.dumps({
        "version": args.version, "tokenizer": TOKENIZER_ID, "seed": args.seed,
        "n_tools": n_tools, "char_len": len(text), "token_count": ntok}, indent=2))
    print(f"wrote {out.name}: {ntok} tokens ({len(text)} chars, n_tools={n_tools})")


if __name__ == "__main__":
    main()
