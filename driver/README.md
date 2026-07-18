# ADR-013 trajectory driver

Runs one agentic trajectory (Deep Agents / langgraph) against an
OpenAI-compatible endpoint and emits an inferscope ADR-013 steps-file
(JSONL, wall-clock UTC ns, one record per whole llm_call/tool segment).

- `steps_callback.py` — callback handler; pairs start/end on run_id,
  timestamps taken in-handler, error hooks drop open segments,
  `open_runs != 0` after a run = boundary anomaly, run discarded.
- `serialize_tools.py` — middleware serializing tool execution
  (llama.cpp ignores `parallel_tool_calls=false`; truncation would
  alter the model-decided trajectory, so execution is serialized).
- `run_trajectory.py` — driver with hard gates: no open runs,
  positive spans, >=2 llm_call + >=2 tool steps, zero overlapping
  segments.

Rehearsal target: llama.cpp `llama-server` b10068, Qwen2.5-0.5B-Instruct
GGUF q8_0, `--parallel 1 -c 8192`. GPU run: same driver, `--base-url`
pointed at vLLM.
