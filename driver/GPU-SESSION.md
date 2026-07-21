# ADR-013 GPU session runbook (session "a" — short, de-risked)

Everything below was rehearsed at zero cost; the session pays for
execution only. Budget hard cap: $3 (1× A10, ~1h at Lambda rates —
verify the current rate before launch; abort setup if node not usable
within 20 min).

## Pins (frozen)

- Node: 1× A10, Lambda. Model: Qwen/Qwen2.5-7B-Instruct.
- vLLM: same digest as item-4 sessions (docker, digest-pinned).
- Repo transfer: rsync from optim-dev with ~/.ssh/runpod_optimdev
  (private repo rule — no clone on Lambda). Same dir name.
- Driver venv: recreate from driver/requirements.txt (pinned).
- inferscope: build on node from public repo (cargo build --release).

## Sequence (each step has its assert; stop-and-decide on any FAIL)

1. vLLM up, OpenAI-compat on :8000, model loaded.
   Assert: curl /v1/models returns the model id.
2. Tool-calling sanity + parallel_tool_calls honored by vLLM.
   Assert: a direct /v1/chat/completions request with two tools and
   parallel_tool_calls=false returns at most one tool_call per turn.
   (Middleware serialization stays regardless — belt and braces.)
3. Driver run at real length against vLLM:
   .venv/bin/python run_trajectory.py \
     --base-url http://127.0.0.1:8000/v1 \
     --model Qwen/Qwen2.5-7B-Instruct --steps-file runs/gpu-<ts>.jsonl
   Assert: driver gates pass (>=2+2 steps, zero overlap, positive spans).
4. Concurrent attach, the rehearsed chain:
   inferscope --sample-only --pid <vllm-pid> --duration-secs <span+30> \
     --steps-file runs/gpu-<ts>.jsonl
   started ~2s before the driver (same launch pattern as the VM e2e).
   Assert (THE session assert): trajectory section PRESENT in the
   report — per-step attribution with GPU timeline, energy per step,
   reconciliation sums (steps + unattributed == total), zero dropped
   steps for out-of-window reasons.
5. Repeat 3+4 once (two trajectories total): variance sanity, not
   statistics.
6. Evidence off-box BEFORE teardown: rsync runs/ back to optim-dev.

## Abort criteria

- Step 2 fails (vLLM ignores parallel_tool_calls): NOT a session
  blocker — middleware already serializes; record the finding, go on.
- Step 4 yields absent trajectory section with NVML present: stop,
  capture report + steps-file + timestamps off-box, teardown, root
  cause at zero cost. No live debugging beyond 15 min.
- Budget cap reached: teardown regardless of state.

## Deferred to session "b" (not this session)

NVML in the operator reporter, EA-vs-WF 8 reps, RPS/tariff check on
A10 for the operator experiment.
