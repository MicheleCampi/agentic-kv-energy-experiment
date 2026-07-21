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
- inferscope: build on node from public repo (cargo build --release --features gpu-nvidia).

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
   inferscope --sample-only --gpu --pid <vllm-pid> --duration-secs <span+30> \
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

## Session findings (2026-07-21, executed — PASS)
- Two runbook gaps found live (~9 min of the 15-min window), zero code
  bugs. Root cause common to both: the e2e chain was rehearsed only on
  the VM, where the GPU path is withheld by design, so the GPU flags
  were never exercised.
  1. Build: `gpu-nvidia` is a non-default cargo feature; plain
     `cargo build --release` compiles no GPU path (no --gpu in --help).
  2. Invocation: `--gpu` is explicit opt-in (ADR-005); token/cache
     deltas additionally require `--metrics-endpoint` + `--model`
     (ADR-011) or all deltas read zero and tok/J stays null.
  Corrected step-4 invocation:
    inferscope --sample-only --gpu --pid <vllm-pid> \
      --duration-secs <span+30> \
      --metrics-endpoint http://127.0.0.1:8000/metrics \
      --model <model-id> --steps-file runs/gpu-<ts>.jsonl
- Session assert PASS on both trajectories: per-step energy, exact
  reconciliation (steps + unattributed == total, zero rounding drift),
  dropped_steps empty. Second trajectory with ADR-011 scrape active:
  168 generation tokens, trajectory tok/J populated.
- Secondary finding: this vLLM digest honors parallel_tool_calls=false
  (1 tool_call with 2 tools offered) — unlike the llama.cpp rehearsal.
  Middleware serialization stays regardless.
- Evidence: driver/runs/gpu-session-a/ (includes the three failed
  reports and the orphan gpu-.jsonl documenting the gaps).
- Node: 1x A10 Lambda, ~30 min wall, well under the $3 cap.
