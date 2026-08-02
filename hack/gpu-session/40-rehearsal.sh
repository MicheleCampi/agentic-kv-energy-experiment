#!/usr/bin/env bash
# Dress rehearsal, node-off, zero cost. ALL parameters written: the
# rehearsal proves the SHAPE of the launch, so a launch composed from
# memory proves nothing (lesson 2026-07-10).
#
# Mandatory before every GPU session. Runs run_experiment.py UNMODIFIED
# against fake_engine.py via the versioned shim in rehearsal-bin/.
#
# What this CAN certify: argument contract, engine launch and readiness,
# EngineCore PID discovery, KV series parsed from a real /metrics body
# with the vLLM vocabulary, report shape, cost subcommand reachability.
# What it CANNOT: energy (no NVML on optim-dev -> "gpu": null is EXPECTED
# here and an ABORT criterion on the node), and the H2>H0 separation
# (the fake's hit-rate is synthetic, 50% by construction).
set -euo pipefail
source "$(dirname "$0")/00-env-cost-a10.sh"
export PATH="$(cd "$(dirname "$0")/rehearsal-bin" && pwd):$PATH"
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/../.."
# Deliberately NOT $EXP_OUT_DIR: rehearsal output must never land in
# the session directory, or a resumed campaign could skip real cells.
OUT="$HOME/exp-results/rehearsal-$(date +%Y%m%dT%H%M%S)"
mkdir -p "$OUT"

# --sample-secs 60: NOT a dimensioned window. On the node it is measured
# from the warm-up cell x1.2; here the fake answers instantly and there is
# nothing to measure. Deliberately unlike any session value so it cannot
# be mistaken for one. The oversized-window warning SHOULD fire.
exec python3 run_experiment.py \
  --model "$EXP_MODEL" \
  --endpoint http://127.0.0.1:8000 \
  --metrics-url http://127.0.0.1:8000/metrics \
  --regimes H0,H2 --conditions nominal --reps 1 \
  --n-sessions 2 \
  --target-context 2048 \
  --seed-base 42 \
  --sample-secs 60 \
  --inferscope-bin "$EXP_INFERSCOPE_BIN" \
  --ready-timeout 60 \
  --out-dir "$OUT" \
  2>&1 | tee "$OUT.log"
