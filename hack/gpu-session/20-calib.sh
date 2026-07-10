#!/usr/bin/env bash
# Session script — ALL parameters written (rehearsal 2026-07-10: two of
# three wrong launches were parameters recomposed from memory). Do not
# edit at node-on; deviations = abort.
set -euo pipefail
source "$(dirname "$0")/00-env.sh"
export PATH="$HOME/venv/bin:$PATH"
export PYTHONUNBUFFERED=1  # F14b: block-buffered stdout lags tee/tail monitoring
cd "$(dirname "$0")/../.."
mkdir -p "$HOME/exp-results"  # F13: tee target must exist before the pipe starts

OUT="$HOME/exp-results/calib-$(date +%Y%m%d)"
python3 run_experiment.py \
  --model "$EXP_MODEL" \
  --regimes H1 --conditions nominal --reps "$EXP_CALIB_REPS" \
  --n-sessions "$EXP_CALIB_N_SESSIONS" \
  --target-context 32768 \
  --seed-base 42 --run-nonce 100 \
  --sample-secs 120 \
  --inferscope-bin "$EXP_INFERSCOPE_BIN" \
  --ready-timeout 900 \
  --out-dir "$OUT" \
  2>&1 | tee "$OUT.log"
# Executable verdict, exit code propagated: 0 GREEN / 2 YELLOW / 1 RED
exec python3 hack/gpu-session/gonogo.py "$OUT"
