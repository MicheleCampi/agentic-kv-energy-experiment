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

# --sample-secs is THE ONLY session-time decision: sized from calibration
# wall times (window > slowest cell, checklist §3). No default on purpose.
if [ $# -ne 1 ]; then
  echo "usage: 30-matrix.sh <sample-secs>  (size it from 20-calib wall times)" >&2
  exit 1
fi
exec python3 run_experiment.py \
  --model "$EXP_MODEL" \
  --regimes H0,H1,H2 --conditions nominal,failure --reps 3 \
  --n-sessions 8 \
  --target-context 32768 \
  --seed-base 42 --run-nonce 200 \
  --sample-secs "$1" \
  --inferscope-bin "$EXP_INFERSCOPE_BIN" \
  --ready-timeout 900 \
  --out-dir "$HOME/exp-results/matrix-$(date +%Y%m%d)" \
  2>&1 | tee "$HOME/exp-results/matrix-$(date +%Y%m%d).log"
