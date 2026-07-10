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

# Growth-stop smoke: nonce 9 -> H2 rep1 samples n_turns=63 (scan on
# optim-dev 2026-07-10, deterministic): at least one session MUST build
# to effective target (checklist §2).
exec python3 run_experiment.py \
  --model "$EXP_MODEL" \
  --regimes H2 --conditions nominal --reps 1 \
  --n-sessions 2 \
  --target-context 32768 \
  --seed-base 42 --run-nonce 9 \
  --sample-secs 120 \
  --inferscope-bin "$EXP_INFERSCOPE_BIN" \
  --ready-timeout 900 \
  --out-dir "$HOME/exp-results/smoke-$(date +%Y%m%d)" \
  2>&1 | tee "$HOME/exp-results/smoke-$(date +%Y%m%d).log"
