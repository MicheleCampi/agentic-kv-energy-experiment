#!/usr/bin/env bash
# Session script, phase 2 — cost campaign (ADR-013 + ADR-015).
# ALL parameters written: a launch composed from memory proves nothing
# (lesson 2026-07-10). Do not edit at node-on; deviations = abort.
#
# Two invocations, never one run:
#   50-cost.sh calibrate            one trajectory, ~1 min of node
#   50-cost.sh sweep <span-secs>    the campaign, window sized from it
#
# They are separate because a human decision sits between them. The
# non-generating fraction depends on the tool/LLM ratio, so if the LLM
# span on A10/Qwen2.5-7B differs from the llama.cpp rehearsal, the pinned
# latencies land on different points of the curve and must be recalibrated
# BEFORE any cell is spent. A script that carried on by itself would
# remove that stop.
set -euo pipefail
source "$(dirname "$0")/00-env-cost-a10.sh"
export PATH="$HOME/venv/bin:$PATH"
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/../.."

CELL="driver/run_cost_cell.py"

# The window is sized per cell from the calibration span plus the tool
# wall this cell adds, times the pinned margin. It cannot be one figure
# for the whole sweep: at 5.0s/tool the trajectory is ~15s longer than at
# 0.2s, and a window that fits the short cells truncates the long ones.
# Whole seconds because inferscope rejects a decimal point.
run_cell() {
  local lat="$1" cv="$2" seed="$3" span="$4"
  local win
  win=$(python3 -c "import math,sys; print(int(math.ceil((float(sys.argv[1]) + 3*float(sys.argv[2])) * float(sys.argv[3]))))" \
        "$span" "$lat" "$EXP_WINDOW_MARGIN")
  local name="lat${lat}-cv${cv}-seed${seed}"
  echo "[cell] $name  window=${win}s"
  python3 "$CELL" \
    --out-dir "$EXP_OUT_DIR/$name" \
    --engine-pid "$EXP_ENGINE_PID" \
    --window-secs "$win" \
    --model "$EXP_MODEL" \
    --metrics-url http://127.0.0.1:8000/metrics \
    --engine vllm \
    --inferscope-bin "$EXP_INFERSCOPE_BIN" \
    --tool-latency-s "$lat" \
    --tool-latency-cv "$cv" \
    --seed "$seed" \
    --reentry-secs "$EXP_COST_REENTRY_SECS" \
    --usd-per-hour "$EXP_COST_USD_PER_HOUR"
}
MODE="${1:-}"

if [ -z "${EXP_ENGINE_PID:-}" ]; then
  echo "EXP_ENGINE_PID unset: export the EngineCore PID (NOT the APIServer" >&2
  echo "parent — attaching there samples an idle GPU)." >&2
  exit 1
fi

case "$MODE" in
  calibrate)
    OUT="$EXP_OUT_DIR/calib"
    echo "[calib] one trajectory at ${EXP_COST_CV_CELL}s/tool, window 120s"
    echo "[calib] ABORT if the report carries no trajectory section: the"
    echo "[calib] join is not certifiable off-node, this is where it is"
    echo "[calib] proven (gpu_timeline is None without NVML)."
    exec python3 "$CELL" \
      --out-dir "$OUT" \
      --engine-pid "$EXP_ENGINE_PID" \
      --window-secs 120 \
      --model "$EXP_MODEL" \
      --metrics-url http://127.0.0.1:8000/metrics \
      --engine vllm \
      --inferscope-bin "$EXP_INFERSCOPE_BIN" \
      --tool-latency-s "$EXP_COST_CV_CELL" \
      --seed 42 \
      --reentry-secs "$EXP_COST_REENTRY_SECS" \
      --usd-per-hour "$EXP_COST_USD_PER_HOUR"
    ;;
  sweep)
    SPAN="${2:-}"
    if [ -z "$SPAN" ]; then
      echo "usage: 50-cost.sh sweep <span-secs>   (observed_span_s from calibrate)" >&2
      exit 1
    fi
    RC=0
    for LAT in $EXP_COST_LATENCIES; do
      for SEED in $EXP_COST_SEEDS; do
        run_cell "$LAT" 0.0 "$SEED" "$SPAN" || RC=1
      done
    done
    for SEED in $EXP_COST_SEEDS; do
      run_cell "$EXP_COST_CV_CELL" "$EXP_COST_CV" "$SEED" "$SPAN" || RC=1
    done
    echo "[sweep] done, rc=$RC -> $EXP_OUT_DIR"
    exit "$RC"
    ;;
  *)
    echo "usage: 50-cost.sh calibrate | 50-cost.sh sweep <span-secs>" >&2
    exit 1
    ;;
esac
