#!/usr/bin/env bash
# Session script, ADR-0010 — does the packing bound hold as capacity?
# ALL parameters written: a launch composed from memory proves nothing
# (lesson 2026-07-10). Do not edit at node-on; deviations = abort.
#
#   60-concurrent.sh anchor     one arm at N=1, ~2 min of node
#   60-concurrent.sh test       one arm at N=2, ~2 min of node
#
# Two invocations because the anchor is a gate, not a warm-up. At N=1 the
# prediction is 0.630 running against 0.90 for the exclusion threshold, so
# the anchor already discriminates: if it reads ~1.0 instead, the engine
# holds a request in `running` through the driver's tool sleeps, the
# observable does not mean what D2 assumes, and the test arm would measure
# nothing. That is 41 seconds to find out, and it must be read before the
# second arm runs.
#
# Cell fixed by the ADR-0010 postscript of 2026-08-07: 5.0 s/tool is the
# only latency of the cost campaign where D3's two bands separate
# (gap +0.351; at 2.0 s/tool it is -0.063 and the experiment is
# undecidable). N=2 is ceil(bound) at that cell.
set -euo pipefail
source "$(dirname "$0")/00-env-cost-a10.sh"
export PATH="$HOME/venv/bin:$PATH"
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/../.."

CELL="driver/run_concurrent.py"
LAT=5.0
MODE="${1:-}"

if [ -z "${EXP_ENGINE_PID:-}" ]; then
  echo "EXP_ENGINE_PID unset: export the EngineCore PID (NOT the APIServer" >&2
  echo "parent — attaching there samples an idle GPU)." >&2
  exit 1
fi

run_arm() {
  local n="$1" seed="$2" rep="$3"
  local out="$EXP_OUT_DIR/adr0010/n${n}-rep${rep}"
  echo "[arm] N=$n rep=$rep seed=$seed  -> $out"
  python3 "$CELL" \
    --out-dir "$out" \
    --n "$n" \
    --model "$EXP_MODEL" \
    --base-url http://127.0.0.1:8000/v1 \
    --metrics-url http://127.0.0.1:8000/metrics \
    --tool-latency-s "$LAT" \
    --tool-latency-cv 0.0 \
    --seed-base "$seed" \
    --sample-period-ms 250
}

case "$MODE" in
  anchor)
    echo "[anchor] N=1, 3 reps at ${LAT}s/tool. Expect mean running ~0.63."
    echo "[anchor] READ THE VERDICT BEFORE RUNNING THE TEST ARM: a mean near"
    echo "[anchor] 1.0 means the engine counts a request as running through"
    echo "[anchor] the tool sleeps, and D2's observable does not hold."
    RC=0
    i=0
    for SEED in $EXP_COST_SEEDS; do
      i=$((i+1))
      run_arm 1 "$SEED" "$i" || RC=1
    done
    echo "[anchor] done, rc=$RC"
    exit "$RC"
    ;;
  test)
    echo "[test] N=2, 3 reps at ${LAT}s/tool. Predicted running 1.260,"
    echo "[test] exclusion threshold 1.80, clear space 0.351."
    RC=0
    i=0
    for SEED in $EXP_COST_SEEDS; do
      i=$((i+1))
      run_arm 2 "$SEED" "$i" || RC=1
    done
    echo "[test] done, rc=$RC -> $EXP_OUT_DIR/adr0010"
    exit "$RC"
    ;;
  *)
    echo "usage: 60-concurrent.sh anchor | 60-concurrent.sh test" >&2
    exit 1
    ;;
esac
