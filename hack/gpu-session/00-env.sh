#!/usr/bin/env bash
# Pin di sessione GPU — agentic-kv-energy-experiment, matrice H100.
# Uso: source hack/gpu-session/00-env.sh   (sul nodo GPU, dentro il venv)
#
# Regola: NESSUN valore qui dentro si cambia a nodo acceso. Deviazioni =
# abort + root cause su optim-dev (convenzione item-4).
set -euo pipefail

# --- Pin software (verificati su optim-dev prima della sessione) ---
export EXP_MODEL="Qwen/Qwen2.5-32B-Instruct"
export EXP_VLLM_VERSION="0.23.0"          # stessa versione del digest operator sha256:6d8429e3
export EXP_INFERSCOPE_VERSION="0.3.0"     # release binary @ c0bebb9
export EXP_INFERSCOPE_BIN="$HOME/inferscope-bin/inferscope"   # rsync da optim-dev target/release

# --- Parametri calibrazione H1 (decisioni 2026-07-08, evidenza sim) ---
export EXP_CALIB_N_SESSIONS=8   # legittimato: mediana n=8 vs n=20 dista 0.006
                                # (sim-results/nsessions-check-n{8,20}, 8 rep per lato)
export EXP_CALIB_REPS=3
# Go/no-go sulla MEDIANA delle run di calibrazione:
#   verde  [0.40, 0.60] -> matrice
#   giallo [0.30,0.40)|(0.60,0.70] -> UNA sola run adattiva, poi verde o abort
#   rosso  fuori [0.30, 0.70] -> abort, nodo spento, root cause a freddo
# Divergenza intra-calibrazione > +-0.04 -> ROSSO a prescindere dalla mediana
# (inviluppo empirico a n=8: +-0.031 su 8 rep; +-0.03 del sim n=20 scatterebbe
#  falsi rossi su un sistema sano).

# --- Budget hard matrice ---
export EXP_MATRIX_RUNS_BASE=18
export EXP_MATRIX_RUNS_ADAPTIVE_MAX=3
export EXP_BUDGET_HOURS_MAX=5    # stima 3-5h H100 PCIe; a 5h si chiude comunque

# --- Da registrare A NODO ACCESO (append, non edit) ---
# nvidia-smi --query-gpu=driver_version,name --format=csv,noheader >> "$EXP_OUT_DIR/node-pins.txt"
# nvcc --version | grep release >> "$EXP_OUT_DIR/node-pins.txt"   # se presente
export EXP_OUT_DIR="$HOME/exp-results/$(date +%Y%m%d)-h100-matrix"

echo "env pinned: vllm=$EXP_VLLM_VERSION inferscope=$EXP_INFERSCOPE_VERSION model=$EXP_MODEL"
