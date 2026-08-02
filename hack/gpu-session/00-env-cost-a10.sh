#!/usr/bin/env bash
# Pin di sessione GPU — campagna COSTO su A10 (ADR-015).
# Uso: source hack/gpu-session/00-env-cost-a10.sh   (sul nodo GPU, nel venv)
#
# Regola invariata: NESSUN valore qui si cambia a nodo acceso. Deviazioni =
# abort + root cause su optim-dev.
#
# Questa NON e' la matrice di luglio. Modello e GPU sono diversi, quindi i
# tok/J non sono confrontabili con il +69% H0->H2 di 00-env.sh: quel numero
# viene da Qwen2.5-32B su H100 SXM5. Qui il confronto e' interno alla
# sessione, H0 contro H2 sullo stesso nodo e sullo stesso binario.

set -euo pipefail

# --- Pin software ---
export EXP_MODEL="Qwen/Qwen2.5-7B-Instruct"   # entra in A10 24GB; stesso
                                              # modello di ADR-013 (21/07)
export EXP_VLLM_VERSION="0.23.0"              # digest pinnato item-4
export EXP_INFERSCOPE_VERSION="0.5.0"         # >= cd0ece6: prima build in cui
                                              # lo schema KV nomina le serie
                                              # come vLLM le ESPONE, e prima
                                              # con il sottocomando `cost`
export EXP_INFERSCOPE_BIN="$HOME/inferscope-bin/inferscope"

# --- Obiettivo e assert di sessione ---
# ASSERT PRIMARIO (PASS/FAIL, deciso a nodo spento):
#   cache_queries_delta > 0 AND cache_hits_delta > 0 in un report inferscope.
#   E' la prima lettura KV su vLLM reale in questo repo. Se FAIL: catturare
#   /metrics grezzo + report + stderr off-box, teardown, root cause a freddo.
# ASSERT SECONDARIO: hit-rate H2 > hit-rate H0 con margine oltre la
#   divergenza intra-regime. Senza questo il costo non ha due bracci.

# --- Calibrazione: --sample-secs si MISURA, non si indovina ---
# La cella di warm-up gira con finestra generosa; si legge
# generator_wall_s dal manifest e si fissa la finestra a wall x 1.2 per
# le celle misurate. Il 20% copre la varianza H0 (piu' lento, nessun
# riuso) e tiene il riempimento sopra l'80%, dentro entrambe le soglie
# di window_warning. Su A10 con 7B il wall e' ignoto a priori: e' il
# motivo per cui la calibrazione precede la matrice.
export EXP_WARMUP_SAMPLE_SECS=180   # solo per il warm-up, scartato
export EXP_WINDOW_MARGIN=1.2

export EXP_CALIB_N_SESSIONS=8
export EXP_CALIB_REPS=3

# --- Matrice ridotta: due bracci, non tre regimi ---
# H1 non serve al confronto di costo: il rapporto H2/H0 e' indipendente
# dal rate dichiarato ed e' cio' che si pubblica. H1 resta disponibile
# se la calibrazione mostra che H0 e H2 non si separano.
export EXP_REGIMES="H0,H2"
export EXP_REPS=3
export EXP_BUDGET_HOURS_MAX=2       # 1x A10 ~$0.75/h -> cap ~$1.50,
                                    # ampiamente sotto il tetto 30 EUR

export EXP_OUT_DIR="$HOME/exp-results/$(date +%Y%m%d)-a10-cost"
echo "env pinned: vllm=$EXP_VLLM_VERSION inferscope=$EXP_INFERSCOPE_VERSION model=$EXP_MODEL regimes=$EXP_REGIMES"
