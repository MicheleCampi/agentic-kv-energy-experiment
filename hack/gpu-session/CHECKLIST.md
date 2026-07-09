# Checklist sessione GPU — matrice H100 (agentic-kv-energy-experiment)

Convenzioni: decision-first, nodo mai acceso durante debugging, workaround
minimi a nodo acceso, root cause a nodo spento. Pin in `00-env.sh` — nessun
valore si cambia a nodo acceso.

## 0. Prerequisiti (optim-dev, nodo SPENTO) — già chiusi 2026-07-08
- [x] Verifica sim n-sessions: mediana n=8 vs n=20 dista 0.006 (<=0.02)
      -> calibrazione a n=8 legittima. Evidenza: `sim-results/nsessions-check-n{8,20}`
- [x] Regressione sim GPU-mode passata (cea4bf2): H0=0.000, H1 in inviluppo, H2=0.933
- [x] Pin inferscope: 0.3.0 @ c0bebb9, release binary

## 1. Pre-flight (nodo ACCESO, engine SPENTO) — ~15 min
- [ ] `python3 agentic_workload.py --help | grep bpe-counter` presente
      (guardia: rsync di una working copy pre-fix = 400 garantito al primo
      prompt, vedi smoke-abort-20260709)
- [ ] rsync repo + inferscope binary da optim-dev
      (`-e "ssh -i ~/.ssh/runpod_optimdev"`; MAI git clone sul nodo)
- [ ] venv bare, `pip install vllm==0.23.0`
- [ ] Coerenza venv: l'orchestrator DEVE girare dallo stesso venv dell'engine
      — `pip-freeze.txt` gira da `sys.executable`; orchestrator fuori dal
      venv = freeze che mente. Check: `which python3` = `$VIRTUAL_ENV/bin/python3`
- [ ] `$EXP_INFERSCOPE_BIN --version` = 0.3.0
- [ ] `source hack/gpu-session/00-env.sh`; driver/CUDA -> `node-pins.txt`
- ABORT se: versioni non combaciano, venv incoerente. Fix banale (<5 min) ok,
  altrimenti nodo spento.

## 2. Readiness engine — ~15 min (32B cache fredda, timeout 900s)
- [ ] `vllm serve` via orchestrator (`--enforce-eager` hardcoded, resto via
      `--engine-args`), probe `/health`
- [ ] EngineCore PID trovato via `pgrep -g <pgid> -f VLLM::EngineCore`
      (attach all'APIServer = GPU idle, artefatto noto da cuda-graphs)
- [ ] Lo smoke DEVE includere una cella che costruisce fino a effective
      target: verifica accettazione al limite del contesto (HTTP 400 = bug
      di budgeting, non crash: causa propria, root cause a freddo)
- ABORT se: timeout readiness, PID EngineCore non trovato, crash engine o
  HTTP 4xx al primo prompt di smoke. Niente retry-loop a nodo acceso: un
  retry secco (solo per errori transienti, MAI per 4xx deterministici),
  poi abort.

## 3. Calibrazione H1 — ~30-45 min
- [ ] 3 run corte: `--regimes H1 --reps $EXP_CALIB_REPS --n-sessions $EXP_CALIB_N_SESSIONS`
- [ ] Annotare wall-time per run -> dimensionare `--sample-secs` per regime
      (finestra > wall-time generatore + margine; warning manifest se >90%)
- [ ] Go/no-go sulla MEDIANA (criteri in 00-env.sh):
      verde [0.40,0.60] -> matrice | giallo -> UNA run adattiva | rosso -> abort
- [ ] Divergenza intra-calibrazione > +-0.04 -> ROSSO anche con mediana verde
- [ ] Se run adattiva: nuovo `history_shared_frac` in provenance come
      deviazione documentata dal frozen sim
- ABORT se: rosso, o la run adattiva non rientra in verde.

- NOTA --sample-secs: in BPE mode i turni/sessione scendono (~3.37
  tok/word sul filler vs proxy word-level) e il wall-time cella cambia
  rispetto alla stima sim: dimensionare SOLO dal wall misurato in
  calibrazione, mai dai wall storici sim.
## 4. Matrice — 3-5h budget hard
- [ ] 18 run base + max 3 adattive (`$EXP_MATRIX_RUNS_*`)
- [ ] Manifest per cella: delta hit-rate + blocco gpu (window/offset/wall/pid/json)
- [ ] Provenance run-level: `engine.json` + `pip-freeze.txt`
- ABORT se: budget 5h sforato (si chiude con le celle complete), >=2 crash
  engine non spiegati, o inferscope.json mancante/vuoto su 2 celle consecutive.

## 5. Evidenza off-box — PRIMA del teardown
- [ ] rsync `$EXP_OUT_DIR` + `node-pins.txt` -> optim-dev
- [ ] Verifica sul LATO optim-dev: count file + size totale combaciano
- [ ] Commit dalla VM, mai dal nodo

## 6. Teardown
- [ ] stop_engine (già in finally, verificare processo morto)
- [ ] Nodo spento SOLO dopo verifica rsync lato optim-dev
