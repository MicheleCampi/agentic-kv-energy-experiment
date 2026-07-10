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
- [ ] Contratto inferscope sul binario TRASFERITO (identita' = versione +
      commit + FEATURES, root cause 2026-07-10: build senza gpu-nvidia
      nasconde --gpu e NVML non parte mai — campagna a zero energia):
      `~/inferscope-bin/inferscope --help | grep -q '\-\-gpu' && echo FEAT_OK`
      (l'orchestrator ha lo stesso check fail-fast prima del launch engine)
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
- [ ] Warm-up cell post-readiness (automatica nell'orchestrator, dir
      warmup/, discarded): scalda il prefix di sistema — la PRIMA richiesta
      dopo l'avvio engine lo manca per intero (~0.06 di depressione sulla
      prima cella misurata, root cause 2026-07-10). Vale a OGNI avvio engine.
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
- [ ] Go/no-go ESEGUIBILE: `python3 hack/gpu-session/gonogo.py <calib-dir>`
      (exit 0 verde / 2 giallo / 1 rosso; formula spread = max-min > 0.04.
      MAI valutazione a occhio dei criteri a nodo acceso)
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

## 5b. Script di sessione (OBBLIGATORI — mai comandi a mano)
- Fasi 2-4 SOLO via `10-smoke.sh` / `20-calib.sh` / `30-matrix.sh <secs>`:
  tutti i parametri scritti (rehearsal 2026-07-10: 3 lanci su 4 composti a
  memoria erano sbagliati). `30-matrix.sh` rifiuta di partire senza il
  sample-secs dimensionato dalla calibrazione.
- Non-regressione sim pre-sessione (da optim-dev, nodo spento), parametri
  SCRITTI: `--sim --model facebook/opt-125m --regimes H0,H1,H2 --reps 1
  --n-sessions 8 --target-context 32768 --prefix-version v1` — attesi
  H0=0.0 esatto, H1 in [0.40,0.60], H2>0.90, monotonia stretta.
  (prefix sim o target ridotti = composizione degenere, regime falsato)

## 6. Teardown
- [ ] stop_engine (già in finally, verificare processo morto)
- [ ] Nodo spento SOLO dopo verifica rsync lato optim-dev

## Regola trasversale — anomalie spiegate
Ogni anomalia osservata e spiegata (anche se derubricata come benigna)
OBBLIGA alla domanda scritta prima di procedere: "questo meccanismo dove
altro agisce nelle fasi successive?" — risposta annotata nel log di
sessione. (Il cold-start visto nello smoke H2 del 2026-07-10 spiegava
gia' il rosso di calibrazione un'ora prima che accadesse.)
