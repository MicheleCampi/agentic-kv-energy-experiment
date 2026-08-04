# Checklist sessione GPU — campagna COSTO su A10 (agentic-kv-energy-experiment)

Convenzioni: decision-first, nodo mai acceso durante debugging, workaround
minimi a nodo acceso, root cause a nodo spento. Pin in `00-env-cost-a10.sh`
— nessun valore si cambia a nodo acceso.

**Questa NON è la checklist della matrice H100** (`CHECKLIST.md`). GPU,
modello e forma sperimentale sono diversi: due file separati perché una
checklist a due colonne costringe a scegliere quale riga applicare mentre
il tassametro corre.

Budget: 1× A10 ~$0.75/h, cap 2h → ~$1.50. Fase 1 ~10 min, fase 2 ~5,2 min
di sola finestra: il tempo di nodo non è il vincolo, **le celle spese sì**.

## 0. Prerequisiti (optim-dev, nodo SPENTO) — chiusi 2026-08-04
- [x] Contract check estratto in `driver/inferscope_contract.py`, sei
      asserzioni, condiviso dai due orchestratori (`d34330e`)
- [x] `tool_wall_s` misurato dagli step, non `n_tool × tool_latency_s`
      (`c20a484`; 22 test verdi fra le due suite)
- [x] `driver/run_cost_cell.py` provato end-to-end contro llama.cpp:
      finestra 45s, replay dentro, sentinella sovrascritta, abort su
      `"gpu": null` (`778cdfa`)
- [x] `50-cost.sh` + pin fase 2, tre gate verificati (`db5eef2`)
- [x] Dress rehearsal `40-rehearsal.sh` PASS dopo il cablaggio: tre celle,
      contract check silenzioso, warm-up scartato

## 1. Pre-flight (nodo ACCESO, engine SPENTO) — ~10 min
- [ ] rsync repo + binario inferscope da optim-dev (MAI git clone sul nodo)
- [ ] `nvidia-smi` risponde e la GPU è una A10 24GB
- [ ] Contratto inferscope sul binario TRASFERITO:
      `for f in --gpu --engine --steps-file cost; do ~/inferscope-bin/inferscope --help 2>&1 | grep -q -- "$f" && echo "OK $f" || echo "MISSING $f"; done`
      Quattro `OK`, nessun `MISSING` (il `--` dentro grep separa le
      opzioni, non e' un flag da cercare). Un `grep -c` con
      alternative conterebbe le RIGHE che
      matchano, non i flag distinti: darebbe 4 anche mancandone uno.
      L'help esce su **stderr con exit 2** (clap): leggere entrambi i
      flussi, non solo stdout (root cause 2026-08-02).
      Lo stesso contratto gira automatico in `inferscope_contract.py`
- [ ] venv bare, `pip install vllm==0.23.0`, e `driver/.venv` con
      `requirements.txt` (il replay importa `openai`)
- [ ] `bash -n hack/gpu-session/50-cost.sh` e `python3 -m py_compile
      driver/run_cost_cell.py`
- [ ] **Dry-run degli argv PRIMA di accendere l'engine**:
      `python3 driver/run_cost_cell.py --dry-run --out-dir /tmp/dry --model "$EXP_MODEL" --tool-latency-s 0.2 --reentry-secs 18 --usd-per-hour 0.75`
      Verificare a occhio: `--steps-file` nella dir di cella, `--gpu`
      presente, `--engine vllm` accanto a `--metrics-endpoint`, una sola
      base di prezzo

## 2. Readiness engine — ~5 min (7B, cache fredda)
- [ ] `vllm serve` dal venv, `--enforce-eager` come invariante di matrice
- [ ] `wait_ready` passato. **La riga "engine on :8000" NON è evidenza**:
      il fake la stampa prima di bindare, e il 2026-08-04 è comparsa con
      il bind fallito. L'unico segnale è la readiness HTTP
- [ ] EngineCore PID risolto (NON il parent APIServer: attaccarsi lì
      campiona una GPU inattiva) → `export EXP_ENGINE_PID=<pid>`
- [ ] Nessun residuo sulla :8000 da sessioni precedenti (`ss -ltnp`)

## 3. Fase 1 — assert ADR-011 su vLLM reale — ~10 min
- [ ] `30-matrix.sh` con `--sample-secs` dimensionato dal warm-up
- [ ] **ASSERT PRIMARIO**: `cache_queries_delta > 0 AND cache_hits_delta > 0`
      in un report. È la prima lettura KV su vLLM reale in questo repo
- [ ] ABORT se `"gpu": null` → NVML non ha attaccato. Node-off è atteso,
      **a nodo acceso è abort**

## 4. Calibrazione fase 2 — ~2 min, UNA traiettoria
`bash hack/gpu-session/50-cost.sh calibrate`

Tre criteri di abort, in ordine. Nessuna cella dello sweep parte prima che
tutti e tre siano verdi.

- [ ] **`"gpu": null` → ABORT.** NVML non ha attaccato; ogni cella
      successiva sarebbe a energia zero
- [ ] **Sezione `trajectory` assente → ABORT.** È il criterio che questa
      sessione esiste per esercitare: il join `--sample-only --steps-file`
      **non è certificabile a nodo spento** (`gpu_timeline` è `None` senza
      NVML e la derivazione si astiene su assenza, verificato a sorgente
      `main.rs:530` il 2026-08-04). Qui è l'unico posto dove si prova, e
      costa una traiettoria invece di quindici celle
- [ ] **Span LLM fuori dall'intorno del rehearsal → RICALIBRARE le
      latenze, non proseguire.** `f_nongen` dipende dal rapporto
      tool/LLM: se lo span LLM su A10/Qwen2.5-7B differisce dagli 11,197s
      del rehearsal llama.cpp, le latenze pinnate cadono su punti diversi
      della curva. Si aggiorna `EXP_COST_LATENCIES` nel pin **prima** di
      spendere celle
- [ ] Annotare `observed_span_s` dal `.meta.json`: è l'argomento di
      `50-cost.sh sweep <span-secs>`
- [ ] Decisione repliche: con lo span reale, ricalcolare il costo dello
      sweep. Se resta ampiamente sotto budget, portare `EXP_COST_SEEDS` da
      3 a 5 stringe l'intervallo sulla dispersione di `f_nongen` — il
      numero che decide se i punti della curva sono distinguibili

## 5. Sweep fase 2 — ~10 min, 15 celle
`bash hack/gpu-session/50-cost.sh sweep <span-secs>`

- [ ] La finestra è **irreversibile per cella** e non correggibile a
      posteriori. Lo script la deriva dallo span di calibrazione: non
      passarla a mano
- [ ] Lo steps-file è **irreversibile per cella**: `--sample-only` deriva
      la traiettoria una volta sola, in volo. Non è ri-joinabile dopo
- [ ] Il prezzo si deriva **cella per cella sul nodo**, mai a fine
      campagna: è l'unico momento in cui un'astensione di `cost` è
      diagnosticabile. Lo fa `run_cost_cell.py`; verificare che
      `cost.log` esista in ogni dir di cella
- [ ] Riempimento finestra fra 60% e 90%: sotto, la finestra è
      sovradimensionata e il denominatore energetico si gonfia; sopra, la
      traiettoria rischia il troncamento
- [ ] Una cella fallita **non si rifà cambiando parametri**: si annota,
      si prosegue, root cause a nodo spento

## 6. Evidenza off-box — PRIMA del teardown
- [ ] `rsync` dell'intera `$EXP_OUT_DIR` su optim-dev. **La directory di
      cella è l'unità archiviabile**: `steps.jsonl`, il suo `.meta.json`,
      `inferscope.json`, `argv.json`, `cost.log`, `decision.json`. Senza
      lo steps-file il report non è ri-analizzabile — la traiettoria
      dentro è già joinata, quindi un difetto di join non è più
      diagnosticabile
- [ ] Verificare il conteggio delle dir di cella a destinazione prima di
      spegnere: 15 attese, più `calib`
- [ ] `nvidia-smi` e `pip freeze` archiviati accanto ai risultati

## 7. Teardown
- [ ] Engine spento, poi istanza terminata dal dashboard
- [ ] Costo effettivo annotato contro il cap di $1,50

## Regola trasversale — anomalie spiegate
Ogni deviazione dal previsto si spiega **prima** di proseguire, o si
annota come aperta. Un'anomalia non spiegata a nodo acceso diventa un
numero non difendibile a nodo spento.

L'eco del terminale mente sistematicamente: verificare sempre dal file,
mai dall'output rincollato (quattro falsi allarmi il 2026-08-04, tutti
smentiti dal file).

## Backlog strumentazione (non bloccante)
- `inferscope --help` esce 2 stampando su stderr (clap
  `DisplayHelpOnMissingArgumentOrSubcommand`): il contract check legge
  entrambi i flussi, la causa è in inferscope
- `--sample-only` con `--steps-file` e senza GPU si astiene in silenzio.
  Corretto per disciplina, ma una diagnostica che nomini la causa
  dell'astensione risparmierebbe la deviazione fatta il 2026-08-04
