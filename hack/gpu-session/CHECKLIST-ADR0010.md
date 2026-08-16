# Checklist sessione GPU — ADR-0010, il bound come capacità

Convenzioni invariate: decision-first, nodo mai acceso durante debugging,
workaround minimi a nodo acceso, root cause a nodo spento. Pin in
`00-env-cost-a10.sh` — nessun valore si cambia a nodo acceso.

**Sessione molto corta.** Traffico totale: 4,1 minuti (due bracci × 3
repliche × 40,5s). Con pre-flight ed engine, **20-25 minuti** e ~$0,50.
Il costo non è il tempo di calcolo: è arrivarci.

## 0. Prerequisiti (nodo SPENTO) — chiusi 2026-08-16
- [x] `driver/run_concurrent.py` provato contro llama.cpp, tre rami di D3
      esercitati con serie sintetiche (`b5dad9d`)
- [x] Scrape irrobustito: il nome della metrica deve finire dove comincia
      il campione, cinque casi verificati (`dd17dfc`)
- [x] `60-concurrent.sh` con i due gate, rehearsal a tre bracci (`9f16eb1`)
- [x] D4 emendato: cella 5,0 s/tool, N=2 — l'unica dove le bande di D3
      separano (gap +0,351; a 2,0 s/tool è −0,063, indecidibile)

## 1. Pre-flight (nodo ACCESO, engine SPENTO) — ~10 min
- [ ] rsync repo da optim-dev (MAI git clone sul nodo)
- [ ] `nvidia-smi` risponde, GPU A10 24GB
- [ ] venv: `pip install vllm==0.23.0`, e `driver/.venv` da `requirements.txt`
- [ ] **Pre-cache dei DUE tokenizer** — il generatore conta con quello del
      prefisso (`Qwen/Qwen2.5-0.5B-Instruct`), non con `--model`, e gira
      con `HF_HUB_OFFLINE=1`:
      `~/venv/bin/python -c "from transformers import AutoTokenizer; [AutoTokenizer.from_pretrained(m) for m in ('Qwen/Qwen2.5-7B-Instruct','Qwen/Qwen2.5-0.5B-Instruct')]"`
- [ ] `bash -n hack/gpu-session/60-concurrent.sh`
- [ ] **Dry-run degli argv**:
      `python3 driver/run_concurrent.py --dry-run --out-dir /tmp/dry --n 2 --model "$EXP_MODEL" --tool-latency-s 5.0`
      Verificare: due argv identici tranne `--steps-file` e `--seed`, e la
      riga del sampler su `/metrics` ogni 250ms

## 2. Engine — ~5 min
- [ ] **`PATH` deve contenere `~/venv/bin`** o vLLM muore con
      `FileNotFoundError: 'ninja'` durante `determine_available_memory`,
      dopo aver già caricato i pesi
- [ ] `--disable-log-requests` NON esiste in vLLM 0.23.0
- [ ] Avviare a mano: `PATH="$HOME/venv/bin:$PATH" nohup setsid ~/venv/bin/vllm serve "$EXP_MODEL" --port 8000 --enforce-eager`
      (qui l'engine manuale è corretto: nessun orchestratore ne possiede uno)
- [ ] Readiness via HTTP, non dal log. La riga "engine on :8000" compare
      anche a bind fallito
- [ ] **EngineCore PID**, non l'APIServer: `ps --ppid <apiserver> -o pid,comm`
      — il `comm` è troncato a `VLLM::EngineCor`. `export EXP_ENGINE_PID=<pid>`
- [ ] **ASSERT PRELIMINARE, 10 secondi**: la serie esiste ed è leggibile?
      `curl -s http://127.0.0.1:8000/metrics | grep '^vllm:num_requests_running'`
      Deve stampare una riga con label. Se è assente, l'esperimento non ha
      osservabile e **si aborta prima di spendere i bracci**

## 3. Braccio ANCORA (N=1) — ~2 min
`bash hack/gpu-session/60-concurrent.sh anchor`

- [ ] Atteso `observed_running_windowed` ≈ **0,630** (predetto), soglia di
      esclusione 0,90
- [ ] **ABORT se ≈ 1,0**: l'engine conta la richiesta come running anche
      durante le pause sui tool, l'osservabile di D2 non regge e il braccio
      di test misurerebbe altro. Costo per saperlo: 41 secondi
- [ ] ABORT se `error` su tutte e tre le repliche → nessuno scrape riuscito
- [ ] Annotare lo span osservato: se diverge molto da 40,5s, il predetto
      cambia e va ricalcolato prima del secondo braccio

## 4. Braccio TEST (N=2) — ~2 min
`bash hack/gpu-session/60-concurrent.sh test`

- [ ] Predetto **1,260**, soglia esclusione **1,80**, spazio libero 0,351
- [ ] Il verdetto lo scrive il codice (D3), non si interpreta a mano:
      `BOUND SUPPORTED` / `BOUND RULED OUT` / `INCONCLUSIVE`
- [ ] `INCONCLUSIVE` è un esito legittimo e si pubblica come tale
- [ ] Leggere anche `observed_running_untrimmed` accanto al ritagliato: la
      differenza è la dimensione dell'artefatto di coda, e va riportata

## 5. Evidenza off-box — PRIMA del teardown
- [ ] `rsync` di `$EXP_OUT_DIR/adr0010/` su optim-dev. Unità archiviabile:
      per braccio steps-file, meta, `running-series.json`, `argv.json`,
      `analysis.json`
- [ ] Sei directory attese (3 ancora + 3 test)
- [ ] `nvidia-smi` e `pip freeze` accanto ai risultati

## 6. Teardown
- [ ] Engine spento, istanza terminata dal dashboard
- [ ] Costo effettivo annotato

## Regola trasversale
Ogni deviazione si spiega **prima** di proseguire, o si annota come aperta.
Nota operativa: incanalare lo script in `tail` maschera il suo exit code —
leggere `rc` dalla riga che lo script stampa.
