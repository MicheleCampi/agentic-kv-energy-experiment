# PROTOCOL

Protocollo sperimentale: firma energetica (tokens/joule e divergenza dual-basis)
di un workload agentico ReAct-style attraverso il regime di KV-cache hit-rate.

## Tesi operativa

Il paper-fonte (arXiv:2605.26297) stabilisce che il workload agentico è
decode-dominated *condizionatamente all'hit-rate*, ma misura solo tempo/token.
Questo protocollo misura l'asse energia: come variano tokens/joule e la
divergenza prefill/decode al variare dell'hit-rate, e dove la firma collassa
quando la cache non regge (caso fallimento). L'ipotesi — da verificare sui dati,
NON assunta — è che esista un "ginocchio" nella curva tokens/joule vs hit-rate,
e che la divergenza dual-basis ne catturi la posizione/bruschezza.

## Invariante

- `enforce_eager=True` (CUDA graphs OFF) su tutte le run. Causa unica sulla
  curva: l'hit-rate è il knob, non i graph. Coerente con cuda-graphs e
  chunk-size experiments.

## Asse primario: regime di KV-cache hit-rate

L'hit-rate NON è un flag: è indotto dalla forma del traffico. Metodo:
- **caldo (high hit)**: N richieste condividono un lungo prefisso comune
  (system prompt + tool-def + storia accumulata); solo l'append per-turno è
  nuovo input. Realizza l'alto riuso (target hit-rate alto, ~90%+).
- **freddo (low hit)**: prefissi disgiunti tra richieste e/o prefix-cache
  disabilitato; ogni turno ricomputa. Realizza il basso riuso.
- **sweep**: livelli intermedi variando la frazione di prefisso condiviso.

### Composizione del contesto: prefisso cacheable + storia accumulata

Il contesto a regime (target 32-48K token) si compone di DUE componenti con
ruoli distinti nell'hit-rate — modellati separatamente perché è così che il
workload agentico reale è fatto (la fonte documenta che la quasi totalità
dell'input è riusata, solo l'append per-turno è nuovo: Input/Output ≫
Append/Output, §5):

- **prefisso cacheable condiviso** (~15K token, fisso): system prompt +
  tool-def. Artefatto deterministico versionato (`prefixes/agentic_system_v1`,
  14785 token misurati col tokenizer Qwen2.5, seed 42, 40 tool). È la base
  stabile, identica tra richieste → sempre cache-hit a regime caldo.
- **storia accumulata** (variabile, porta il contesto fino al target): turni
  precedenti, message, tool-call, observation. È la parte che CRESCE e di cui
  si modula la condivisione per realizzare l'hit-rate.

Realizzazione dei livelli in questa struttura:

| livello | prefisso | storia                         | hit-rate effettivo |
|---------|----------|--------------------------------|--------------------|
| H2 caldo| condiviso| largamente condivisa           | ~90%+              |
| H1 medio| condiviso| parzialmente disgiunta          | ~50%               |
| H0 freddo| disgiunto/off | disgiunta o ricomputo     | ~0%                |

La condizione **failure** opera su questa struttura: append di observation di
errore UNICO e gonfio (1.8× contesto, Fig. 6) che fa crescere la quota
non-cached della storia ed erode l'hit-rate effettivo anche col prefisso
condiviso — la rottura misurabile del decode-dominated.

Livelli hit-rate target (da calibrare/confermare sul simulatore, poi sul nodo):

| livello | descrizione                          | hit-rate target |
|---------|--------------------------------------|-----------------|
| H0      | cache fredda / disabilitata          | ~0% (CALIBRATE) |
| H1      | prefisso parzialmente condiviso      | ~50% (CALIBRATE)|
| H2      | prefisso largamente condiviso        | ~90%+ (CALIBRATE)|

3 livelli (estremi + centro) per individuare il ginocchio della curva
tokens/joule vs hit-rate con spesa-nodo minima. Campionamento adattivo: se la
calibrazione (sim) mostra il ginocchio cadere tra due livelli, si infittisce
UN quarto livello mirato lì — non una griglia a 4 punti a priori.

L'hit-rate REALIZZATO è misurato da inferscope (ADR-011), non assunto. I target
sopra sono obiettivi di calibrazione del generatore, non valori imposti.

## Asse secondario: stress di fallimento

Condizione che replica la firma del fallimento agentico (Fig. 6 della fonte:
i task falliti accumulano fino a 1.8× il contesto medio). Meccanismo: contesto
che gonfia con observation di errore ripetute (append non-cached continuo),
che erode l'hit-rate effettivo anche a prefisso condiviso. Punto in cui il
decode-dominated si rompe e il prefill ritorna a mordere.

| condizione | descrizione                                            |
|------------|--------------------------------------------------------|
| nominal    | traiettoria normale, append per-turno contenuto        |
| failure    | loop di errore: append non-cached gonfio (~1.8× ctx)   |

## Parametri del generatore — provenienza per-parametro

Fonte = arXiv:2605.26297. Modello di riferimento: Qwen (questo esperimento usa
Qwen2.5; il paper usa Qwen3.6-27B — divergenza dichiarata in PROVENANCE).

| parametro                  | valore (Qwen, dalla fonte)         | figura |
|----------------------------|------------------------------------|--------|
| turni/task (thinking)      | mean ~12–41 per benchmark          | Fig. 3 |
| turni/task (instant, SWE)  | mean 62.4, distrib. concentrata    | Fig. 3 |
| contesto accumulato (SWE)  | mean 68.7K–80.1K, max 146K–166K    | Fig. 4 |
| contesto (Terminal/GAIA)   | mean 52.5K–65.1K (Qwen)            | Fig. 4 |
| output: thinking (Qwen T)  | 29.0–40.7% dell'output             | Fig. 5 |
| output: tool-call (Qwen I) | 70.4–81.6% dell'output             | Fig. 5 |
| tempo LLM vs tool          | LLM 71–98%, tool 2–29% (GAIA max)  | Fig. 7 |
| Input/Output per turno     | ~120–560× (mean, workload-dep.)    | §5     |
| Append/Output per turno    | ~3.6–6.1× mean, ~0.7–1.4× median   | §5     |
| failure context inflation  | fino a 1.8× contesto medio         | Fig. 6 |

### Assunzioni dichiarate (dove la fonte dà intervallo, non forma)

- La FORMA della distribuzione interna a min/max/mean±std non è data dalla
  fonte. ASSUNZIONE: campionamento entro [min,max] centrato su mean con spread
  std (log-normale per le code lunghe dei turni; da fissare nel generatore).
- La sequenza temporale read/explore→execute/write (Fig. del paper) è
  qualitativa; ASSUNZIONE: non modellata nel generatore v1 (il segnale energia
  dipende da cache/contesto, non dal tipo semantico di tool). Rivedibile.
- Dimensione esatta delle observation per tipo di tool: non quantificata per
  Qwen; ASSUNZIONE: derivata dal rapporto Append/Output, non per-tool.

## Natura del workload: replay deterministico, non agente reale

Il generatore NON anima un vero agente LLM. Emette un **replay deterministico e
seeded** di una forma di traffico parametrizzata sulle distribuzioni della fonte
(turni, contesto, composizione, rapporto cache/append). Questa è una scelta
consapevole, non una semplificazione di comodo:

- **Razionale**: un agente reale è non-deterministico (stesso task → traiettorie
  diverse, come la fonte stessa nota). Quel non-determinismo distruggerebbe la
  riproducibilità della misura energetica. Un esperimento di caratterizzazione
  di sistema richiede una forma di carico riproducibile, non viva.
- **Cosa si misura**: la firma energetica della FORMA DI CARICO (rapporto
  cache/ricomputo, profondità contesto, ripartizione fase), ancorata token-per-
  token a distribuzioni misurate pubblicate — non il comportamento semantico
  dell'agente.
- **Limite dichiarato**: un revisore può obiettare "non è traffico agentico
  vero". La risposta è sopra — è la firma della forma, non dell'agente. Questo
  limite è esplicito, non nascosto. Lo stesso approccio dei benchmark di serving
  standard (es. `vllm bench serve` replica una distribuzione, non lancia agenti).

## Modelli

- **Qwen2.5-32B** (fissato). Razionale: realismo del regime agentico (contesti
  52-80K nel paper richiedono una taglia credibile), stessa famiglia del paper
  (Qwen), coerenza con i gemelli cuda-graphs/chunk-size che già girano 32B.
- **Target contesto operativo: ~32-48K token** (dentro il range del paper, sotto
  i massimi 146-166K che farebbero esplodere i tempi di run). Scelta esplicita
  per stare nel budget-nodo (3-5h), non limite nascosto. `max_model_len`
  dimensionato di conseguenza, da confermare sul nodo.

## Ripetizioni

3 rep per cella (coerente con la serie).

## Conteggio run

`n_hitrate × n_failure × n_model × 3 rep`.
Con 3 livelli hit × 2 condizioni × 1 modello × 3 rep = 18 run base
(+3 se si aggiunge il quarto livello adattivo = 21). Stima nodo: 3-5h su
Qwen2.5-32B / H100 PCIe. Si chiude a
livelli/modello fissati.

## Metodologia di cattura

inferscope NON modificato. Per ogni run: hit-rate realizzato (ADR-011), energia
per-fase + divergenza dual-basis (ADR-012), su finestra che bracketta il
traffico generato. Si preservano dalla serie: guard di troncamento finestra +
`active_fraction` (vivono dentro il report inferscope).

### Manifest per-run (riproducibilità da terzi)

Il generatore emette, per ogni run, un manifest machine-readable accanto al
report inferscope. Contenuto minimo:
- hit-rate **target** (livello H*) vs hit-rate **realizzato** (misurato ADR-011)
- token-count del prefisso condiviso (dal tokenizer Qwen2.5)
- profondità di contesto raggiunta (token)
- condizione (nominal/failure), rep
- **seed RNG** (la run è ribattibile bit-per-bit)
- parametri di campionamento usati (turni, append, composizione)

Insieme allo snapshot di provenance per-run, il manifest rende l'esperimento
riproducibile da un terzo: dichiara esattamente quale carico è stato generato,
con quale seme, e quanto il realizzato si discosta dal target. La separazione
target-vs-realizzato è onestà sperimentale di prima classe: non si assume che il
generatore abbia centrato il regime, lo si misura e si registra lo scarto.

## Validazione pre-GPU (costo zero)

Generatore + path di cattura sviluppati e validati su `llm-d-inference-sim`
(CPU-only, KV-cache abilitato) su `optim-dev`. Obiettivo della validazione:
(1) il generatore realizza i regimi di hit-rate target misurati via ADR-011;
(2) il path di cattura produce record completi. Output in `sim-results/`.
Solo dopo validazione in simulazione si passa al nodo GPU per la cattura energia.

## Tertiary axis: cost per agentic trajectory (phase 2)

The hit-rate matrix measures tok/J as cache reuse varies. This phase measures
something different on the same node: **how much of a GPU's price is paid
while that GPU is not generating**, as the latency of the tools the agent
calls varies.

The two phases do not merge into one orchestrator. ADR-013 wants **one**
trajectory in flight to attribute energy per step; the matrix wants N
concurrent sessions to move the hit-rate. The shapes are incompatible: two
sequential phases on the same node, not a factorial experiment.

### The denominator: trajectory span, not window span

The load-bearing decision, and the measurement that forced it.

`f_nongen` = (Σ tool durations + Σ gaps between consecutive steps) / span.

The **span is the trajectory's** — `last.end_elapsed_ns −
first.start_elapsed_ns` — not `run_duration_ns`, which is the span of the
sampling window. Both definitions are defensible on paper and they differ by
an order of magnitude on real data.

Evidence: `~/inferscope/validation-results/adr-013-a10-vllm/report-20260721T193436.txt`
(A10, Qwen2.5-7B, vLLM, 200 ms tools). Sampling starts 3.878s before the first
step and the window is 150s against a 6.218s trajectory — 24×. On that same
run: unattributed energy **91% of the window**, non-generating time **9.78% of
the trajectory**.

If the headline were "the fraction of the cost paid while the GPU is not
generating" and the denominator were the window, the published figure would be
dominated by an instrument oversizing artefact. With `EXP_WINDOW_MARGIN=1.2`
the inflation would stay at a structural 20% even with a well-calibrated
window.

`run_duration_ns` stays in the analysis as a **window-excess diagnostic**,
declared alongside the result, never as a denominator. A test pins the
published anchor (f_nongen 9.78%, packing bound 1.11).

### Anchoring to the source

The provenance table above gives `LLM vs tool time: LLM 71–98%, tool 2–29%`
(Fig. 7, GAIA at the top of the range). That is the interval `f_nongen` has to
fall inside for the measurement to be representative of real agentic traffic:
the 9.78% measured on 21/07 sits within it, towards the bottom. The latency
sweep is chosen to **cover the source's interval**, not to explore an
arbitrary range.
### The two policies, and the one the measurement falsifies

The decision arm (`driver/analyze_cost_decision.py`) evaluates two platform
policies over the same report. Both **dimensionless**, and therefore
independent of the declared price: the $/M token figure comes separately from
`inferscope cost` on the node, and the two readings do not contaminate each
other.

**P1 — release per segment.** Frees the GPU on every tool segment longer than
the re-entry price. Saving = Σ(d − C) over segments with d > C, where C is the
cold start **measured** in `vllm-coldstart-probe` (~18s, with the 27s/96s
finding declaring its variance). `--reentry-secs` is mandatory with no
default: the value must be visible in the command that produced the numbers.

On the 21/07 A10 anchor, P1 is **0.000s**. It is zero by construction: the
longest tool segment in the sweep is 5.0s against ~18s of re-entry. **The
obvious policy is falsified by the measurement, and that is the result to
publish** — not a negative outcome to bury. The domain of the claim travels
with it: the saving from interrupted occupancy is real only if the freed GPU
serves something else, and on a rental billed at hourly granularity it saves
nothing.

**P2 — packing.** Non-generating time is not freed: it is filled. Overlap
bound `1/(1 − f_nongen)`, i.e. how many trajectories one GPU can host before
the generating segments contend. On the anchor: 1.11. It is an **upper bound
under declared non-interference** — real batching changes throughput — and the
limit travels beside the number, not after it.

### Sweep, replicas, dispersion

Tool latency is a **parameter we chose**: what gets published is the curve
with its crossover, not a point. Four values (0.2 / 0.5 / 2.0 / 5.0 s per
tool) cover two orders of magnitude and the source's interval.

**Three replicas per cell at declared seeds**, not one run. With `max_tokens`
fixed the seed does not move the structure — it moves the generated text — so
the spread across replicas measures engine jitter on the LLM span, which is
the denominator. One run per cell would publish a point without knowing
whether it is distinguishable from its neighbour.

**The tool-latency CV is not a second sweep dimension.** `f_nongen` sums 3-5
durations and the mean dominates: variance does not move the curve. What it
moves is the **reliability of the bound under concurrency**. So it is one cell
of the sweep (2.0 s per tool) repeated at CV 0.5 — three extra cells, and the
difference between publishing a limit and publishing a limit with its
dispersion. Lognormal distribution parametrised by arithmetic mean and CV,
which is the only form in which the two flags mean what they say; at CV 0 the
behaviour is bit-identical to the cells already validated.
### Measurement arm and anchoring arm

The same conclusion the "deterministic replay, not a real agent" section
reached, arrived at again for a different measurement.

Driving the sweep with the real Deep Agents loop does not work, and the
measurement says so: one prompt, one model, temperature 0.0, three runs gave
spans of 7.4s / 12.7s / 34.7s with 4-8 steps. The model decides the shape of
the trajectory, so numerator and denominator both move for reasons unrelated
to the parameter being swept. A curve built on that arm would not be readable.

`run_replay.py` fixes the number of LLM calls, the number of tool steps and
the tokens generated per call, and leaves tool latency as the only variable.
Three repetitions at 0.2 s per tool: spans 11.1 / 10.9 / 11.0s — **1.8%
spread** against the agentic arm's factor of 4.7.

`run_trajectory.py` is not replaced: it **anchors** the replay. Run at the same
latency on the same node, it shows real trajectories landing in the region the
replay describes (31.5% against 36.3% at 2.0 s per tool in the llama.cpp
rehearsal, the gap explained by n_tool 2 against 3). Two arms, each proving
what it can.

Both write the same format through the same `StepsFileCallback` and pass the
same gates in `trajectory_gates.py`: a gate that differed between the arms
would make the anchoring cells useless as evidence.

### Irreversible parameters, per cell

Two, and neither correctable after the fact:

- **The sampling window.** Sized from the span of a calibration trajectory
  measured on the node, per cell: (span + that cell's tool wall) × margin. Not
  one window for the whole sweep: at 5.0 s per tool the trajectory is ~14s
  longer than at 0.2s.
- **The steps-file.** On `--sample-only` the trajectory is derived once, in
  flight, and cannot be re-joined afterwards.

Hence: the **cell directory is the archivable unit** (steps-file, meta,
report, argv, cost, decision). Without its steps-file a report cannot be
re-analysed, because the trajectory inside it is already joined and a join
defect is no longer diagnosable.

Hence also: the **price is derived cell by cell on the node**, never at the
end of the campaign. That is the only moment at which an abstention by `cost`
is still diagnosable.
### Measured results — A10 session, 2026-08-04

Lambda us-east-1, 1× A10 24GB PCIe at $1.29/h, Lambda Stack 24.04, vLLM
0.23.0, Qwen2.5-7B-Instruct, `--enforce-eager`, inferscope 0.5.0. Evidence in
`exp-results/20260804-a10-cost/`: sixteen cell directories, eight files each,
with provenance (`nvidia-smi`, `pip freeze`, binary version, instance
description) archived alongside.

**Phase 1 — ADR-011 on real vLLM.** The first KV-cache reading against real
vLLM in this repo; until now the claim existed only against the simulator.

| regime | hits_delta | queries_delta | realized | mean util. |
|--------|-----------:|--------------:|---------:|-----------:|
| H0     |          0 |       251,320 |    0.000 |        95% |
| H2     |    203,456 |       235,209 |    0.865 |        52% |

H0 gives exactly zero by construction — disjoint prefixes, no reuse possible —
and H2 realises 0.865. The two regimes separate by 0.865 in absolute terms.

Declared limit: the H0 cell ran for 115.7s against a 90s window (129%
filled), so its energy is truncated. The hit-rate assert is untouched by this
— hits and queries come from the Prometheus scrape before and after the cell,
not from the window — but **no tok/J should be derived from H0**.

An incidental finding, not sought: H2 runs at 52% GPU utilisation against
H0's 95%. With a warm cache the GPU does less work for the same volume of
tokens, which is part of why the packing bound is interesting.

### The curve

Fifteen cells: four latencies × three replicas at declared seeds, plus three
dispersion cells at CV 0.5. Window fill between 83% and 87% on all of them
(guard interval 60-90%). `gaps = 0.00%` of span everywhere: the framework
overhead between steps is below the resolution.

| tool latency | f_nongen | packing bound | $/M token (span) |
|-------------:|---------:|--------------:|-----------------:|
| 0.2 s        |    2.30% |          1.02 |          $12.16  |
| 0.5 s        |    5.55% |          1.06 |          $12.62  |
| 2.0 s        |   19.01% |          1.23 |          $14.72  |
| 5.0 s        |   37.01% |          1.59 |          $18.91  |

The LLM span holds **constant at 25.5s** across all fifteen cells: only the
added term moves, which is the condition for the curve to be readable and the
reason the measurement arm has fixed structure.
### The cost of generating does not move: what you pay for is the waiting

The figure that makes the curve hard to argue with is the per-cell
decomposition of the price, at the declared rate of $1.29/h:

| tool latency | generating cost | tool cost  | ratio |
|-------------:|----------------:|-----------:|------:|
| 0.2 s        |       $0.009127 | $0.000215  |    1× |
| 0.5 s        |       $0.009155 | $0.000538  |  2.5× |
| 2.0 s        |       $0.009158 | $0.002150  |   10× |
| 5.0 s        |       $0.009147 | $0.005375  |   25× |

The cost of generating is **flat across all fifteen cells within a 0.5% band,
$0.00911-0.00916** — the same 768 tokens, the same GPU, the same model. The
band is quoted rather than a rounded single figure: on a claim that a quantity
does not move, how much it does not move is the claim. All of the price
growth, +56% from $12.16 to $18.91 per M token, is time during which the GPU
is allocated and not generating.

### A reading constraint on the $/M token figure

The `$/M gen tokens` figure `inferscope cost` prints is computed **over the
sampling window**, not over the trajectory span. The difference is not
academic: the three CV 0.5 cells share the same 38s window and print
$17.7069 / $17.7068 / $17.7068 — indistinguishable — while their `f_nongen`
differs by 4.09 points.

The window is an instrument parameter we chose. Therefore:

- the figures in the table above are recomputed **over the span**, which is
  the only form comparable across cells and defensible in review;
- the window-based `$/M` remains valid as the cost of a rental interval
  actually occupied, but it is **not** a list price and must not be presented
  as one;
- the `generating` / `in tools` split that `cost` prints is by contrast
  instrument-independent, and indeed reproduces the measured `f_nongen`
  exactly (20.7% / 19.6% / 16.6% across the three CV cells).

**Correct headline**: not "$14.72 per M token", but *19.0% of the attributed
cost is paid while the GPU sits idle on tools, rising to 37.0% at 5 s per
tool*.

### The two policies, tested

**P1 = 0.000s across all fifteen cells.** The longest tool segment in the
sweep is 5.0s against a measured re-entry price of ~18s: no segment repays
the release. The obvious policy — free the GPU while the agent waits on a
tool — is falsified by the measurement across the whole latency interval the
source covers. Break-even would fall beyond 18 s per tool, outside the
published interval (Fig. 7, tools 2-29% of the time).
**P2, packing bound**: from 1.02 to 1.59 across the interval. It remains an
upper bound under declared non-interference.

### The dispersion branch, and what it demonstrated

The design argued that the tool-latency CV does not move the curve but does
move the reliability of the bound. The measurement quantifies it.

| cells                | mean f_nongen | spread across replicas |
|----------------------|--------------:|-----------------------:|
| 2.0 s/tool, CV 0     |        19.01% |          **0.006 pt** |
| 2.0 s/tool, CV 0.5   |        18.95% |          **4.09 pt**  |

Mean unchanged within 0.06 points, spread **three orders of magnitude**
larger. The packing bound at 2.0 s per tool is not 1.23: it is **1.23 ranging
1.20-1.26 under CV 0.5**, and on price that translates to $15.02 / $14.81 /
$14.30 per M token against ±0.03% for the deterministic cells.

Without those three cells we would have published a limit; with them we
publish a limit and its dispersion. They cost three cells out of fifteen.

It also serves as a check on the measurement arm: at CV 0 the three replicas
give spans identical to the tenth (27.8 / 32.3 / 41.3 s by latency), so the
dispersion observed at CV 0.5 comes from the sampling and not from the engine.

### Session cost

~25 minutes of node, ~$0.55 against a declared cap of $1.94. Node time was
never the constraint: the full sweep occupies ~10 minutes of window. The
constraint was cells, and none was wasted.

An earlier session the same day was abandoned after ~$0.90 on four
environmental obstacles — the prefix tokenizer not cached (the generator asks
for the prefix's, not `--model`), `PATH` without `~/venv/bin` and therefore
`ninja` not found, an engine started by hand holding the port the
orchestrator's own engine needed, and a flag that does not exist in vLLM
0.23.0. None was a design defect; all four are now entries in
`CHECKLIST-A10-COST.md` with their remedy, and that is why the second session
ran without interruption.
### What this phase does NOT prove

- It is not real agentic traffic (the limit declared above applies: it is the
  signature of the shape, anchored by the agentic arm).
- The packing bound is an upper bound under **declared** non-interference: it
  is not a throughput measurement under real concurrency.
- P1 = 0 holds in the declared domain (single-tenant, hourly rental
  granularity). It is not a general statement about the futility of releasing.
- The tok/J are not comparable with July's H100 matrix: different GPU,
  different model. The comparison is internal to the session.

## Fuori scope

Replica della caratterizzazione task del paper (accuratezza, success rate).
Questo esperimento misura SOLO la firma energetica della forma di carico.

## Calibrazione H1 — storia completa (chiusa 2026-07-05)
Tre root cause verificate in sequenza, ciascuna fino alla riga di sorgente,
con predizione quantitativa dove possibile.

**1. Artefatto di scala del tokenizer** (pavimento ~0.84 con storia
interamente unica). Il generatore dimensiona in chars/4 (BPE reale); il sim
usa SimpleTokenizer (pkg/tokenizer/tokenizer.go, regex word-level: `\w+`
ingloba underscore) -> l'unita' `UNIQ_s3_b7_obs_xxxx` = 1 token-sim, storia
compressa ~6x nello spazio-token sim, prefisso no -> quota-prefisso ~0.81 =
pavimento. Predizione verificata: 0.812 vs 0.837 misurato (+2.5pt: counter
Prometheus alimentati da membership any-position via startRequest, non dal
prefix-truncated countCachedBlockPrefix — divergenza semantica dal vLLM
reale, secondaria). Ipotesi intermedia "contabilita' content-addressed su
filler ripetitivo" falsificata da esperimento di controllo prima della root
cause. FIX: filler granulare — 1 parola hex da 3 chars senza underscore =
1 token-sim = ~1 token BPE, sizing e conteggio coincidono per costruzione.
Floor probe post-fix (frac=0.0, prefisso condiviso): 0.448/0.476 —
quota-prefisso pura ~0.46, predizione ~0.45 confermata.

**2. Knob interleaved concettualmente rotto sotto hashing a catena.** Il sim
delega a llm-d-kv-cache prefixHashes (pkg/kvcache/kvblock/
token_processor.go:130-134): `prefix = hash(prefix, chunk)` — l'hash di ogni
blocco incorpora la catena dei precedenti. Un blocco shared mid-history
preceduto da blocchi unique divergenti ha hash diverso in ogni sessione: la
condivisibilita' mid-history NON ESISTE, a qualunque frac (il draw Bernoulli
per-turn del design originale non poteva funzionare). vLLM reale usa lo
stesso schema di hashing cumulativo: il ridisegno vale identicamente per il
GPU run. FIX: shared-HEAD design — history_shared_frac = frazione del
budget-token di storia coperta da una testa condivisa CONTIGUA al prefisso
(primi turni identici cross-session per costruzione, poi divergenza;
semanticamente piu' fedele a traiettorie agentiche con bootstrap comune).
Troncamento token-preciso dell'ultimo blocco shared al budget residuo:
senza, la testa e' quantizzata a blocchi interi (massa fissa 1449 tok,
insensibile al frac nell'intervallo ~0.03-0.09, misurato). Residuo di
quantizzazione post-fix: le ~4 righe template per blocco (~15 tok), non
clampate.

**3. Collisione della derivazione seed additiva.** seed_base+rep+nonce
collide per qualunque coppia (nonce, rep) a somma uguale: scoperta da un
realizzato identico a 16 cifre cross-campagna (nonce 20260709 rep3 ==
20260710 rep2 -> prompt identici al bit, realizzato inquinato dalla cache
warm). FIX: derivazione multi-asse prime-weighted
(nonce*1000003 + regime_idx*10007 + cond_idx*101 + rep), nessuna tupla
collide. Evidenza della collisione conservata in runs/matrix-recal-confirm.

**Valori congelati e matrice di conferma** (runs/matrix-recal-full, nonce
20260711, 18/18): H1 history_shared_frac=0.065. H0=0.000 esatto 6/6 (prefissi
disgiunti reggono anche col filler granulare); H1 nominal 0.484-0.518 (media
0.505), failure 0.471-0.502 (media 0.488); H2 nominal ~0.940, failure ~0.922.
Monotonia ovunque; failure < nominal in H1/H2 (la massa unique bloated dei
blocchi failure diluisce i hit — direzione fisicamente attesa). La varianza
per-sessione della frazione shared (sessioni corte da clamp turns_min pesano
la testa fissa fino a ~3x) e' proprieta' accettata del disegno: il regime e'
definito dalla media di campagna, che e' cio' che i counter aggregano.
La coordinata pubblicata resta il realizzato su vLLM reale, calibrato con
2-3 run corte a inizio sessione GPU prima della matrice.

## Nota di validazione (verificata sul sim, 2026-06-29)

Confine accertato di ciò che `llm-d-inference-sim` v0.8.2 può validare:
- **Valida** l'asse hit-rate: espone `vllm:prefix_cache_hits` e
  `vllm:prefix_cache_queries` (counter) → hit-rate via window-delta (ADR-011).
  Riscontro concreto: fixture esistente mostra che 12 completion con prefisso
  lungo condiviso producono hit_rate=0.490 misurato. Il meccanismo dell'asse
  primario è già osservato funzionare su questo sim.
- **NON valida** l'asse energia: il sim è CPU-only, non espone energia NVML né
  separazione prefill/decode time. tokens/joule e divergenza dual-basis si
  misurano SOLO sul nodo GPU. La fase sim prova che il generatore realizza i
  regimi di hit-rate target; non prova nulla sulla firma energetica.

## Confini di validita' della conferma sim
La matrice 18/18 in sim valida il GENERATORE (calibrazione hit-rate,
determinismo, trim bisezione), NON la topologia di esecuzione GPU. Fuori
copertura sim, verificati solo a nodo acceso o per audit statico:
- engine unico persistente tra celle (effetto d'ordine prima-cella:
  prefix freddo una volta per avvio — mitigato con warm-up cell)
- ramo inferscope (contratto CLI, feature set del binario, NVML)
- rete a runtime (HF Hub per pesi e tokenizer)
- semantica --sample-secs (timer fisso, costo idle per cella)
Ogni claim "validato in sim" implica SOLO la prima categoria.
