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

## Asse terziario: costo per traiettoria agentica (fase 2)

La matrice hit-rate misura tok/J al variare del riuso di cache. Questa fase
misura una cosa diversa sullo stesso nodo: **quanta parte del prezzo di una
GPU si paga mentre quella GPU non sta generando**, al variare della latenza
degli strumenti che l'agente chiama.

Le due fasi non si uniscono in un orchestratore solo. ADR-013 vuole **una**
traiettoria in volo per attribuire energia per-step; la matrice vuole N
sessioni concorrenti per muovere l'hit-rate. Sono forme incompatibili: due
fasi sequenziali sullo stesso nodo, non un esperimento fattorializzato.

### Il denominatore: span della traiettoria, non della finestra

Decisione portante, e la misura che l'ha imposta.

`f_nongen` = (Σ durate tool + Σ gap fra step consecutivi) / span.

Lo **span è quello della traiettoria** — `last.end_elapsed_ns −
first.start_elapsed_ns` — non `run_duration_ns`, che è lo span della finestra
di campionamento. Le due definizioni sono entrambe difendibili sulla carta e
differiscono di un ordine di grandezza sui dati reali.

Evidenza: `~/inferscope/validation-results/adr-013-a10-vllm/report-20260721T193436.txt`
(A10, Qwen2.5-7B, vLLM, tool 200 ms). Il campionamento inizia 3,878s prima
del primo step e la finestra è 150s contro una traiettoria di 6,218s — 24×.
Sulla stessa run: energia unattributed **91% della finestra**, tempo non
generante **9,78% della traiettoria**.

Se l'headline fosse "la frazione del costo pagata mentre la GPU non genera" e
il denominatore fosse la finestra, il numero pubblicato sarebbe dominato da un
artefatto di sovradimensionamento dello strumento. Con `EXP_WINDOW_MARGIN=1.2`
il gonfiaggio resterebbe del 20% strutturale anche a finestra ben calibrata.

`run_duration_ns` resta nell'analisi come **diagnostica di eccesso finestra**,
dichiarata accanto al risultato, mai come denominatore. Un test pinna
l'ancoraggio pubblicato (f_nongen 9,78%, packing bound 1,11).

### Ancoraggio alla fonte

La tabella di provenienza sopra dà `tempo LLM vs tool: LLM 71–98%, tool 2–29%`
(Fig. 7, GAIA al massimo). È l'intervallo in cui `f_nongen` deve cadere perché
la misura sia rappresentativa di traffico agentico reale: il 9,78% misurato il
21/07 ci sta dentro, verso il basso. Lo sweep delle latenze è scelto per
**coprire l'intervallo della fonte**, non per esplorare un range arbitrario.

### Le due politiche, e quella che la misura falsifica

Il braccio decisionale (`driver/analyze_cost_decision.py`) valuta due politiche
di piattaforma sullo stesso report. Entrambe **adimensionali**, quindi
indipendenti dal prezzo dichiarato: il $/M token viene separatamente da
`inferscope cost` sul nodo, e le due letture non si contaminano.

**P1 — rilascio per segmento.** Libera la GPU su ogni segmento tool più lungo
del prezzo di rientro. Saving = Σ(d − C) sui segmenti con d > C, dove C è il
cold start **misurato** in `vllm-coldstart-probe` (~18s, con il finding
27s/96s che ne dichiara la varianza). `--reentry-secs` è obbligatorio senza
default: il valore dev'essere visibile nel comando che ha prodotto i numeri.

Sull'ancoraggio A10 del 21/07, P1 vale **0,000s**. È zero per costruzione:
il segmento tool più lungo dello sweep è 5,0s contro ~18s di rientro. **La
politica ovvia è falsificata dalla misura, e questo è il risultato da
pubblicare** — non un esito negativo da nascondere. Il dominio del claim va
dichiarato con esso: il risparmio da occupancy interrotta è reale solo se la
GPU liberata serve altro, e su un noleggio a granularità oraria non risparmia
nulla.

**P2 — packing.** Il tempo non generante non si libera: si riempie. Bound di
sovrapposizione `1/(1 − f_nongen)`, cioè quante traiettorie una GPU può
ospitare prima che i segmenti generanti si contendano. Sull'ancoraggio: 1,11.
È un **upper bound sotto non-interferenza dichiarata** — il batching reale
cambia il throughput — e il limite viaggia accanto al numero, non dopo.

### Sweep, repliche, dispersione

La tool latency è un **parametro scelto da noi**: si pubblica la curva col
crossover, non un punto. Quattro valori (0,2 / 0,5 / 2,0 / 5,0 s/tool)
coprono due ordini di grandezza e l'intervallo della fonte.

**Tre repliche per cella a seed dichiarati**, non una run. Con `max_tokens`
fisso il seed non muove la struttura — muove il testo generato — quindi la
dispersione fra repliche misura il jitter dell'engine sullo span LLM, che è
il denominatore. Una run per cella pubblicherebbe un punto senza sapere se è
distinguibile dal vicino.

**Il CV della tool latency non è una seconda dimensione dello sweep.**
`f_nongen` somma 3-5 durate e la media domina: la varianza non muove la curva.
Quello che muove è l'**affidabilità del bound sotto concorrenza**. Quindi una
sola cella dello sweep (2,0 s/tool) ripetuta a CV 0,5 — tre celle in più, e la
differenza fra pubblicare un limite e pubblicare un limite con la sua
dispersione. Distribuzione lognormale parametrizzata per media aritmetica e
CV, che è l'unica forma in cui i due flag significano ciò che dicono; a CV 0
il comportamento è bit-identico alle celle già validate.

### Braccio di misura e braccio di ancoraggio

Stessa conclusione della sezione "replay deterministico, non agente reale",
raggiunta di nuovo per una misura diversa.

Guidare lo sweep con il vero loop Deep Agents non funziona, e la misura lo
dice: stesso prompt, stesso modello, temperature 0.0, tre run hanno dato span
7,4s / 12,7s / 34,7s con 4-8 step. Il modello decide la forma della
traiettoria, quindi numeratore e denominatore si muovono entrambi per ragioni
scorrelate dal parametro che si sweeppa. Una curva costruita su quel braccio
non sarebbe leggibile.

`run_replay.py` fissa numero di chiamate LLM, numero di step tool e token
generati per chiamata, e lascia la tool latency come unica variabile. Tre
ripetizioni a 0,2 s/tool: span 11,1 / 10,9 / 11,0s — **dispersione 1,8%**
contro il fattore 4,7 del braccio agentico.

`run_trajectory.py` non è sostituito: **ancora** il replay. Girato alla stessa
latenza sullo stesso nodo, mostra traiettorie reali che cadono nella regione
che il replay descrive (31,5% contro 36,3% a 2,0 s/tool nella prova su
llama.cpp, scarto spiegato da n_tool 2 contro 3). Due bracci, ciascuno prova
ciò che può.

Entrambi scrivono lo stesso formato attraverso lo stesso `StepsFileCallback` e
passano gli stessi gate in `trajectory_gates.py`: un gate che differisse fra i
bracci renderebbe inutili le celle di ancoraggio come evidenza.

### Parametri irreversibili per cella

Due, e nessuno correggibile a posteriori:

- **La finestra di campionamento.** Dimensionata dallo span di una traiettoria
  di calibrazione misurata sul nodo, per cella (span + tool wall della cella)
  × margine. Non una finestra per lo sweep: a 5,0 s/tool la traiettoria è ~14s
  più lunga che a 0,2s.
- **Lo steps-file.** Su `--sample-only` la traiettoria si deriva una volta
  sola, in volo, e non è ri-joinabile dopo.

Da cui: la **directory di cella è l'unità archiviabile** (steps-file, meta,
report, argv, costo, decisione). Senza lo steps-file il report non è
ri-analizzabile, perché la traiettoria dentro è già joinata e un difetto di
join non è più diagnosticabile.

Da cui anche: il **prezzo si deriva cella per cella sul nodo**, mai a fine
campagna. È l'unico momento in cui un'astensione di `cost` è ancora
diagnosticabile.

### Risultati misurati — sessione A10 del 2026-08-04

Lambda us-east-1, 1× A10 24GB PCIe a $1,29/h, Lambda Stack 24.04, vLLM
0.23.0, Qwen2.5-7B-Instruct, `--enforce-eager`, inferscope 0.5.0.
Evidenza in `exp-results/20260804-a10-cost/`: sedici directory di cella,
otto file ciascuna, provenienza (`nvidia-smi`, `pip freeze`, versione
binario, descrizione istanza) archiviata accanto.

**Fase 1 — ADR-011 su vLLM reale.** Prima lettura KV-cache su vLLM vero in
questo repo; finora il claim esisteva solo su simulatore.

| regime | hits_delta | queries_delta | realized | util. media |
|--------|-----------:|--------------:|---------:|------------:|
| H0     |          0 |       251.320 |    0,000 |         95% |
| H2     |    203.456 |       235.209 |    0,865 |         52% |

H0 dà zero esatto per costruzione — prefissi disgiunti, nessun riuso
possibile — e H2 realizza 0,865. I due regimi si separano di 0,865 in
valore assoluto.

Limite dichiarato: la cella H0 è durata 115,7s contro una finestra di 90s
(129% riempito), quindi la sua energia è troncata. L'assert sull'hit-rate
non ne è toccato — hits e queries vengono dallo scrape Prometheus prima e
dopo la cella, non dalla finestra — ma **nessun tok/J va derivato da H0**.

Nota collaterale non cercata: H2 gira al 52% di utilizzazione GPU contro
il 95% di H0. Con la cache calda la GPU lavora meno per lo stesso volume
di token, il che è parte del motivo per cui il packing bound è
interessante.

### La curva

Quindici celle, quattro latenze × tre repliche a seed dichiarati, più tre
celle di dispersione a CV 0,5. Riempimento finestra fra 83% e 87% su tutte
(intervallo di guardia 60-90%). `gaps = 0,00%` dello span ovunque:
l'overhead di framework fra step è sotto la risoluzione.

| tool latency | f_nongen | packing bound | $/M token (span) |
|-------------:|---------:|--------------:|-----------------:|
| 0,2 s        |    2,30% |          1,02 |          $12,16  |
| 0,5 s        |    5,55% |          1,06 |          $12,62  |
| 2,0 s        |   19,01% |          1,23 |          $14,72  |
| 5,0 s        |   37,01% |          1,59 |          $18,91  |

Lo span LLM resta **costante a 25,5s** su tutte e quindici le celle: solo
il termine aggiunto si muove, che è la condizione perché la curva sia
leggibile e la ragione per cui il braccio di misura ha struttura fissa.

### Il costo di generazione non si muove: si paga l'attesa

Il dato che rende la curva difficile da contestare è la scomposizione del
prezzo per cella, a rate dichiarato $1,29/h:

| tool latency | costo generazione | costo tool | rapporto |
|-------------:|------------------:|-----------:|---------:|
| 0,2 s        |        $0,009127  | $0,000215  |      1× |
| 0,5 s        |        $0,009155  | $0,000538  |    2,5× |
| 2,0 s        |        $0,009158  | $0,002150  |     10× |
| 5,0 s        |        $0,009147  | $0,005375  |     25× |

Il costo di generazione è **costante a $0,00915 su tutte e quindici le
celle** — stessi 768 token, stessa GPU, stesso modello. Tutta la crescita
del prezzo, +55% da $12,16 a $18,91 per M token, è tempo in cui la GPU è
allocata e non genera.

### Un vincolo di lettura sul $/M token

Il `$/M gen tokens` che `inferscope cost` stampa è calcolato **sulla
finestra di campionamento**, non sullo span della traiettoria. La
differenza non è accademica: le tre celle a CV 0,5 condividono la stessa
finestra da 38s e stampano $17,7069 / $17,7068 / $17,7068, cioè
indistinguibili, mentre il loro `f_nongen` varia di 4,09 punti.

La finestra è un parametro strumentale scelto da noi. Quindi:

- le cifre della tabella sopra sono ricalcolate **sullo span**, ed è
  l'unica forma confrontabile fra celle e difendibile in review;
- il `$/M` su finestra resta valido come costo di un intervallo di
  noleggio realmente occupato, ma **non** è un prezzo di listino e non va
  presentato come tale;
- la ripartizione `generating` / `in tools` che `cost` stampa è invece
  indipendente dallo strumento, e infatti riproduce esattamente i
  `f_nongen` misurati (20,7% / 19,6% / 16,6% sulle tre celle CV).

**Headline corretta**: non "$14,72 per M token", ma *il 19,0% del costo
attribuito si paga mentre la GPU è ferma sui tool, e sale al 37,0% a 5
s/tool*.

### Le due politiche, alla prova

**P1 = 0,000s su tutte e quindici le celle.** Il segmento tool più lungo
dello sweep è 5,0s contro un prezzo di rientro misurato di ~18s: nessun
segmento ripaga il rilascio. La politica ovvia — liberare la GPU mentre
l'agente aspetta uno strumento — è falsificata dalla misura in tutto
l'intervallo di latenza coperto dalla fonte. Il punto di pareggio
cadrebbe oltre i 18 s/tool, fuori dall'intervallo pubblicato (Fig. 7,
tool 2-29% del tempo).

**P2, packing bound**: da 1,02 a 1,59 sull'intervallo. Resta un upper
bound sotto non-interferenza dichiarata.

### Il ramo dispersione, e cosa ha dimostrato

Il disegno aveva argomentato che il CV della tool latency non muove la
curva ma muove l'affidabilità del bound. La misura lo quantifica.

| celle | media f_nongen | escursione fra repliche |
|-------|---------------:|------------------------:|
| 2,0 s/tool, CV 0   |     19,01% |            **0,006 pt** |
| 2,0 s/tool, CV 0,5 |     18,95% |            **4,09 pt**  |

Media invariata entro 0,06 punti, escursione **tre ordini di grandezza**
più grande. Il packing bound a 2,0 s/tool non è 1,23: è **1,23 con
escursione 1,20-1,26 sotto CV 0,5**, e sul prezzo si traduce in $15,02 /
$14,81 / $14,30 per M token contro ±0,03% delle celle deterministiche.

Senza quelle tre celle avremmo pubblicato un limite; con esse pubblichiamo
un limite e la sua dispersione. Sono costate tre celle su quindici.

Vale anche come verifica del braccio di misura: a CV 0 le tre repliche
danno span identico al decimo (27,8 / 32,3 / 41,3 s per latenza), quindi
la dispersione osservata a CV 0,5 viene dal campionamento e non dal
motore.

### Costo della sessione

~25 minuti di nodo, ~$0,55 contro un cap dichiarato di $1,94. Il tempo di
nodo non è stato il vincolo in nessun momento: lo sweep completo occupa
~10 minuti di finestra. Il vincolo erano le celle, e nessuna è stata
sprecata.

Una sessione precedente, lo stesso giorno, era stata interrotta dopo ~$0,90
su quattro ostacoli ambientali — tokenizer del prefisso non in cache
(il generatore chiede quello del prefisso, non `--model`), `PATH` senza
`~/venv/bin` e quindi `ninja` non trovato, un engine acceso a mano che
occupava la porta dell'engine dell'orchestratore, un flag inesistente in
vLLM 0.23.0. Nessuno era un difetto di disegno; tutti e quattro sono ora
voci di `CHECKLIST-A10-COST.md` con il loro rimedio, ed è il motivo per cui
la seconda sessione è filata senza interruzioni.

### Cosa questa fase NON prova

- Non è traffico agentico vero (vale il limite già dichiarato sopra: è la
  firma della forma, ancorata dal braccio agentico).
- Il packing bound è un limite superiore sotto non-interferenza **dichiarata**:
  non è una misura di throughput sotto concorrenza reale.
- P1 = 0 vale nel dominio dichiarato (single-tenant, granularità oraria del
  noleggio). Non è un enunciato generale sull'inutilità del rilascio.
- I tok/J non sono confrontabili con la matrice H100 di luglio: GPU e modello
  diversi. Il confronto è interno alla sessione.

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
