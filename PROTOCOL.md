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

## Fuori scope

Replica della caratterizzazione task del paper (accuratezza, success rate).
Questo esperimento misura SOLO la firma energetica della forma di carico.

## Nota di validazione — confine di calibrazione H1 (2026-07-05)

Con scrape per-pod (somma prefill+decode) e run isolate via nonce, il sim
realizza: H0=0.000 (esatto, prefisso disgiunto — niente knob engine per H0),
H2=0.915, monotonia confermata. MA il pavimento con prefisso condiviso è
~0.85 ANCHE a storia interamente unica (history_shared_frac=0.0). ROOT CAUSE
(2026-07-05, verificata quantitativamente dal sorgente v0.8.2): artefatto di
scala tra il budget del generatore e il tokenizer del sim. Catena: (1) il
generatore dimensiona la storia in chars/4 (calibrato su BPE reale); (2) il
sim usa SimpleTokenizer (pkg/tokenizer/tokenizer.go: regex word-level con
`\w+` che ingloba gli underscore + hash FNV) -> l'unita' di filler
`UNIQ_s3_b7_obs_xxxx` = 1 token-sim, la storia si comprime ~6x nello
spazio-token del sim mentre il prefisso (testo naturale) no; (3) la
proporzione vista dal sim diventa ~81% prefisso / 19% storia -> pavimento
teorico 0.812; (4) residuo +2.5pt: i counter Prometheus del sim sono
alimentati da membership any-position dei blocchi (startRequest), NON dal
conteggio prefix-truncated (countCachedBlockPrefix, usato solo per lo score
interno) — divergenza semantica dal vLLM reale, secondaria con hash
incatenati. PREDIZIONE VERIFICATA: quota-prefisso replicando il regex del sim
sul prompt della floor probe = 0.812 vs 0.837 misurato. CONSEGUENZA: il
pavimento e' compensabile (filler a unita' granulari -> token-sim separati),
quindi H1 torna calibrabile sul sim; in ogni caso il realizzato su vLLM reale
resta la coordinata pubblicata. Ipotesi intermedia "filler ripetitivo /
contabilita' content-addressed" falsificata da esperimento di controllo prima
della root cause. Nota storica sotto: CONSEGUENZA: il sim valida il MECCANISMO (monotonia,
estremi, pipeline) ma NON la posizione di H1, che si calibra sul vLLM reale
con un check economico a inizio sessione GPU (2-3 run corte prima della
matrice). history_shared_frac resta al valore di design 0.5.

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
