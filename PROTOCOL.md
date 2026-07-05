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
