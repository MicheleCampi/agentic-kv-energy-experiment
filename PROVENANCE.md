# PROVENANCE

Record canonico dell'ambiente di esecuzione e della **provenienza dei parametri
di workload**. I JSON in `results/` e `sim-results/` non portano tag
hardware/stack: ogni output è prodotto sotto l'ambiente qui descritto. In caso
di conflitto con uno snapshot per-run, prevale lo snapshot per-run.

## Identità della misura (inferscope)

inferscope **non misura energia per-fase.** I contatori NVML sono whole-device;
prefill e decode co-locano sullo stesso array di SM. inferscope espone una
**apportionment dual-basis** dell'energia totale misurata (time-share e
token-share) e la **divergenza** tra le due basi come segnale derivato di prima
classe — onesto sulla co-locazione, non una pretesa di attribuzione fisica.

inferscope **NON è modificato** per questo esperimento. I due segnali necessari
esistono già:
- KV-cache hit-rate via scrape Prometheus (ADR-011).
- Energia per-fase / divergenza dual-basis (ADR-012).
L'esperimento varia il regime di hit-rate e osserva come si muovono entrambi.

## Tesi (e sua fonte)

Fonte: Yuan et al., *Agentic AI Workload Characteristics*, arXiv:2605.26297
(25 May 2026).

Il paper stabilisce — misurando tempo e token, **non energia** — che i workload
agentici ReAct-style sono *decode-dominated condizionatamente all'hit-rate*:
con prefix caching efficace l'input è largamente riusato (hit-rate empirici
84.6–99.5%, decode 91.0–98.6% del tempo LLM), ma "se questo stato viene evicted,
un workload decode-dominated può diventare costosa ricomputazione" (§5).

Questo esperimento rende **quantitativa e energetica** quella frase: misura come
variano tokens/joule e la divergenza dual-basis attraverso il regime di hit-rate
(freddo → caldo), e caratterizza la transizione nel caso di stress (contesto
gonfio da fallimento, append non-cached). L'asse energia è lo spazio che il
paper lascia esplicitamente vuoto.

## Provenienza dei parametri di workload

Il generatore di traffico è **sintetico, derivato da distribuzioni pubblicate**
— NON da trace reali (il paper non rilascia trace). Ogni parametro è ancorato a
una figura/tabella della fonte. I valori esatti sono fissati in PROTOCOL.md; qui
si dichiara la mappatura di provenienza:

- turni per task → Fig. 3 (min/max/mean±std per benchmark e modalità)
- contesto accumulato → Fig. 4 (mean/max in token)
- composizione output (thinking/message/tool-call) → Fig. 5
- ripartizione tempo LLM-vs-tool → Fig. 7
- rapporti Input/Output, Append/Output per turno → §5 (tabella per-turn)
- firma del fallimento (contesto fino a 1.8× medio) → Fig. 6

Dove il paper dà un intervallo ma non la forma della distribuzione interna, la
forma assunta è dichiarata come **assunzione esplicita** in PROTOCOL.md (es.
campionamento entro min/max attorno a mean±std), non spacciata per misurata.

## Divergenza dal setup della fonte (dichiarata, non nascosta)

Questo esperimento NON replica il setup del paper. Differenze deliberate:

| dimensione   | paper (2605.26297)        | questo esperimento          |
|--------------|---------------------------|-----------------------------|
| modello      | Qwen3.6-27B / Gemma4-31B  | Qwen2.5 (coerente con serie)|
| vLLM         | v0.20.0, TP=2             | 0.23.0 cu13, single-GPU     |
| hardware     | 2× H100 NVL, 12 NVLink    | 1× H100 PCIe (Lambda)       |
| misura       | tempo, token (OTel/Jaeger)| **energia per-fase** (NVML) |

Razionale: l'obiettivo non è riprodurre la caratterizzazione task del paper, ma
misurare un asse ortogonale (energia) che il paper non copre, su uno stack
coerente con gli esperimenti gemelli (cuda-graphs, chunk-size). La forma del
workload è presa dal paper; l'hardware e l'engine sono i propri.

## Hardware

- GPU: NVIDIA H100 PCIe (singolo dispositivo) — *UUID catturato per-run*
- Host: istanza on-demand Lambda Cloud

## Stack software

- Base OS: lambda-stack-24-04 (Ubuntu 24.04)
- NVIDIA driver: 580.105.08
- CUDA: 13.0
- vLLM: 0.23.0 (build cu13)
- Python: *catturato per-run*
- inferscope: *commit HEAD catturato per-run*

## Validazione pre-GPU

Il generatore e il path di cattura sono sviluppati e validati su
`llm-d-inference-sim` (CPU-only, KV-cache abilitato) su `optim-dev`, PRIMA di
qualsiasi ora-GPU. Gli output di simulazione vivono in `sim-results/` e sono
distinti dalla cattura energia reale in `results/`. "Validato in simulazione" e
"misurato su GPU" non si confondono.

## Cattura per-run

L'orchestratore scrive uno snapshot di provenance machine-readable accanto a
ogni run (nvidia-smi, vllm --version, inferscope --version, git HEAD).
