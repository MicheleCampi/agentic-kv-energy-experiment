# PROTOCOL

Experimental protocol: the energy signature (tokens/joule, and dual-basis
divergence) of a ReAct-style agentic workload across the KV-cache hit-rate
regime.

## Operating thesis

The source paper (arXiv:2605.26297) establishes that the agentic workload is
decode-dominated *conditional on hit-rate*, but measures only time and tokens.
This protocol measures the energy axis: how tokens/joule and the
prefill/decode divergence move with hit-rate, and where the signature
collapses once the cache stops holding (failure case). The hypothesis — to be
checked against the data, NOT assumed — is that a knee exists in the
tokens/joule vs hit-rate curve, and that the dual-basis divergence captures
its position and sharpness.

## Invariant

- `enforce_eager=True` (CUDA graphs OFF) on every run. One cause on the curve:
  hit-rate is the knob, not the graphs. Consistent with the cuda-graphs and
  chunk-size experiments.

## Primary axis: KV-cache hit-rate regime

Hit-rate is not a flag: it is induced by the shape of the traffic. Method:

- **warm (high hit)**: N requests share one long common prefix (system prompt
  + tool definitions + accumulated history); only the per-turn append is new
  input. Realises high reuse (target hit-rate ~90%+).
- **cold (low hit)**: disjoint prefixes across requests and/or prefix cache
  disabled; every turn recomputes. Realises low reuse.
- **sweep**: intermediate levels, by varying the shared-prefix fraction.

### Context composition: cacheable prefix + accumulated history

The steady-state context (target 32-48K tokens) is made of TWO components
with distinct roles in the hit-rate — modelled separately because that is how
the real agentic workload is built (the source documents that nearly all of
the input is reused and only the per-turn append is new: Input/Output ≫
Append/Output, §5):

- **shared cacheable prefix** (~15K tokens, fixed): system prompt + tool
  definitions. A deterministic versioned artefact
  (`prefixes/agentic_system_v1`, 14785 tokens measured with the Qwen2.5
  tokenizer, seed 42, 40 tools). It is the stable base, identical across
  requests → always a cache hit in the warm regime.
- **accumulated history** (variable, carries the context up to target):
  previous turns, messages, tool calls, observations. This is the part that
  GROWS, and whose sharing is modulated to realise the hit-rate.

How the levels are realised on this structure:

| level    | prefix        | history                | effective hit-rate |
|----------|---------------|------------------------|--------------------|
| H2 warm  | shared        | largely shared         | ~90%+              |
| H1 mid   | shared        | partially disjoint     | ~50%               |
| H0 cold  | disjoint/off  | disjoint or recomputed | ~0%                |

The **failure** condition operates on this structure: appending a UNIQUE and
inflated error observation (1.8× the context, Fig. 6) grows the non-cached
share of the history and erodes the effective hit-rate even with a shared
prefix — the measurable break of the decode-dominated regime.

Target hit-rate levels (to be calibrated and confirmed on the simulator, then
on the node):

| level | description                   | target hit-rate  |
|-------|-------------------------------|------------------|
| H0    | cold / disabled cache         | ~0% (CALIBRATE)  |
| H1    | partially shared prefix       | ~50% (CALIBRATE) |
| H2    | largely shared prefix         | ~90%+ (CALIBRATE)|

3 levels (both ends plus the centre) to locate the knee of the tokens/joule vs
hit-rate curve at minimum node spend. Adaptive sampling: if calibration on the
simulator shows the knee falling between two levels, ONE fourth level is added
there — not an a-priori 4-point grid.

The REALISED hit-rate is measured by inferscope (ADR-011), not assumed. The
targets above are calibration objectives for the generator, not imposed
values.

## Secondary axis: failure stress

A condition replicating the signature of agentic failure (source Fig. 6: failed
tasks accumulate up to 1.8× the mean context). Mechanism: context inflated by
repeated error observations (continuous non-cached append), which erodes the
effective hit-rate even with a shared prefix. The point at which the
decode-dominated regime breaks and prefill starts biting again.

| condition | description                                              |
|-----------|----------------------------------------------------------|
| nominal   | normal trajectory, contained per-turn append             |
| failure   | error loop: inflated non-cached append (~1.8× ctx)       |

## Generator parameters — per-parameter provenance

Source = arXiv:2605.26297. Reference model: Qwen (this experiment uses Qwen2.5;
the paper uses Qwen3.6-27B — divergence declared in PROVENANCE).

| parameter                   | value (Qwen, from the source)      | figure |
|-----------------------------|------------------------------------|--------|
| turns/task (thinking)       | mean ~12–41 per benchmark          | Fig. 3 |
| turns/task (instant, SWE)   | mean 62.4, concentrated distrib.   | Fig. 3 |
| accumulated context (SWE)   | mean 68.7K–80.1K, max 146K–166K    | Fig. 4 |
| context (Terminal/GAIA)     | mean 52.5K–65.1K (Qwen)            | Fig. 4 |
| output: thinking (Qwen T)   | 29.0–40.7% of output               | Fig. 5 |
| output: tool-call (Qwen I)  | 70.4–81.6% of output               | Fig. 5 |
| LLM vs tool time            | LLM 71–98%, tool 2–29% (GAIA max)  | Fig. 7 |
| Input/Output per turn       | ~120–560× (mean, workload-dep.)    | §5     |
| Append/Output per turn      | ~3.6–6.1× mean, ~0.7–1.4× median   | §5     |
| failure context inflation   | up to 1.8× the mean context        | Fig. 6 |

### Declared assumptions (where the source gives an interval, not a shape)

- The SHAPE of the distribution within min/max/mean±std is not given by the
  source. ASSUMPTION: sampling within [min,max] centred on mean with spread std
  (log-normal for the long tails of turn counts; to be pinned in the generator).
- The read/explore→execute/write temporal sequence (paper figure) is
  qualitative; ASSUMPTION: not modelled in generator v1 (the energy signal
  depends on cache and context, not on the semantic type of the tool).
  Revisable.
- Exact observation size per tool type: not quantified for Qwen; ASSUMPTION:
  derived from the Append/Output ratio, not per tool.

## Nature of the workload: deterministic replay, not a real agent

The generator does NOT animate a real LLM agent. It emits a **deterministic,
seeded replay** of a traffic shape parametrised on the source's distributions
(turns, context, composition, cache/append ratio). This is a deliberate choice,
not a convenient simplification:

- **Rationale**: a real agent is non-deterministic (same task → different
  trajectories, as the source itself notes). That non-determinism would destroy
  the reproducibility of the energy measurement. A system characterisation
  experiment needs a reproducible load shape, not a live one.
- **What is measured**: the energy signature of the LOAD SHAPE (cache/recompute
  ratio, context depth, phase split), anchored token by token to published
  measured distributions — not the semantic behaviour of the agent.
- **Declared limit**: a reviewer may object that "this is not real agentic
  traffic". The answer is above — it is the signature of the shape, not of the
  agent. The limit is explicit, not hidden. It is the same approach as standard
  serving benchmarks (`vllm bench serve` replays a distribution, it does not
  launch agents).

## Models

- **Qwen2.5-32B** (fixed). Rationale: realism of the agentic regime (the
  52-80K contexts in the paper require a credible size), same family as the
  paper (Qwen), consistency with the cuda-graphs and chunk-size siblings
  already running 32B.
- **Operating context target: ~32-48K tokens** (inside the paper's range, below
  the 146-166K maxima that would blow up run times). An explicit choice to stay
  within the node budget (3-5h), not a hidden limit. `max_model_len` sized
  accordingly, to be confirmed on the node.

## Repetitions

3 reps per cell (consistent with the series).

## Run count

`n_hitrate × n_failure × n_model × 3 reps`.

With 3 hit levels × 2 conditions × 1 model × 3 reps = 18 base runs (+3 if the
fourth adaptive level is added = 21). Node estimate: 3-5h on Qwen2.5-32B /
H100 PCIe. Settled once levels and model are fixed.

## Capture methodology

inferscope UNMODIFIED. Per run: realised hit-rate (ADR-011), per-phase energy
plus dual-basis divergence (ADR-012), over a window bracketing the generated
traffic. Carried over from the series: the window-truncation guard and
`active_fraction` (both live inside the inferscope report).

### Per-run manifest (third-party reproducibility)

For every run the generator emits a machine-readable manifest alongside the
inferscope report. Minimum contents:

- **target** hit-rate (level H*) vs **realised** hit-rate (measured, ADR-011)
- token count of the shared prefix (from the Qwen2.5 tokenizer)
- context depth reached (tokens)
- condition (nominal/failure), rep
- **RNG seed** (the run is bit-for-bit repeatable)
- sampling parameters used (turns, append, composition)

Together with the per-run provenance snapshot, the manifest makes the
experiment reproducible by a third party: it declares exactly what load was
generated, with which seed, and how far the realised value sits from the
target. Separating target from realised is first-class experimental honesty:
the generator is not assumed to have hit the regime — it is measured, and the
gap is recorded.

## Pre-GPU validation (zero cost)

Generator and capture path developed and validated against `llm-d-inference-sim`
(CPU-only, KV-cache enabled) on `optim-dev`. Validation objectives: (1) the
generator realises the target hit-rate regimes measured via ADR-011; (2) the
capture path produces complete records. Output in `sim-results/`. Only after
validation in simulation does the campaign move to the GPU node for energy
capture.

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

## Out of scope

Replicating the paper's task characterisation (accuracy, success rate). This
experiment measures ONLY the energy signature of the load shape.

## H1 calibration — full history (closed 2026-07-05)

Three root causes verified in sequence, each down to the source line, with a
quantitative prediction where one was possible.

**1. Tokenizer scale artefact** (floor ~0.84 with entirely unique history). The
generator sizes in chars/4 (real BPE); the sim uses SimpleTokenizer
(pkg/tokenizer/tokenizer.go, word-level regex: `\w+` swallows underscores) ->
the unit `UNIQ_s3_b7_obs_xxxx` = 1 sim-token, history compressed ~6x in
sim-token space while the prefix is not -> prefix share ~0.81 = the floor.
Prediction verified: 0.812 against 0.837 measured (+2.5pt: the Prometheus
counters are fed by any-position membership via startRequest, not by the
prefix-truncated countCachedBlockPrefix — a semantic divergence from real vLLM,
secondary). The intermediate hypothesis "content-addressed accounting over
repetitive filler" was falsified by a control experiment before the root cause
was found. FIX: granular filler — 1 hex word of 3 chars without underscores = 1
sim-token = ~1 BPE token, so sizing and counting coincide by construction.
Post-fix floor probe (frac=0.0, shared prefix): 0.448/0.476 — pure prefix share
~0.46, prediction ~0.45 confirmed.

**2. The interleaved knob is conceptually broken under chained hashing.** The
sim delegates to llm-d-kv-cache prefixHashes (pkg/kvcache/kvblock/
token_processor.go:130-134): `prefix = hash(prefix, chunk)` — each block's hash
incorporates the chain of its predecessors. A shared mid-history block preceded
by divergent unique blocks has a different hash in every session:
mid-history shareability DOES NOT EXIST, at any frac (the original design's
per-turn Bernoulli draw could not have worked). Real vLLM uses the same
cumulative hashing scheme, so the redesign applies identically to the GPU run.
FIX: shared-HEAD design — history_shared_frac = the fraction of the history
token budget covered by a shared head CONTIGUOUS with the prefix (first turns
identical cross-session by construction, then divergence; semantically closer
to agentic trajectories with a common bootstrap). Token-precise truncation of
the last shared block to the remaining budget: without it the head is quantised
to whole blocks (fixed mass 1449 tok, insensitive to frac over ~0.03-0.09,
measured). Post-fix quantisation residue: the ~4 template lines per block (~15
tok), not clamped.

**3. Additive seed derivation collides.** seed_base+rep+nonce collides for any
(nonce, rep) pair with the same sum: found through a realised value identical to
16 digits across campaigns (nonce 20260709 rep3 == 20260710 rep2 -> bit-identical
prompts, realised value polluted by a warm cache). FIX: prime-weighted
multi-axis derivation (nonce*1000003 + regime_idx*10007 + cond_idx*101 + rep),
no tuple collides. Evidence of the collision preserved in
runs/matrix-recal-confirm.

**Frozen values and confirmation matrix** (runs/matrix-recal-full, nonce
20260711, 18/18): H1 history_shared_frac=0.065. H0=0.000 exact 6/6 (disjoint
prefixes hold even with granular filler); H1 nominal 0.484-0.518 (mean 0.505),
failure 0.471-0.502 (mean 0.488); H2 nominal ~0.940, failure ~0.922. Monotonic
throughout; failure < nominal in H1/H2 (the bloated unique mass of failure
blocks dilutes the hits — the physically expected direction). The per-session
variance of the shared fraction (short sessions from the turns_min clamp weight
the fixed head by up to ~3x) is an accepted property of the design: the regime
is defined by the campaign mean, which is what the counters aggregate. The
published coordinate remains the value realised on real vLLM, calibrated with
2-3 short runs at the start of the GPU session before the matrix.

## Validation note (verified on the sim, 2026-06-29)

The established boundary of what `llm-d-inference-sim` v0.8.2 can validate:

- **Validates** the hit-rate axis: it exposes `vllm:prefix_cache_hits` and
  `vllm:prefix_cache_queries` (counters) → hit-rate via window delta (ADR-011).
  Concrete evidence: an existing fixture shows 12 completions with a long
  shared prefix producing a measured hit_rate=0.490. The mechanism of the
  primary axis has already been observed working on this sim.
- **Does NOT validate** the energy axis: the sim is CPU-only and exposes
  neither NVML energy nor prefill/decode time separation. tokens/joule and the
  dual-basis divergence are measured ONLY on the GPU node. The sim phase proves
  that the generator realises the target hit-rate regimes; it proves nothing
  about the energy signature.

## Validity boundaries of the sim confirmation

The 18/18 matrix in sim validates the GENERATOR (hit-rate calibration,
determinism, bisection trim), NOT the GPU execution topology. Outside sim
coverage, verified only with the node on or by static audit:

- a single engine persisting across cells (first-cell ordering effect: the
  prefix is cold once per start — mitigated with a warm-up cell)
- the inferscope branch (CLI contract, binary feature set, NVML)
- the network at runtime (HF Hub for weights and tokenizer)
- --sample-secs semantics (fixed timer, idle cost per cell)

Any claim of "validated in sim" implies ONLY the first category.
