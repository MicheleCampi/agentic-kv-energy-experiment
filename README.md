# agentic-kv-energy-experiment

Two measured axes on one agentic ReAct workload, on rented GPUs, with the
design written down before each node was switched on:

- **energy against cache reuse** — tokens/joule across KV-cache hit-rate
  regimes (Qwen2.5-32B, H100 SXM5, 18 cells);
- **cost against tool latency** — how much of a GPU's price is paid while it
  is allocated and not generating (Qwen2.5-7B, A10, 15 cells).

Both measured with [inferscope](https://github.com/MicheleCampi/inferscope) on
vLLM 0.23.0, evidence and per-cell provenance committed.

**What the two axes are for.** Anyone running agents on their own GPUs has to
answer one question before buying hardware: *how many concurrent trajectories
does a replica actually hold, and what does each one cost while it waits on a
tool?* The energy axis gives the per-trajectory cost of the cache regime; the
cost axis shows that the increase with tool latency is allocation rather than
work — the GPU is paid for, and idle. A third result, a packing bound derived
from this data and then
[tested on hardware](https://github.com/MicheleCampi/vllm-coldstart-operator),
says how many trajectories fit before the arithmetic breaks. That is capacity
planning for agent infrastructure, measured rather than modelled.

That bound was measured under lockstep, and three more campaigns went after what
that meant. They are worth reading in order, because **the last one overturns
the first two**.

[Staggering the starts](hack/adr0010-interleaving) of two trajectories by half a
tool call took idle time from 38.5% to 18.0%.
[Raising concurrency](hack/adr0010-concurrency) to N=4 and N=8 made the gap
*wider*, not narrower — 37.5 points, saturating — because synchronised
trajectories stay idle 37.7% of the time however many there are.

Then [the arrival campaign](hack/adr0010-arrival) removed the assumption all
three shared: that trajectories are identical apart from start time. Give them
different lengths and generation sizes, the way real agents differ, and
**lockstep collapses on its own** — synchronised idle falls from 37.7% to 11.1%
with nothing scheduling anything. Let them arrive at random and it reaches
**0.3%**. Spacing them deliberately, the policy an admission controller would
implement, makes it **worse**: −2.1 points against unmanaged arrival, because
even spacing pushes trajectories back toward a common phase that random gaps
keep them out of.

So the honest conclusion is a **negative build recommendation**: do not write an
admission controller to space agent trajectories. The idle it would recover
mostly is not there once the trajectories are realistic. The earlier numbers
were correctly measured and were an artefact of clones — which the designs said
in advance was their weakest assumption.

One methodological finding travels with all of them: the **mean** running count
*falls* while the GPU gets busier, because staggering trades time at two
concurrent requests for time at one. A capacity calculation built on the average
reads the better arrangement as packing *worse*. The primary metric was fixed as
the distribution before any node was booked, which is the only reason these runs
produced results instead of shrugs.

## First axis: energy against cache reuse

| Regime | KV-cache reuse | tok/J (window-based, nominal) |
|---|---|---|
| H0 | none (realized hit-rate 0.0000 exact) | 5.692 ± 0.147 |
| H1 | partial (realized hit-rate 0.449-0.548) | 6.923 ± 0.091 |
| H2 | high (realized hit-rate 0.931-0.934) | 9.628 ± 0.019 |

**+69.2% tokens/joule at H2 vs H0** — monotonic across regimes, std < 3%
everywhere, 18/18 measurement matrix (regime × condition × rep) accepted
with zero aborts. The gradient survives intact under the failure-injection
condition (5.449 / 6.846 / 9.610).

## Metric definition

tok/J = (prompt + generation token delta, engine-side, per ADR-011
Prometheus scrape) / (energy over a fixed 120 s window, NVML cumulative
energy counter delta).

**This is a conservative bound.** The fixed window includes an idle tail,
and the idle tail is *longest where tok/J is highest* (measured active time
36.2 s at H2 vs 62.8 s at H0, see below). An exact active-window figure
would therefore widen
the gradient, not shrink it: the real workload-attributable gradient is
**≥ +69%**. Per-sample power series persistence (needed for exact
active-window attribution) is tracked as instrumentation backlog (F16).

## Workload shape — read before interpreting

Generation is held constant by design (~1024 tokens per rep: 8 requests,
`--max-tokens 128`), so the workload is **prefill-dominated** and the
gradient is a **prefill / KV-reuse effect**. This mirrors agentic loops
with long shared context and short tool-call outputs; it does not claim
anything about decode-heavy regimes.

## Two further results from the same evidence

**The knob acts on prefill only.** Engine-side phase counters put prefill at
30.66 s (H0) -> 17.09 s (H1) -> 2.85 s (H2), a 91% collapse, while decode
holds at 27.9 s in H1 and H2 (std 0.09 s and 0.01 s over six cells each).
H0 decode averages 27.13 s with a wider spread, explained by one cell that
hit early EOS at 884 generated tokens instead of 1024. Nothing else in the
energy gradient is moving.

**Token-share energy apportionment is blind to cache reuse.** ADR-012
apportions window energy to prefill/decode on two bases and publishes their
divergence. Across the matrix `share_prefill_tok` is flat at 0.996 (total
spread 0.001 over all 18 cells) while `share_prefill_time` falls 0.52 ->
0.09; the divergence separates into three non-overlapping bands
(-0.465 +/- 0.023, -0.616 +/- 0.021, -0.903 +/- 0.0009). A prompt-token
counter has no term that responds to reuse, so it keeps assigning ~99.6% of
device energy to prefill while actual prefill work drops by an order of
magnitude. This is one axis of the calibration ADR-012 calls for, not its
closure: prefill-only and decode-heavy isolation remain unmeasured.

**Idle tail, measured.** `phase_timeline` samples at 1 Hz carry cumulative
counters; the sample where they stop advancing is end-of-work. Active time
is 62.8 s / 50.2 s / 36.2 s (idle 47.6% / 58.2% / 69.9% of the fixed
window), agreeing with generator wall-clock to within 0.4-0.9 s on all 18
cells. Per-sample energy was not persisted, so the exact active-window
figure remains out of reach (F16) and the published metric stays
window-based.

## Second axis: what an agentic trajectory costs while it waits

The matrix above measures energy against cache reuse. A second campaign on
the same workload measures **cost against tool latency** — how much of a
GPU's price is paid while that GPU is allocated and not generating. Fifteen
cells on 1×A10 (Qwen2.5-7B, vLLM 0.23.0), 2026-08-04.

| tool latency | non-generating fraction | packing bound | $/M gen tokens |
|-------------:|------------------------:|--------------:|---------------:|
| 0.2 s        |                   2.30% |          1.02 |         $12.16 |
| 0.5 s        |                   5.55% |          1.06 |         $12.62 |
| 2.0 s        |                  19.01% |          1.23 |         $14.72 |
| 5.0 s        |                  37.01% |          1.59 |         $18.91 |

**The cost of generating does not move: $0.00911-0.00916 per trajectory
across all fifteen cells** — same 768 tokens, same GPU, same model, a 0.5%
band. All of the +56% in $/M token is waiting. Prices are derived at a declared $1.29/h and computed
over the trajectory span, not over the sampling window; the window is an
instrument parameter, and the distinction is documented in `PROTOCOL.md`
rather than smoothed over.

**The obvious policy is falsified.** Releasing the GPU on tool segments
longer than the measured re-entry price saves **0.000s on every one of the
fifteen cells**: the longest segment in the interval the source documents
(Fig. 7, tools 2-29% of the time) is 5.0s against a ~18s cold start measured
in [vllm-coldstart-probe](https://github.com/MicheleCampi/vllm-coldstart-probe).
Break-even sits outside the published range. The time is not freed — it is
filled, which is what the packing bound quantifies.

**The bound travels with its dispersion.** Three cells were spent repeating
one latency at CV 0.5 on the tool sleep. The mean is unchanged (18.95% vs
19.01%) while the spread across replicas goes from 0.006 to 4.09 points, so
the bound at 2.0 s/tool is 1.23 ranging 1.20-1.26 under realistic variance
rather than a bare 1.23. It remains an upper bound under declared
non-interference.

**The same session closed ADR-011 against real vLLM.** The hit-rate scrape
that the H100 matrix above could not use — it did not read vLLM's
prefix-cache series until 2026-08-02 — was validated on the node before the
cost sweep: 203,456 hits over 235,209 queries at H2 against exactly zero hits
over 251,320 queries at H0, the two regimes separating by 0.865 in absolute
terms. Until then the claim held only against the simulator. One declared
limit: the H0 cell overran its window (115.7s against 90s), so its energy is
truncated and no tok/J should be derived from it — the hit-rate figures are
unaffected, coming from the Prometheus delta rather than the window.

Design decisions were written down **before** the node was switched on and
the results sit beside them unedited, including the ones the measurement
falsified: [`PROTOCOL.md`](PROTOCOL.md), tertiary axis. Sixteen cell
directories of evidence — steps-file, its meta, report, argv, cost,
decision — under `validation-results-a10-cost/`.

## Measurement

- Profiling: [inferscope](https://github.com/MicheleCampi/inferscope) —
  KV-cache hit-rate via vLLM Prometheus scrape (ADR-011), energy via NVML
  cumulative counter, per-phase attribution (ADR-012).
- Engine: vLLM 0.23.0, Qwen2.5-32B-Instruct, single H100 80GB SXM5
  (Lambda), 9.35 GiB KV cache. `--enforce-eager` on every run as a matrix
  invariant: CUDA graphs are a second lever with their own trade-offs, and
  the knob under study here is the hit-rate. Absolute tok/J figures are
  therefore eager-mode figures; the gradient is what transfers.
- Full analysis, anomaly log (zero exclusions), cross-checks and
  reproduction steps: [`analysis/RESULTS.md`](analysis/RESULTS.md).
- Per-cell and aggregate computation: [`analysis/tokj_matrix.py`](analysis/tokj_matrix.py).
- `driver/` holds the second axis: the ADR-013 per-trajectory attribution
  driver and the phase-2 cost cell orchestrator. It does not feed the
  hit-rate matrix above — the two axes are separate campaigns on the same
  workload, run sequentially on the same node because ADR-013 needs one
  trajectory in flight while the matrix needs N concurrent sessions.

Two limits of the evidence itself, stated rather than discovered later:
per-cell manifests carry no timestamp, so execution order is recoverable
from the engine log but not from the structured evidence; and requests are
sent with `temperature: 0.0` while the model's own `generation_config.json`
supplies `repetition_penalty: 1.05`, which the server applies. The latter
affects which tokens are generated, not how many, so it does not touch a
prefill-driven gradient - but "greedy decoding" would be the wrong phrase.

## Process discipline

Every GPU session in this campaign was preceded by a mandatory node-off
dress rehearsal against a fake engine; abort criteria were fixed before
node-on; evidence was copied off-box before teardown. The hardening cycle
that preceded the measurement campaign (14 findings, incl. a critical
silently-null energy path) is documented in `hack/gpu-session/`.

## Write-up

[KV-cache reuse is an energy lever. Per-token attribution can't see
it.](https://michelecampi.github.io/observability/systems-engineering/llm-inference/2026/07/30/agentic-kv-energy.html)
(2026-07-30) — the write-up, including the two further results above and
the limits of the evidence.
