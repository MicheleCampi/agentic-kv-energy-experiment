# agentic-kv-energy-experiment

Measuring the energy signature (tokens/joule) of agentic ReAct workloads
across KV-cache hit-rate regimes — Qwen2.5-32B on H100 SXM5, vLLM.

## Headline result

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
- `driver/` is a separate experiment sharing this repo: the ADR-013
  per-trajectory attribution driver, run on 1xA10 on 2026-07-21. It does not
  feed the matrix results above.

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

Article forthcoming (in pipeline). <!-- go-live: add link -->
