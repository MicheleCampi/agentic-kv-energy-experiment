# agentic-kv-energy-experiment

Measuring the energy signature (tokens/joule) of agentic ReAct workloads
across KV-cache hit-rate regimes — Qwen2.5-32B on H100 SXM5, vLLM.

## Headline result

| Regime | KV-cache reuse | tok/J (window-based, nominal) |
|---|---|---|
| H0 | none (cold prompts) | 5.692 ± 0.147 |
| H1 | low (`history_shared_frac=0.065`) | 6.923 ± 0.091 |
| H2 | high (shared-prefix heavy) | 9.628 ± 0.019 |

**+69.2% tokens/joule at H2 vs H0** — monotonic across regimes, std < 3%
everywhere, 18/18 measurement matrix (regime × condition × rep) accepted
with zero aborts. The gradient survives intact under the failure-injection
condition (5.449 / 6.846 / 9.610).

## Metric definition

tok/J = (prompt + generation token delta, engine-side, per ADR-011
Prometheus scrape) / (energy over a fixed 120 s window, NVML cumulative
energy counter delta).

**This is a conservative bound.** The fixed window includes an idle tail,
and the idle tail is *longest where tok/J is highest* (wall time ~35 s at
H2 vs ~65 s at H0). An exact active-window figure would therefore widen
the gradient, not shrink it: the real workload-attributable gradient is
**≥ +69%**. Per-sample power series persistence (needed for exact
active-window attribution) is tracked as instrumentation backlog (F16).

## Workload shape — read before interpreting

Generation is held constant by design (~1024 tokens per rep: 8 requests,
`--max-tokens 128`), so the workload is **prefill-dominated** and the
gradient is a **prefill / KV-reuse effect**. This mirrors agentic loops
with long shared context and short tool-call outputs; it does not claim
anything about decode-heavy regimes.

## Measurement

- Profiling: [inferscope](https://github.com/MicheleCampi/inferscope) —
  KV-cache hit-rate via vLLM Prometheus scrape (ADR-011), energy via NVML
  cumulative counter, per-phase attribution (ADR-012).
- Engine: vLLM, Qwen2.5-32B-Instruct, single H100 80GB SXM5 (Lambda).
- Full analysis, anomaly log (zero exclusions), cross-checks and
  reproduction steps: [`analysis/RESULTS.md`](analysis/RESULTS.md).
- Per-cell and aggregate computation: [`analysis/tokj_matrix.py`](analysis/tokj_matrix.py).

## Process discipline

Every GPU session is preceded by a mandatory node-off dress rehearsal
against a fake engine; abort criteria are fixed before node-on; evidence
is copied off-box before teardown. The hardening cycle that preceded the
measurement campaign (14 findings, incl. a critical silently-null energy
path) is documented in `hack/gpu-session/`.

## Write-up

Article forthcoming (in pipeline). <!-- go-live: add link -->
