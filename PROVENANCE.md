# PROVENANCE

Canonical record of the execution environment of the H100 sessions
(2026-07-09 to 2026-07-11) and of the **provenance of the workload
parameters**. The sessions' evidence is in `gpu-evidence/`; `results/` holds
no runs. Later campaigns record their environment next to their own evidence
(`validation-results-a10-cost/provenance-*.txt`,
`hack/*/runs/*/PROVENANCE.txt`). Of `sim-results/`, the directories other
documents cite are published (`calibration/`, `nsessions-check-n8/`,
`nsessions-check-n20/`); the rest is local working material.

## Identity of the measurement (inferscope)

inferscope **does not measure per-phase energy.** NVML counters are
whole-device; prefill and decode co-locate on the same SM array. inferscope
exposes a **dual-basis apportionment** of the measured total energy (time-share
and token-share) and the **divergence** between the two bases as a first-class
derived signal — honest about the co-location, not a claim of physical
attribution.

inferscope is **NOT modified** for this experiment. The two signals it needs
already exist:

- KV-cache hit-rate via Prometheus scrape (ADR-011).
- Per-phase energy / dual-basis divergence (ADR-012).

The experiment varies the hit-rate regime and observes how both move.

## The thesis (and its source)

Source: Yuan et al., *Agentic AI Workload Characteristics*, arXiv:2605.26297
(25 May 2026).

The paper establishes — measuring time and tokens, **not energy** — that
ReAct-style agentic workloads are *decode-dominated conditional on hit-rate*:
with effective prefix caching the input is largely reused (empirical hit-rates
84.6–99.5%, decode 91.0–98.6% of LLM time), but "if this state is evicted, a
decode-dominated workload can become expensive recomputation" (§5).

This experiment makes that sentence **quantitative and energetic**: it measures
how tokens/joule and the dual-basis divergence move across the hit-rate regime
(cold → warm), and characterises the transition under stress (context inflated
by failure, non-cached append). The energy axis is the space the paper
explicitly leaves empty.

## Provenance of the workload parameters

The traffic generator is **synthetic, derived from published distributions** —
NOT from real traces (the paper releases none). Every parameter is anchored to
a figure or table in the source. The exact values are fixed in PROTOCOL.md;
what is declared here is the provenance mapping:
- turns per task → Fig. 3 (min/max/mean±std per benchmark and mode)
- accumulated context → Fig. 4 (mean/max in tokens)
- output composition (thinking/message/tool-call) → Fig. 5
- LLM-vs-tool time split → Fig. 7
- Input/Output and Append/Output ratios per turn → §5 (per-turn table)
- failure signature (context up to 1.8× the mean) → Fig. 6

Where the paper gives an interval but not the shape of the internal
distribution, the assumed shape is declared as an **explicit assumption** in
PROTOCOL.md (e.g. sampling within min/max around mean±std), not passed off as
measured.

## Divergence from the source's setup (declared, not hidden)

This experiment does NOT replicate the paper's setup. Deliberate differences:

| dimension | paper (2605.26297)         | this experiment              |
|-----------|----------------------------|------------------------------|
| model     | Qwen3.6-27B / Gemma4-31B   | Qwen2.5 (consistent w/ series)|
| vLLM      | v0.20.0, TP=2              | 0.23.0 cu13, single-GPU      |
| hardware  | 2× H100 NVL, 12 NVLink     | 1× H100 SXM5 (Lambda)        |
| measures  | time, tokens (OTel/Jaeger) | **per-phase energy** (NVML)  |

Rationale: the goal is not to reproduce the paper's task characterisation but
to measure an orthogonal axis (energy) the paper does not cover, on a stack
consistent with the sibling experiments (cuda-graphs, chunk-size). The shape of
the workload is taken from the paper; the hardware and the engine are our own.

## Hardware

- GPU: NVIDIA H100 SXM5, single device. The sessions did not capture it.
  Lambda's billing record names the instance type `gpu_1x_h100_sxm5` for the
  sessions of 9, 10 and 11 July 2026; each session's
  `provenance-instance.txt` transcribes it.
  `hack/gpu-session/00-env.sh:32-34` planned a `node-pins.txt` with the GPU
  name and the driver; no such file was committed.
- Host: Lambda Cloud on-demand instance

## Software stack

Captured, in the `pip-freeze.txt` of each session stage:

- vLLM 0.23.0, torch 2.11.0, and the CUDA 13 libraries (`cu13` wheels)

Declared, not captured:

- inferscope at `c0bebb9` (`hack/gpu-session/00-env.sh:12`), a release build
  copied from optim-dev with rsync (`:13`). Its identity was not checked on
  the node, and its version, 0.3.0, does not tell it apart: `c0bebb9` is
  `v0.3.0-27`, and the workspace version is 0.3.0 at both.
- The base OS, the NVIDIA driver and the Python version. The A10 sessions of
  August record driver 580.105.08 and CUDA 13.0
  (`hack/*/runs/*/PROVENANCE.txt`); the H100 sessions did not record theirs.

## Pre-GPU validation

The generator and the capture path are developed and validated against
`llm-d-inference-sim` (CPU-only, KV-cache enabled) on `optim-dev`, BEFORE any
GPU hour. Simulation outputs live in `sim-results/` and are kept distinct from
the GPU evidence in `gpu-evidence/`. "Validated in simulation" and "measured on
GPU" are never conflated.

## Per-run capture

Each run directory holds a `manifest.json` (regime, seed, sessions, token
counts, realised hit rate, capture window) next to the generator log and the
inferscope report. Each session stage adds an `engine.json` (engine command,
PIDs) and a `pip-freeze.txt`. There is no nvidia-smi, `vllm --version`,
`inferscope --version` or git HEAD snapshot, which this section used to
promise.
