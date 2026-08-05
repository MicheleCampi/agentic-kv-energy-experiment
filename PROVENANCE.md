# PROVENANCE

Canonical record of the execution environment and of the **provenance of the
workload parameters**. The JSON files in `results/` and `sim-results/` carry no
hardware/stack tags: every output is produced under the environment described
here. Where a per-run snapshot conflicts with this document, the per-run
snapshot wins.

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
| hardware  | 2× H100 NVL, 12 NVLink     | 1× H100 PCIe (Lambda)        |
| measures  | time, tokens (OTel/Jaeger) | **per-phase energy** (NVML)  |

Rationale: the goal is not to reproduce the paper's task characterisation but
to measure an orthogonal axis (energy) the paper does not cover, on a stack
consistent with the sibling experiments (cuda-graphs, chunk-size). The shape of
the workload is taken from the paper; the hardware and the engine are our own.

## Hardware

- GPU: NVIDIA H100 PCIe (single device) — *UUID captured per run*
- Host: Lambda Cloud on-demand instance

## Software stack

- Base OS: lambda-stack-24-04 (Ubuntu 24.04)
- NVIDIA driver: 580.105.08
- CUDA: 13.0
- vLLM: 0.23.0 (cu13 build)
- Python: *captured per run*
- inferscope: *HEAD commit captured per run*

## Pre-GPU validation

The generator and the capture path are developed and validated against
`llm-d-inference-sim` (CPU-only, KV-cache enabled) on `optim-dev`, BEFORE any
GPU hour. Simulation outputs live in `sim-results/` and are kept distinct from
real energy capture in `results/`. "Validated in simulation" and "measured on
GPU" are never conflated.

## Per-run capture

The orchestrator writes a machine-readable provenance snapshot alongside every
run (nvidia-smi, vllm --version, inferscope --version, git HEAD).
