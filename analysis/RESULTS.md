# Matrix results — session 2026-07-11 (H100 SXM5, Lambda)

Repo state: matrix accepted 18/18 (3 regimes × nominal/failure × 3 reps).
Evidence: `gpu-evidence/session-20260711/` (smoke, calib, matrix-20260711).
Analysis script: `analysis/tokj_matrix.py`.

## Headline

Energy efficiency (tokens/joule, full 120s window, nominal condition):

| Regime | realized hit-rate | tok/J (mean ± std, n=3) | vs H0 |
|--------|-------------------|--------------------------|-------|
| H0     | 0.0000 exact      | 5.692 ± 0.147            | —     |
| H1     | 0.449–0.515       | 6.923 ± 0.091            | +21.6% |
| H2     | 0.933             | 9.628 ± 0.019            | +69.2% |

Failure condition tracks nominal closely (H0 5.449 ± 0.103, H1 6.846 ± 0.046,
H2 9.610 ± 0.029): the KV-cache efficiency gradient survives the failure
regime essentially intact.

Monotone in hit-rate, per-regime std < 3% of mean everywhere. Gross energy
per fixed window: H0 ~42.5 kJ → H1 ~35 kJ → H2 ~26.5 kJ.

## Metric definition and conservative-bound argument

tok/J = (prompt_tokens_delta + generation_tokens_delta) / (window energy),
engine-side token counters (inferscope ADR-011 scrape), NVML energy counter
(`energy_source: counter` on all 18 cells, 2400 samples/cell @ 50ms, 120s).

The 120s window is fixed; generator wall time varies by regime (H2 ~35s,
H1 ~50s, H0 ~65s). The idle tail is therefore *longest where tok/J is
highest* (H2), so window-based tok/J understates the high-hit-rate regimes
more than the low ones: **the true active-workload gradient is ≥ +69%**.
Window-based numbers are the published metric (fully evidence-backed, no
idle-power estimation); exact active-window attribution requires per-sample
power series — backlog F16.

Note on ADR-012 phase attribution: `energy_prefill_by_time_mj +
energy_decode_by_time_mj` equals window energy by construction (it
apportions the window, it does not exclude idle), so it does not provide an
independent active-window figure for this design.

## Workload shape

generation_tokens_delta ≈ 1024 constant across all cells: 8 requests with
`--max-tokens 128` (generator default, sampled output targets mean 180 are
capped), so nearly all requests saturate the cap (8 × 128 = 1024; the one
884 cell had early EOS). The workload is prefill-dominated by design. The efficiency
gradient is a prefill/KV-reuse effect; generation-only tok/J (~0.02–0.04)
is reported by the script but is not meaningful here.

## Anomalies (recorded, none excluded)

- `H1_nominal_rep1`: realized 0.4489 (band [0.40, 0.60], low side) with
  growth-stop 7/8. tok/J 6.818 vs sibling reps 6.968/6.983 — within 2.2% of
  the cell family; excluding it moves the H1 nominal mean by ~0.05 (0.7%).
  Kept, no exclusion.
- Growth-stop 6–7/8 on 4 further cells with max session tokens ≤ effective
  target: physiological (pre-append stop working as designed, session
  2026-07-10 hardening).

## Cross-checks

- Calibration green: rep realized 0.5152/0.5239/0.5236, spread 0.0087;
  cross-session confirmation vs 2026-07-10 corrected rep1 (0.5153).
- KV invariant confirmed third time on third node: 38,272 tokens / 9.35 GiB.
- H0 realized = 0.0 exact on all 6 cells (no accidental prefix sharing).
- H2 stability ~0.6% across 6 cells (0.9312–0.9334).

## Reproduction

    python3 analysis/tokj_matrix.py \
      --matrix-dir gpu-evidence/session-20260711/matrix-20260711

Session cost ~$5.70, ~1h20, zero aborts.
