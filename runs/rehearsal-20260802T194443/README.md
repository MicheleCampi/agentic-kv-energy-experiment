# Dress rehearsal, cost-A10 campaign harness — 2026-08-02

Zero-cost rehearsal of `run_experiment.py` in the exact form the A10 session
will run it: `--engine vllm` in the cell command, contract check extended to
`--engine` and the `cost` subcommand, oversized-window warning active.
Launched by `hack/gpu-session/40-rehearsal.sh` against
`hack/gpu-session/fake_engine.py` through the versioned shim in
`hack/gpu-session/rehearsal-bin/vllm`.

## Primary assert: PASS

Both cells produced non-zero KV deltas through the ADR-011 scrape path:

| cell | hits_delta | queries_delta | turns_generated |
|------|-----------:|--------------:|----------------:|
| H0_nominal_rep1 | 17631 | 35264 | 3 |
| H2_nominal_rep1 | 17824 | 35650 | 6 |

Manifests carry `served_model` (Qwen2.5-7B-Instruct) and `count_tokenizer`
(Qwen2.5-0.5B-Instruct) as distinct fields — the two are deliberately
different here and must not be conflated when reading token counts.

## What this rehearsal does NOT certify

1. **Energy.** optim-dev has no NVML, so `gpu` carries no device data and no
   energy figure is produced. On the paid node, `"gpu": null` in a cell
   manifest is an abort criterion, not a warning.
2. **Regime separation.** `fake_engine.py` returns a 50% hit rate by
   construction, so `hitrate_realized` is ~0.4999 in every cell regardless
   of regime. H2 > H0 separation is unproven until real vLLM.
3. **Window sizing.** `--sample-secs 60` here is an arbitrary rehearsal
   value, not a dimensioned window; the oversized-window warning fires as
   expected (generator filled 2% of the window). On the node the window is
   sized from calibration wall-times per regime, never chosen.

## Warm-up cell: reference, not defect

`warmup/H1_nominal_rep1` shows `turns_generated: 0` and
`target_context_tokens: 1000`, below the 14785-token prefix. This is
deliberate (`run_experiment.py`, the `target below the prefix size ->
history_budget = 0` branch): the warm-up sends the bare prefix to warm the
prefix cache and defuse the first-cell cold-start artefact found 2026-07-11,
and its "H1" label is a directory name, not a realized regime. It is
excluded from all statistics and is not resume-safe by design (F12).

Consequence, recorded because it was checked and found not to hold: the
warm-up cannot be used to size `--sample-secs`. It generates no turns and
runs at a context an order of magnitude below any cell.

## Defects found and fixed during this rehearsal

Four in the harness and its fixtures, one environment residue, all
committed before the run: `7e4007a` (fake engine labelled its KV series
with a hardcoded name while `/v1/models` returned the served model, giving
a silently empty KV timeline), `de83c76` (the July shim lived outside any
repo and silently ignored unrecognised arguments), `6cff105` (no rehearsal
launch script existed — July's was hand-composed), `2135a7c` (contract
check read stdout alone; inferscope 0.5.0 prints `--help` on stderr with
exit 2, so every gate failed on a correct binary), plus a stray
`llama-server` holding :8000 from an earlier session.
