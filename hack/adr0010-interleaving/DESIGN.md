# Staggered starts — does interleaving change what a replica holds?

Design written 2026-08-20, before any node is booked. Frozen parameters are
marked as such; results go in a separate section after the run.

## Why this exists

ADR-0010 measured a packing bound on real hardware and its postscript recorded
something the design had not anticipated. From the raw series of one N=2
replica:

    running = 0  →  65 samples  (39.2%)
    running = 2  → 101 samples  (60.8%)
    running = 1  →   0 samples

Never one. The two trajectories generate together and wait together for the
whole run: same script, same seeds, launched milliseconds apart, so they march
in **lockstep** and *share* their idle time instead of interleaving it. The
observed mean of 1.2393 in this rep is 2 × 0.608, not a smoothed overlap — and the
1.2434 reported by ADR-0010 is the mean across its three N=2 reps.

The prediction held, but the mechanism was not the one assumed — and the
postscript says so, leaving the interleaved case explicitly unmeasured.

## The question, in the terms that matter

A production fleet does not launch its agents in lockstep. Trajectories arrive
whenever users arrive. So:

**If the same replica serves trajectories that start at different times, does it
hold more of them than the lockstep measurement suggests?**

That is capacity planning: the packing bound decides how many replicas a fleet
needs for N concurrent agents, and if the bound is pessimistic under realistic
arrival, fleets are being over-provisioned.

## Hypothesis (falsifiable)

With a start offset large enough to decorrelate the trajectories, the mean
running count over the measurement window is **higher** than under synchronised
starts, because one trajectory generates while another waits on a tool.

The null is that it does not change: the trajectories keep their own rhythm and
the idle stays idle regardless of when they started. That is a **negative
result about interleaving**, and it makes the existing bound *stronger* rather
than weaker — a number that holds whatever the arrival pattern is more useful
in production than one that holds only when everything is synchronised.

Both outcomes are published.

## Experimental design

One A10, one vLLM, N=2 trajectories — the same shape ADR-0010 used, so the
lockstep arm is directly comparable to the measurement already on record.

Two arms, four reps each, in ABBA+BAAB order:

- **SYNC** — both trajectories start together. Reproduces ADR-0010's condition.
- **STAGGERED** — the second starts after a fixed offset.

Per rep: launch, sample `vllm:num_requests_running` at 250ms, analyse over the
declared window, record the full series.

## The offset, and why this value (frozen input)

The offset has to be long enough to decorrelate and short enough that both
trajectories still overlap for most of the run.

From the ADR-0010 data: at 5.0 s/tool a trajectory spans ~40s, of which ~37%
is tool wall. A single tool call is 5.0s. An offset of **2.5s — half a tool
call** — puts one trajectory's generation against the other's wait if their
rhythms are otherwise identical, which is exactly the condition that would
produce interleaving.

Frozen: **2.5s**, chosen from the published span arithmetic before the run, not
tuned afterwards.

| parameter | value | why |
|---|---|---|
| offset | 2.5s | half a tool call at the 5.0 s/tool cell |
| tool latency | 5.0 s/tool | the cell ADR-0010 used; the only one where D4 separates |
| N | 2 | comparable to the existing measurement |
| model | Qwen2.5-7B-Instruct | continuity with ADR-0010 |
| engine | vLLM 0.23.0, `--enforce-eager` | matrix invariant |
| reps | 4 per arm, ABBA+BAAB | same ordering discipline |
| sampling | 250ms | same as ADR-0010, so series are comparable |

## Decision criterion (fixed now)

**Primary metric: the fraction of samples where `running == 1`.**

This is the direct measurement of interleaving, and it is currently **exactly
zero** — 0 of 166 samples in the rep examined. It is a better primary than the
mean running count because it cannot be moved by anything except one trajectory
generating while the other does not.

**Secondary: mean running count over the window**, which is what the packing
bound is expressed in and what a capacity calculation would use.

Verdict:

- `running == 1` fraction rises clearly above zero **and** the mean increases →
  **interleaving happens**; the lockstep bound is conservative under staggered
  arrival, and by how much is the useful number.
- fraction stays at or near zero → **negative result**: starts being staggered
  does not make the trajectories share a replica any better. The existing bound
  holds regardless of arrival pattern, which is the stronger claim.
- mean moves but the fraction does not → something else changed; report it as
  unexplained rather than as interleaving.

**Guardrail:** total span per trajectory. If the staggered arm's trajectories
take materially longer, the two arms are not measuring the same workload and
the comparison is void.

## Gates

**1. Zero-cost rehearsal.** Run the harness against a CPU-only stub that exposes
`vllm:num_requests_running`, with and without offset, and confirm the analysis
distinguishes the two. A rehearsal that cannot tell the arms apart on a stub
will not tell them apart on hardware either.

**2. Pre-flight on the node, with an abort criterion.** Before the first
measured rep, one SYNC rep must reproduce the ADR-0010 shape — `running == 1`
at or near zero, mean around 1.24. If it does not, the harness or the engine
config differs from the recorded run and the comparison to it is void. Abort
rather than continue against an unknown baseline.

## Cost envelope

One A10 at $1.29/h. ADR-0010's session was ~$0.50 for six arms in twenty
minutes; this is eight arms of the same length, so budget **under $1** with the
engine already warm between reps.

## What this does not settle

**One offset is not a curve.** 2.5s answers whether interleaving can happen at
a plausible stagger. It says nothing about the shape of the relationship, or
whether a different offset would interleave better. That is a sweep, and a
sweep is only worth its cost if this single point shows an effect.

**Two trajectories are not a fleet.** N=2 is where ADR-0010 measured, so the
comparison is clean, but a real replica serves more. Whether the effect grows,
saturates or reverses with N is not measured here.

**And the workload is a replay, not live traffic.** Trajectories have identical
structure and differ only by start time. Real agents differ in length, tool
count and generation size, all of which decorrelate them further — so if
interleaving does not appear here, under the most favourable synthetic
condition, that is meaningful; if it does appear, real traffic may show more,
not less.

## Measured results (outputs)

To be filled after the run, beside the design rather than in place of it.
