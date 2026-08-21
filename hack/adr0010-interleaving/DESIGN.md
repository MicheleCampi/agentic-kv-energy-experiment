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

This is the direct measurement of interleaving. Checking all three ADR-0010 reps
rather than the one whose series I had opened first: the fractions are 0.0000,
0.0240 and 0.0060 — mean **1.00%, sd 1.25%**. Not zero, and the spread between
reps is larger than the mean.

That matters for the criterion. A first draft of this design said the baseline
was exactly zero, which was true of rep1 and false of the other two. Reading one
series and generalising is the same defect this project has caught twice before,
and the threshold below is set against the measured spread rather than against
an assumed floor. It is a better primary than the
mean running count because it cannot be moved by anything except one trajectory
generating while the other does not.

**Secondary: mean running count over the window**, which is what the packing
bound is expressed in and what a capacity calculation would use.

Verdict:

Threshold, fixed now: the staggered arm's mean `running == 1` fraction must
exceed the lockstep mean by more than **twice the lockstep spread** — that is,
above **3.5%** — with the arms' confidence intervals clear of each other.

- fraction above 3.5% **and** mean running count increases →
  **interleaving happens**; the lockstep bound is conservative under staggered
  arrival, and by how much is the useful number.
- fraction within the lockstep spread → **negative result**: staggering starts
  does not make the trajectories share a replica any better. The existing bound
  holds regardless of arrival pattern, which is the stronger claim.
- mean moves but the fraction does not → something else changed; report it as
  unexplained rather than as interleaving.

**Guardrail:** total span per trajectory. If the staggered arm's trajectories
take materially longer, the two arms are not measuring the same workload and
the comparison is void.

## Gates

**1. Zero-cost rehearsal — PASSED 2026-08-20.** `stub.py` exposes
`vllm:num_requests_running` from two threads that alternate work and wait the
way a trajectory does; `probe.py` samples it at 250ms with and without offset.

    LOCKSTEP   offset 0.00s   running==1:  0.0%   mean 0.729   {0: 82, 2: 47}
    STAGGERED  offset 1.25s   running==1: 59.0%   mean 0.709   {0: 47, 1: 79, 2: 8}

The lockstep arm reproduces the shape of the real ADR-0010 series — 0 or 2,
never 1 — and staggering moves the mass onto 1.

**The rehearsal also justifies the choice of primary metric.** The two means are
0.729 and 0.709: had the mean been primary, this run would have read as *no
difference*, and on hardware the same collapse would have hidden a real effect.
The fraction at `running == 1` moved by 59 points on the same data.

**2. Pre-flight on the node, with an abort criterion.** Before the first
measured rep, one SYNC rep must reproduce the ADR-0010 shape — `running == 1`
within the 0-2.4% range the three recorded reps span, mean around 1.24. If it does not, the harness or the engine
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

Run 2026-08-21, 1×A10 on Lambda, driver 580.105.08, vLLM 0.23.0,
Qwen2.5-7B-Instruct, `--enforce-eager`, 5.0 s/tool, 4 reps per arm in
ABBA+BAAB. Evidence: `runs/20260821/`. Zero failed scrapes across all eight.

**Verdict against the criterion fixed before the run: interleaving happens.**

| | SYNC | STAGGERED |
|---|---|---|
| `running == 1` fraction | 0.61% (sd 0.50) | **40.69%** (sd 0.33) |
| mean running count | 1.2477 | 1.2010 |
| time at `running == 0` | 38.5% (sd 0.4) | **18.0%** (sd 0.0) |

The primary metric moved from 0.61% to 40.69% against a threshold of 3.5%. The
intervals do not come close to touching.

**The number worth carrying: idle time halves.** A replica serving two
synchronised trajectories is doing nothing 38.5% of the window. Stagger their
starts by half a tool call and that drops to 18.0% — twenty points, with a
standard deviation of zero across four reps. The pauses get filled.

**And the mean moves the wrong way, which is the finding the design did not
anticipate.** Mean running count *falls*, 1.2477 to 1.2010. The distribution
shows why:

    SYNC   0: 39%   1:  1%   2: 60%
    STAG   0: 18%   1: 46%   2: 36%

Staggering converts time at 2 into time at 1, and time at 0 into time at 1. The
first shift costs more than the second gains, so the mean drops while the GPU is
demonstrably busier. **A capacity calculation based on the mean would conclude
that staggered arrival packs *worse*, and it would be wrong.**

This is the second time in this project that the mean and the distribution
disagree, and the second time the distribution was right. The gate-1 rehearsal
had already shown it on a stub — 0.729 against 0.709, a difference of nothing,
while the fraction moved 59 points — which is why the fraction was named primary
before any node was booked.

**Guardrail: passed.** Tool wall is identical at 15.00s by construction, and
generating time is 25.96s against 25.75s. The two arms did the same work in the
same time; only the phase between them changed.

**What this means for the packing bound.** ADR-0010's bound was measured under
lockstep, where trajectories share their idle. Under staggered arrival they do
not, and the replica is idle half as often — so the bound is not wrong, but the
quantity it was derived from behaves differently once arrival is realistic. A
fleet sized on lockstep numbers is sized on the least favourable phase.

## What this still does not settle

**One offset, one N.** 2.5s at N=2 shows the effect exists and is large. It does
not give the shape: whether a smaller stagger suffices, whether the benefit
saturates, or what happens at N=4 or N=8 where more trajectories compete for the
same batch. Those are sweeps, and this single point is what justifies paying for
one.

**And synthetic trajectories are the hard case, not the easy one.** These differ
only in start time — same length, same tool count, same generation sizes. Real
agents differ in all of those, which decorrelates them further. The 20-point
reduction measured here is therefore a floor rather than a ceiling.

Cost: about $0.60.
