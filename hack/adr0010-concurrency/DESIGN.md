# Does staggering still pay at higher concurrency?

Design written 2026-08-27, before any node is booked. Frozen inputs are marked
as such; results go in a separate section after the run.

## Why this exists

The staggered-start experiment (2026-08-21) measured two trajectories on one
replica and found the effect is large: offsetting their starts by half a tool
call takes the time the GPU spends idle from **38.5% to 18.0%**, standard
deviation zero across four reps per arm.

Its own limits section says what it does not settle:

> One offset at one N gives the existence and the size of the effect, not its
> shape. Whether the benefit saturates, or what happens at larger N where more
> trajectories compete for the same batch, is unmeasured.

That is the gap this closes.

## The question, in the terms a fleet owner asks

Nobody runs two agents. The question that decides how many replicas a fleet
needs is:

**As concurrency rises, does staggering still help — or do the trajectories
start filling each other's pauses on their own?**

Both answers are useful and they point opposite ways:

- **If the benefit shrinks with N**, staggering is a small-fleet optimisation.
  At scale, arrival is already jittered by real users and the effect is free.
- **If it holds**, then arrival order matters at any size, and a scheduler that
  admits agent trajectories should care when it starts them, not just where.

## Hypothesis (falsifiable)

The idle-time reduction from staggering **shrinks as N rises**, because with
more trajectories in flight the chance that at least one is generating
approaches certainty regardless of when each started.

Concretely: at N=2 the SYNC arm is idle 38.5% of the window. If trajectories
were independent, SYNC idle would fall roughly as the probability that all N
are simultaneously in a tool call — so the room staggering has to work with
shrinks with every added trajectory.

The null is that the gap between SYNC and STAGGERED stays flat: the lockstep
arm keeps its trajectories in phase however many there are, and the idle stays
shared rather than shrinking.

## Experimental design

One A10, one vLLM, same model and engine flags as the two experiments this
builds on — so the N=2 arms should reproduce what is already on record, which is
the check that the harness has not drifted.

**Six arms: N ∈ {2, 4, 8} × {SYNC, STAGGERED}**, three reps each, order
counterbalanced so neither arm systematically runs on a warmer engine.

The N=2 pair is not new information — it is the control. If it does not
reproduce 38.5% against 18.0%, the comparison to the larger N is void and the
run aborts.

## Frozen inputs

| parameter | value | why |
|---|---|---|
| N | 2, 4, 8 | 2 reproduces the record; 8 is where the A10 still has headroom |
| offset | 2.5s | half a tool call — unchanged from the 2026-08-21 run |
| tool latency | 5.0 s/tool | matrix invariant across all three campaigns |
| per trajectory | 4 LLM calls, 3 tools, 192 max tokens | unchanged |
| model | Qwen2.5-7B-Instruct, `--enforce-eager` | unchanged |
| reps | 3 per arm | 4 reps at N=2 gave sd 0.0-0.4pp; 3 is enough at this spread |
| sampling | 250ms | unchanged, so series stay comparable |

**Why 8 and not 16.** The KV cache on this node holds roughly 93k tokens
(20.6 GB allocated, ~15.2 GB weights, ~5.4 GB cache at ~0.06 MB/token). Eight
trajectories at ~3k tokens each occupy about **26%** of it. Sixteen would still
fit, but the point of stopping at 8 is that the measurement stays about *phase*
rather than about *queueing* — once requests wait for cache blocks, the idle
figure measures admission, not interleaving. That is a different experiment.

## Decision criterion (fixed now)

**Primary: the SYNC−STAGGERED gap in time at `running == 0`, as a function of N.**

At N=2 that gap is **20.5 percentage points** (38.5% → 18.0%).

- The gap **narrows monotonically** and is below **5pp** at N=8 → the hypothesis
  holds: staggering is a low-concurrency optimisation, and at scale natural
  arrival jitter does the same work for free.
- The gap **stays above 15pp** at N=8 → the null holds: lockstep keeps
  trajectories in phase however many there are, and start order matters at any
  fleet size.
- Anything between → report the curve and say it is not resolved by three
  points.

**Secondary: absolute idle in the SYNC arm as N rises.** If SYNC idle collapses
on its own — say below 10% at N=8 — then the gap narrowing is not evidence that
staggering stopped working; it is evidence there was nothing left to recover.
The two readings must be reported together or the primary is misleading.

**Guardrail: per-trajectory span.** Every arm must keep tool wall at 15.00s by
construction and generating time within a few percent of the 25.96s recorded at
N=2. If spans stretch at N=8, the engine is queueing and the arms are no longer
measuring the same workload.

## Gates

**1. Zero-cost rehearsal.** The stub from the previous experiment
(`../adr0010-interleaving/stub.py`) already separates the arms at N=2. Extend it
to N=4 and N=8 and confirm the analysis still distinguishes them — a harness
that collapses the arms on a stub will collapse them on hardware.

**2. Pre-flight on the node, with an abort criterion.** Run one N=2 SYNC rep
first. It must reproduce the record: idle at `running == 0` near 38.5%, `running
== 1` fraction near zero, span near 41s. If it does not, the harness or the
engine differs from the recorded run and every comparison in this design is
void. Abort rather than continue against an unknown baseline.

**3. Cache headroom, checked not assumed.** Read `vllm:gpu_cache_usage_perc` at
the end of an N=8 rep. If it is above 60%, the margin computed above is wrong
and the N=8 arm may be measuring queueing. Record the number either way.

## Cost envelope

One A10 at $1.29/h. Eighteen arms at roughly 45s each plus settling is about 25
minutes of measurement; with setup, budget **under $2**. Abort if the pre-flight
is not clean within 20 minutes of the node coming up.

## What this will not settle

**Three points are not a curve.** N ∈ {2,4,8} can show a direction and a rough
shape. It cannot distinguish a linear decay from an exponential one, and it says
nothing about N=32.

**One offset throughout.** The 2.5s stagger is held fixed so N is the only
variable. Whether the optimal offset changes with concurrency — plausibly it
should shrink, since more trajectories need less spacing to decorrelate — is a
second dimension this does not touch.

**And the trajectories remain identical apart from start time.** Real agents
differ in length, tool count and generation size, all of which decorrelate them
without any scheduler doing anything. So if the staggering benefit shrinks here,
in production it would shrink at least as fast — this is the *favourable* case
for staggering, not the realistic one.

## Measured results (outputs)

To be filled after the run, beside the design rather than in place of it.
