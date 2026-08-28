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

**1. Zero-cost rehearsal — PASSED 2026-08-27.** `stub.py` parameterised on N,
driven by `probe2.py` at 250ms:

    N   SYNC idle   STAG idle    gap
    2      63.6%       34.3%   +29.2 pp
    4      62.8%        8.3%   +54.5 pp
    8      62.8%        7.3%   +55.5 pp

The analysis separates the arms at every N, which is what this gate is for.

**It also flags that the hypothesis is at risk.** On the stub the gap *widens*
rather than narrowing, because SYNC idle stays flat at ~63% however many threads
run: they start together, have identical durations, and therefore never drift
out of phase. Adding threads adds nothing.

That is a property of the stub, not a prediction. Its trajectories are
deterministically identical; vLLM's share a real batch with generation times
that vary, so lockstep may break down on its own at higher N in a way the stub
cannot show. The rehearsal proves the instrument works — it does not forecast
the result, and the criterion above stays exactly as written.

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

Run 2026-08-28, 1×A10 on Lambda, driver 580.105.08, vLLM 0.23.0,
Qwen2.5-7B-Instruct, `--enforce-eager`, 18 arms, order counterbalanced.
Evidence: `runs/20260828/`. Zero failed scrapes across all eighteen.

**Verdict against the criterion fixed before the run: the hypothesis is wrong.**

| N | SYNC idle | STAGGERED idle | gap |
|---|---|---|---|
| 2 | 38.0% (sd 0.8) | 17.7% (sd 0.4) | **+20.3 pp** |
| 4 | 37.7% (sd 0.3) | 0.2% (sd 0.3) | **+37.5 pp** |
| 8 | 37.7% (sd 0.2) | 0.2% (sd 0.3) | **+37.5 pp** |

The criterion said below 5pp at N=8 confirms the hypothesis and above 15pp
refutes it. The measured gap is **37.5pp**, nowhere near either boundary.

**The gap does not decay — it saturates.** It grows from 20.3pp to 37.5pp and
then stops moving between N=4 and N=8. Staggering takes idle to essentially zero
from N=4 onward.

**And the reason is one column.** SYNC idle is *flat at 37.7% at every
concurrency*, with a standard deviation under a point. Adding trajectories to a
synchronised fleet recovers no idle time at all — they wait together however many
there are. The prediction assumed that more trajectories in flight would make
"at least one generating" approach certainty on its own. It does not, because
under lockstep they are not independent draws: they are the same draw, repeated.

**Guardrail: passed cleanly.** Tool wall is 15.00s by construction in every arm,
and generating time runs 25.96s at N=2 to 26.72s at N=8 — **+2.9% across a
fourfold increase in concurrency**. `num_requests_waiting` stayed at 0 and the
cache holds 100,272 tokens (6267 blocks × 16), so the engine served eight
trajectories almost as cheaply as two. Nothing here is queueing.

**What it means for sizing a fleet.** Arrival order is not a small-fleet detail
that washes out at scale. On this workload it is the dominant factor and stays
dominant: a replica serving synchronised trajectories is idle 38% of the time
whether it holds two or eight, and staggering their starts recovers all of it.

## What this still does not settle

**Trajectories identical apart from start time.** This is the strongest possible
case for staggering and the weakest for SYNC — real agents differ in length, tool
count and generation size, and decorrelate on their own. Whether SYNC idle stays
flat when the trajectories are heterogeneous is the obvious next question, and
this design cannot answer it.

**Three points, one offset, one node.** N ∈ {2,4,8} shows saturation between 4
and 8; it does not locate the knee, and says nothing about N=32 or about whether
2.5s remains the right stagger at higher concurrency.

Cost: about $1.60.
