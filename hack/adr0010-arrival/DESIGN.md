# What does controlling arrival actually buy?

Design written 2026-08-28, before any node is booked. Frozen inputs are marked
as such; results go in a separate section after the run.

## Why this exists

Three campaigns measured the same replica under different conditions and each
declared the same limit. From the most recent one:

> These trajectories are identical apart from start time, which is the strongest
> case for staggering and the weakest for SYNC. Whether SYNC idle stays flat when
> the trajectories are heterogeneous is the next question and this design cannot
> answer it.

That limit is not cosmetic. **Clones cannot drift apart.** Eight identical
trajectories started together stay in phase forever, which is why SYNC idle sat
flat at 37.7% at every concurrency. Real agents run different numbers of steps
and generate different amounts, so they decorrelate on their own — for free,
with no scheduler doing anything.

So the 37.5pp gap is an **upper bound** on what controlling arrival could buy,
not an estimate of it.

## The question, in the terms that decide whether to build something

**How much of that gap survives when trajectories are realistic, and how much of
what survives needs a scheduler rather than just luck?**

Three regimes answer it, and the differences between them are what matter:

- **SYNC** — everything starts together. The pathological case, and the baseline
  the previous runs measured.
- **POISSON** — starts drawn from an exponential with a 2.5s mean. This is
  unmanaged arrival: nobody is scheduling anything, requests land when they land.
- **SPACED** — starts fixed 2.5s apart. This is a policy: an admission
  controller deliberately holding trajectories to space them.

**SYNC → POISSON is what you get for free.** **POISSON → SPACED is what a
scheduler would have to earn**, and it is the number that decides whether the
component is worth building at all.

## Hypothesis (falsifiable)

Most of the benefit is free. Heterogeneous trajectories under Poisson arrival
recover the majority of the idle that staggering recovered, and the additional
gain from deliberate spacing is **small — under 5 percentage points**.

If that holds, the honest conclusion is that an admission controller for this is
not worth building, and the useful finding is that *unmanaged arrival is already
close to optimal on this workload*.

The alternative is that spacing still buys double digits over Poisson, in which
case there is a component worth writing and a measured number to justify it.

**Both outcomes are publishable and one of them argues against my own operator
growing a feature.** That is the point of fixing the criterion first.

## Experimental design

One A10, one vLLM, N=8 — the concurrency where the previous run showed the gap
had already saturated, so any change here is attributable to the trajectories
and the arrival process rather than to scale.

**Six arms: {SYNC, POISSON, SPACED} × {homogeneous, heterogeneous}**, three reps
each, order counterbalanced.

The homogeneous row is not new data — it is the bridge to the existing record.
SYNC/homogeneous must reproduce 37.7% idle or the comparison to everything
already published is void.

## Frozen inputs

| parameter | value | why |
|---|---|---|
| N | 8 | where the gap saturated; isolates arrival from scale |
| mean gap | 2.5s | unchanged from the two previous runs |
| SYNC | offset 0 | pathological baseline, matches the record |
| POISSON | exponential, mean 2.5s | unmanaged arrival |
| SPACED | fixed 2.5s | the policy an admission controller would apply |
| heterogeneous shape | 3/4/5 calls, 128/192/256 tokens, cycling on index | mean unchanged at 4 and 192, verified |
| tool latency | 5.0 s/tool | matrix invariant |
| model | Qwen2.5-7B-Instruct, `--enforce-eager` | unchanged |
| reps | 3 per arm | sd was under 1pp in the last run at this N |
| sampling | 250ms | unchanged, series stay comparable |

**Why the mean workload is held constant.** Heterogeneity here means dispersion,
not more work: 3/4/5 calls average to 4, and 128/192/256 tokens average to 192.
If the mean moved, a lower idle in the heterogeneous arms could just mean the
GPU had more to do, and the comparison would say nothing about phase.

## Decision criterion (fixed now)

**Primary: idle time at `running == 0` in the heterogeneous row, and the two
differences within it.**

- **free = SYNC − POISSON**: what unmanaged arrival recovers on its own.
- **earned = POISSON − SPACED**: what a scheduler would add on top.

Verdict:

- **earned < 5pp** → the hypothesis holds. Unmanaged arrival is already close to
  optimal, and an admission controller that spaces trajectories is not worth
  building on this workload. Publish that, and do not build it.
- **earned > 10pp** → deliberate spacing buys something real. That justifies a
  component, and the number is the justification.
- **5–10pp** → report it and say it does not settle the build decision at this
  workload size.

**Secondary: the homogeneous row, as the control.** If SYNC/homogeneous does not
reproduce ~37.7%, the run aborts before the heterogeneous arms are interpreted.
And the SYNC homogeneous-vs-heterogeneous difference is itself informative: it
measures how much decorrelation comes from trajectory shape alone, with arrival
held pathological.

**Guardrail: mean generating time per trajectory across arms.** Heterogeneous
trajectories have different individual spans by construction, but the *mean*
must stay within a few percent of the homogeneous arms. If it drifts, the shapes
are not balanced and the rows are not comparable.

## Gates

**1. Zero-cost rehearsal — PASSED 2026-08-28.** `stub.py` extended to draw
exponential gaps and to vary thread durations; `probe3.py` runs all six cells at
N=8:

                    SYNC    POISSON   SPACED    free    earned
    homogeneous    63.6%     15.0%     7.3%   +48.6     +7.7
    heterogeneous  31.7%     15.2%    12.4%   +16.5     +2.8

The three regimes separate in both rows, which is what this gate is for.

**And the rehearsal already points where the hypothesis says.** Making the
threads heterogeneous halves SYNC idle on its own — 63.6% to 31.7% — with no
scheduler involved: different durations decorrelate them for free. `earned`
falls from 7.7pp to 2.8pp, below the 5pp threshold this design set for "not
worth building".

That is a stub, not vLLM. Threads sleeping on a timer are not requests sharing a
continuous batch, and the absolute numbers mean nothing. But it does establish
that the instrument can tell the three regimes apart, and that the split between
free and earned is measurable rather than lost in noise.

**2. Pre-flight, with an abort criterion.** One SYNC/homogeneous rep must
reproduce ~37.7% idle at N=8. If it does not, the harness differs from the run
this design builds on and nothing here can be compared to it.

**3. Shape balance, checked not assumed.** After the first heterogeneous arm,
compare mean generating time per trajectory against the homogeneous arm. If it
differs by more than a few percent the shapes are unbalanced, and a lower idle
would be extra work rather than better phase.

## Cost envelope

One A10 at $1.29/h. Eighteen arms at roughly 45s plus settling, about 25 minutes
of measurement; with setup, budget **under $2**. Abort if the pre-flight is not
clean within 20 minutes of the node coming up.

## What this will not settle

**One mean gap.** 2.5s is held across all three regimes so the arrival *process*
is the variable, not its rate. Whether a shorter or longer mean changes the
free-versus-earned split is a second dimension.

**One workload shape.** The dispersion here is 3/4/5 calls and 128/192/256
tokens — real agent traffic is likely wider and heavier-tailed. A wider spread
would decorrelate more, which pushes in the direction of the hypothesis, so this
is a conservative test of "spacing is not needed".

**And it does not measure an admission controller.** It measures the ceiling one
could reach. If `earned` turns out to be large, building the thing and measuring
it under the same harness is the next piece of work, not a conclusion of this
one.

## Measured results (outputs)

Run 2026-08-28, 1×A10 on Lambda, driver 580.105.08, vLLM 0.23.0,
Qwen2.5-7B-Instruct, `--enforce-eager`, N=8, mean gap 2.5s, 18 arms, order
counterbalanced. Evidence: `runs/20260828/`. Zero failed scrapes.

**Verdict against the criterion fixed before the run: do not build it.**

| | SYNC | POISSON | SPACED | free | earned |
|---|---|---|---|---|---|
| homogeneous | 37.7% (0.9) | 0.6% (0.0) | 0.4% (0.3) | +37.2 | **+0.2** |
| heterogeneous | 11.1% (0.2) | 0.3% (0.2) | 2.4% (0.0) | +10.8 | **−2.1** |

`earned` — what an admission controller would add over unmanaged arrival — is
**−2.1 percentage points** on realistic trajectories. Not merely below the 5pp
threshold: **negative**. Deliberately spacing starts is *worse* than letting them
land where they land.

**Three findings, in order of how much they change the picture.**

**1. Heterogeneity destroys lockstep on its own.** SYNC idle falls from 37.7% to
**11.1%** with nothing scheduling anything. Trajectories of different shapes
drift apart by themselves, which is exactly what the previous three campaigns
could not show, because their trajectories were clones.

**2. Unmanaged arrival then takes idle to almost nothing.** POISSON sits at
**0.3%**. There is no meaningful idle left for a scheduler to recover.

**3. Fixed spacing makes it worse, and the reason is legible.** SPACED
reintroduces a regularity that Poisson does not have: evenly spaced starts push
trajectories back toward a common phase, where exponential gaps keep them apart.
The policy an admission controller would implement is the one that partially
recreates the problem.

**What the previous result becomes.** The 37.5pp gap measured on 2026-08-28 was
real and correctly measured, and it is an artefact of clones. On this workload,
with trajectories that differ the way real ones do, **the idle that staggering
recovered mostly is not there to begin with**.

**Guardrail: passed.** Mean generating time runs 26.45s to 27.59s across all six
cells — under 4.3% spread — and within each row the three regimes agree to
0.35s. The dispersion of 11s in the heterogeneous rows is the heterogeneity
itself, by construction. The idle differences are phase, not workload.

## What this still does not settle

**One mean gap, one workload shape.** 2.5s across all regimes; dispersion of
3/4/5 calls and 128/192/256 tokens. Real traffic is likely wider and
heavier-tailed, which would decorrelate more — so this is a conservative test of
"spacing is not needed", and a wider spread would only strengthen it.

**And it says nothing about admission control for other reasons.** Cost,
fairness, priority and cache locality are all reasons to control admission that
this experiment does not touch. What it rules out is admitting *for phase*.

Cost: about $1.70.
