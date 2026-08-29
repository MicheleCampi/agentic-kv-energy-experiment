# What does losing a replica cost an agent mid-trajectory?

Design written 2026-08-29, before any node is booked. Frozen inputs are marked
as such; results go in a separate section after the run.

## Why this exists

Two results from this portfolio meet here for the first time.

**From the operator:** a fleet survives spot preemption in 57 seconds with a
2.3s worst-case service gap, make-before-break, measured on a 3-node A10 fleet.
That number describes the *fleet*.

**From the cold-start probe:** kernel I/O is only ~7% of an ~18s cold start —
the rest is GPU warmup, which is to say the KV cache being rebuilt.

Put together they imply something neither measured. **An agent mid-trajectory
is not a stateless request.** By its fourth turn it carries thousands of tokens
of accumulated context, and that context lives in the KV cache of the replica
serving it. When that replica goes away, the context goes with it: the client
resends the whole conversation and the new replica reprocesses it from cold.

So the fleet recovering in 57 seconds does not mean an agent recovers in 57
seconds, and nobody has measured the difference.

## The question

**How much does a preemption cost a trajectory, and does it depend on when it
lands?**

A trajectory preempted at its first turn resends ~46 tokens. The same
trajectory preempted at its fourth resends ~187 — four times the context, into
a replica whose cache is empty. If that scaling holds on hardware, the cost of
a preemption is not a constant a capacity model can add: it depends on where
the agents were.

## Hypothesis (falsifiable)

The recovery cost per trajectory **grows with how late the preemption lands**,
because the context to resend and reprocess grows with the turn count.

The null is that it does not: the replacement replica's cold start dominates
whatever the context size, so a preemption at turn 1 and at turn 4 cost the
same. That would be the more useful result for anyone sizing a fleet — a
constant is easier to plan for than a function — and it is the one I expect to
be wrong.

## Experimental design

Three A10 nodes, k3s, the operator in-cluster via the chart — the same topology
the ADR-0009 preemption run used, so its 57s figure is the reference this one
compares against.

**Four arms, three reps each:**

- **CONTROL** — trajectories run to completion, nothing preempted. Establishes
  the baseline cost of a turn.
- **EARLY** — preemption injected while trajectories are at turn 1.
- **LATE** — injected at turn 4, the last LLM call.
- **DRAIN-ONLY** — the node is marked draining but no replica is lost, to
  separate the cost of the replacement from the cost of the notice.

Preemption is injected by patching `preemptionNoticeDetected` on the NodeState
status at a chosen instant. The operator's state machine does the rest —
`Draining` then `Rescheduling` — which is behaviour already tested on hardware,
not something this experiment adds.

**Timing the injection is the whole difficulty.** A trajectory takes ~41s with
5.0 s/tool: turn 1 lands around t+2s and turn 4 around t+32s. The injector
sleeps to those marks. A miss makes the arm invalid rather than merely noisy,
which is why gate 2 verifies the mark before any measured rep.

## Frozen inputs

| parameter | value | why |
|---|---|---|
| fleet | 3× A10, k3s, operator in-cluster | matches the ADR-0009 preemption topology |
| N trajectories | 4 | enough to see a distribution, small enough to fit one replica |
| injection marks | t+2s (EARLY), t+32s (LATE) | turn 1 and turn 4 of a ~41s trajectory |
| retries | `--max-retries 10`, backoff 2.0s fixed | a trajectory that dies measures nothing |
| tool latency | 5.0 s/tool | matrix invariant across all campaigns |
| trajectory shape | 4 LLM calls, 3 tools, 192 tokens | unchanged |
| model | Qwen2.5-7B-Instruct, `--enforce-eager` | unchanged |
| reps | 3 per arm | sd was under 1pp at this fleet size in ADR-0009 |

## Decision criterion (fixed now)

**Primary: wall-clock cost of the preempted turn, EARLY against LATE.**

Defined as the duration of the LLM step that carried the retry, minus the
CONTROL median for a turn at the same position. That subtraction matters: turn 4
is slower than turn 1 even without preemption, because the prompt is longer.

- **LATE exceeds EARLY by more than 30%** → the hypothesis holds. Preemption
  cost scales with accumulated context, and a capacity model needs to know where
  the agents were, not just that a node was lost.
- **within 15% of each other** → the null holds. The replacement replica's cold
  start dominates and the cost is effectively a constant, which is the friendlier
  answer for anyone planning capacity.
- **15–30%** → report the numbers and say three reps at one fleet size do not
  separate the two.

**Secondary: prompt tokens resent per preempted trajectory.** This is the
mechanism, and it is arithmetic rather than measurement — but if the primary
shows no difference while the tokens differ fourfold, that locates the answer in
the engine's prefill rather than in the transfer.

**Guardrail: CONTROL must reproduce the trajectory span already on record**,
~41s with 15.00s tool wall. If it does not, the fleet is not serving the way the
earlier campaigns measured and nothing here compares to them.

## Gates

**1. Retry survives a dead endpoint — PASSED 2026-08-29, zero cost.**
`failstub.py` serves valid completions until a file is removed, then refuses
connections. Against it the driver shows prompt tokens growing 46, 93, 140, 187
across four turns — the accumulating context this campaign is about — and when
the endpoint dies mid-trajectory the affected turn records **2 retries and 4.8s
against 0.3s** for the others, with all four steps completing.

A first attempt proved nothing: the trajectory finished before the endpoint went
down. Lengthening the tool steps put the failure inside the run, which is the
same timing problem the injection marks face on hardware.

**2. The injection mark lands where intended.** Before any measured rep, run one
EARLY and one LATE and confirm from the steps file that the retry appears on
turn 1 and turn 4 respectively. If the mark misses, the arm measures a
preemption at an unknown point, which is worse than no data. Abort and re-time.

**3. CONTROL reproduces the record.** Trajectory span ~41s, tool wall 15.00s.
If it does not, this fleet is not serving the way the earlier campaigns
measured.

## Cost envelope

Three A10 at $1.29/h each. Setup is the long part — k3s, device plugin,
operator, engine on three nodes — call it 40 minutes; twelve arms at ~60s plus
settling is about 20 minutes. Budget **under $5**, and abort if the pre-flight
is not clean within 30 minutes of the nodes coming up.

## What this will not settle

**One trajectory shape, one fleet size.** Four trajectories on three nodes with
a fixed 4-call shape. Whether the cost scales with fleet size, or with longer
agents carrying more context, is not measured.

**Two injection marks, not a curve.** Turn 1 and turn 4 give a direction. They
do not locate where the cost starts growing, or whether it is linear in context.

**The preemption is injected, not real.** ADR-0009 measured its 57s figure the
same way and disclosed notice injection as the simulation boundary; this
inherits that boundary rather than removing it. A real spot reclaim arrives with
its own timing and its own node-level effects, and what is measured here is the
operator's reaction to a notice, not a cloud provider's behaviour.

**And the retry policy is mine, not the world's.** Fixed 2s backoff, resend the
whole conversation, ten attempts. A client that checkpointed its context
elsewhere, or one that resumed rather than resent, would pay a different price.
What this measures is the cost to the naive agent — which is what most agent
frameworks do today, and is the honest baseline.

## Measured results (outputs)

To be filled after the run, beside the design rather than in place of it.
