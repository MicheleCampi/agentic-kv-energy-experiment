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
- **LATE** — injected at turn 3, by which point the trajectory carries two turns of accumulated context.
- **DRAIN-ONLY** — the node is marked draining but no replica is lost, to
  separate the cost of the replacement from the cost of the notice.

**How the replica is actually lost, and why not through the operator.**

The first attempt injected `preemptionNoticeDetected` on the NodeState and
expected the trajectories to feel it. They did not, and the gate caught it: the
operator orchestrates `VllmService` children, while the load here talks to an
engine started by hand on the node. Two tracks that never meet — the node was
marked preempted, nothing served by it moved, and no trajectory saw a failure.

Building the full topology would fix that: a FleetService owning real vLLM
replicas, traffic routed through the dispatcher to the placed child's Service.
It was priced during the session at roughly forty extra minutes and double the
budget — a vLLM image on three nodes, model weights cached on each, and several
new ways to fail with the meter running.

**It was not worth it, because it is not what this measures.** That the operator
places a replacement in 57s is already measured and published (ADR-0009). What
is new here is what a preemption costs a trajectory carrying context, and that
number is identical whether the replacement is chosen by an operator or restored
by hand. The full topology would add the mechanism, not the answer.

So the replica is lost by killing the vLLM process on the node, with the notice
injected alongside so the operator sees and reacts as it would. The replacement
is restored manually.

**The limit that creates, stated here rather than discovered later:** this
measures the cost to the agent, not the operator's replacement policy under
agentic load. Those are separable questions and this design answers the first.

**Timing the injection is the whole difficulty, and the session retuned it.**
Measured on the node: a trajectory spans 41.6s, giving 6.7s of generation per
turn between 5.0s tool calls. Turn 1 runs t+0 to t+6.7s, turn 3 runs t+23.3 to
t+30.0s, turn 4 starts at t+35.0s.

And the engine takes **28s to come back** after being killed, with weights warm
in the page cache. That number decides the marks: a kill at turn 4 would find
the trajectory finished before the engine returned, measuring nothing.

So **EARLY is t+2s (turn 1)** and **LATE is t+27s (turn 3)** — not turn 4 as
first drafted. The contrast survives the change: at turn 1 a trajectory resends
its opening context, at turn 3 it resends two turns of accumulated
conversation.

## Frozen inputs

| parameter | value | why |
|---|---|---|
| fleet | 3× A10, k3s, operator in-cluster | matches the ADR-0009 preemption topology |
| N trajectories | 4 | enough to see a distribution, small enough to fit one replica |
| injection marks | t+2s (EARLY), t+27s (LATE) | turn 1 and turn 3; turn 4 starts at t+35s and the engine needs 28s to return |
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
