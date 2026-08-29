#!/usr/bin/env bash
# Inject a spot-preemption notice at a chosen instant of a trajectory.
#
# The operator learns about preemption from preemptionNoticeDetected on the
# NodeState status; ADR-0009 measured its 57s figure the same way and disclosed
# notice injection as the simulation boundary. This inherits that boundary.
#
# The only hard part is *when*. A trajectory at 5.0 s/tool spans ~41s, so turn 1
# lands around t+2s and turn 4 around t+32s. Sleeping to the mark from the same
# shell that started the load keeps the two clocks together — passing an
# absolute time would drift with ssh latency.
set -uo pipefail
NODE="${1:?node name}"
MARK="${2:?seconds after load start}"
K="${K:-sudo k3s kubectl}"

sleep "$MARK"
NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
$K patch nodestate "$NODE" --subresource=status --type=merge \
  -p "{\"status\":{\"spot\":{\"preemptionNoticeDetected\":true,\"preemptionNoticeTime\":\"$NOW\"}}}" \
  >/dev/null
echo "injected on $NODE at t+${MARK}s ($NOW)"
