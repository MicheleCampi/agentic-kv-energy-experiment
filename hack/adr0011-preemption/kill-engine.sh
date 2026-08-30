#!/usr/bin/env bash
# Lose a replica at a chosen instant of a trajectory, and bring it back.
#
# The notice is injected alongside so the operator sees the preemption and
# reacts as it would; the engine is killed because that is what makes the
# trajectories feel it. See DESIGN.md for why the full FleetService topology
# was priced and declined.
#
# Restart is backgrounded immediately after the kill: the point is to measure
# what a trajectory pays while its replica is gone and coming back, not to
# measure an outage of unbounded length.
set -uo pipefail
MARK="${1:?seconds after load start}"
NODE="${2:-}"
K="${K:-sudo k3s kubectl}"

sleep "$MARK"
T0=$(date +%s.%N)

if [ -n "$NODE" ]; then
  NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  $K patch nodestate "$NODE" --subresource=status --type=merge \
    -p "{\"status\":{\"spot\":{\"preemptionNoticeDetected\":true,\"preemptionNoticeTime\":\"$NOW\"}}}" \
    >/dev/null 2>&1
fi

pkill -f "bin/vllm serve" 2>/dev/null
echo "killed at t+${MARK}s"

# Bring it back. The weights are in the page cache after the first load, so
# this is warm-start territory: the cost the trajectory pays is the engine
# coming up plus its own context being reprocessed, which is the quantity
# under test.
PATH="$HOME/venv/bin:$PATH" nohup setsid "$HOME/venv/bin/vllm" serve \
  Qwen/Qwen2.5-7B-Instruct --port 8000 --host 0.0.0.0 --enforce-eager \
  > /tmp/vllm-restart.log 2>&1 < /dev/null &

IP=$(hostname -I | awk '{print $1}')
for i in $(seq 1 90); do
  c=$(curl -s -o /dev/null -w "%{http_code}" "http://$IP:8000/v1/models" 2>/dev/null)
  [ "$c" = "200" ] && { T1=$(date +%s.%N); printf "back after %.1fs\n" "$(echo "$T1 - $T0" | bc)"; exit 0; }
  sleep 1
done
echo "engine did not come back within 90s" >&2
exit 1
