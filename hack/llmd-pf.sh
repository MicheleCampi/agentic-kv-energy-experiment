#!/usr/bin/env bash
# Re-attach idempotente dei port-forward per lo stack llm-d-sim (Blocco B).
# Uso: ./llmd-pf.sh           -> assicura forward attivi (EPP :8000, decode sim :18000)
#      ./llmd-pf.sh status    -> mostra stato
#      ./llmd-pf.sh stop      -> chiude tutti i forward
#
# Lezioni incorporate (root cause 2026-07-05):
# - kubectl port-forward su POD sopravvive alla morte del pod e muore solo al
#   primo tentativo di connessione -> forward SEMPRE su svc.
# - un check solo-porta (ss -ltn) scambia un forward stale per salute ->
#   alive() fa un probe HTTP reale; se la porta ascolta ma il probe fallisce,
#   il forward stale viene killato e riattaccato.
set -euo pipefail
NS=llmd-sim
KCTX=kind-llmd-sim   # contesto esplicito: il default muta (fleet-test/GKE)
KUBECTL="kubectl --context $KCTX"
EPP_LOCAL=8000      # -> svc router-epp :80 (Envoy) = endpoint OpenAI per il probe
DEC_LOCAL=18000     # -> svc llmd-sim-decode-metrics :8200 = sim diretto (/metrics vllm:prefix_cache_*)
DEC_SVC=llmd-sim-decode-metrics

epp_svc() { $KUBECTL get svc -n "$NS" -o name 2>/dev/null | grep router-epp | head -1 | sed 's#service/##'; }
alive()   { curl -sf -m 2 "http://127.0.0.1:$1/health" >/dev/null 2>&1; }
listening() { ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$1$"; }
kill_pf() { pkill -f "kubectl.*port-forward.*$1:" 2>/dev/null || true; sleep 1; }

ensure() {  # $1=local port  $2=svc name  $3=svc port  $4=label
  if alive "$1"; then echo "$4 forward :$1 already up (probe OK)"; return; fi
  if listening "$1"; then
    echo "$4 :$1 listening but probe FAILED -> killing stale forward"
    kill_pf "$1"
  fi
  [ -z "$2" ] && { echo "ERROR: $4 svc not found in $NS"; return 1; }
  nohup $KUBECTL port-forward -n "$NS" "svc/$2" "$1:$3" >"/tmp/pf-$4.log" 2>&1 &
  sleep 2
  if alive "$1"; then echo "$4 forward :$1 -> svc/$2:$3 UP (probe OK)"
  else echo "$4 forward :$1 FAILED (probe)"; tail -5 "/tmp/pf-$4.log"; return 1; fi
}

case "${1:-up}" in
  up)     ensure "$EPP_LOCAL" "$(epp_svc)" 80 epp
          ensure "$DEC_LOCAL" "$DEC_SVC" 8200 decode ;;
  status) for p in "$EPP_LOCAL" "$DEC_LOCAL"; do
            alive "$p" && echo ":$p UP (probe OK)" || echo ":$p DOWN or stale"; done ;;
  stop)   pkill -f "kubectl.*port-forward.*$NS" 2>/dev/null && echo "stopped" || echo "nothing to stop" ;;
  *)      echo "usage: $0 {up|status|stop}"; exit 1 ;;
esac
