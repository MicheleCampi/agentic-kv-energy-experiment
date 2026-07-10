#!/usr/bin/env python3
"""Fake vLLM engine for the node-off dress rehearsal (2026-07-10).

Stands in for `vllm serve` so the orchestrator's GPU branch can be
exercised end-to-end at zero GPU cost: /health readiness, EngineCore
PID discovery (spawns a child whose cmdline matches VLLM::EngineCore),
/metrics with growing vllm:prefix_cache_*_total counters, and
/v1/completions accepting requests. NOT a simulator: hit-rates are
synthetic (hits = 50% of queries), only the PLUMBING is real.
"""
import json, subprocess, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8000
MODEL = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "fake"
LOCK = threading.Lock()
STATE = {"queries": 0, "hits": 0}

# Child with VLLM::EngineCore in cmdline, same process group (pgrep -g target)
child = subprocess.Popen(["bash", "-c", "exec -a VLLM::EngineCore sleep 3600"])

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, body, ctype="text/plain"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body.encode())
    def do_GET(self):
        if self.path == "/health":
            self._send(200, "ok")
        elif self.path == "/v1/models":
            self._send(200, json.dumps({"data": [{"id": MODEL}]}),
                       "application/json")
        elif self.path == "/metrics":
            with LOCK:
                q, h = STATE["queries"], STATE["hits"]
            self._send(200,
                f'vllm:prefix_cache_hits_total{{model_name="fake"}} {h}\n'
                f'vllm:prefix_cache_queries_total{{model_name="fake"}} {q}\n')
        else:
            self._send(404, "nf")
    def do_POST(self):
        if self.path == "/v1/completions":
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n)) if n else {}
            ptoks = max(1, len(body.get("prompt", "")) // 4)
            with LOCK:
                STATE["queries"] += ptoks
                STATE["hits"] += ptoks // 2
            self._send(200, json.dumps({"choices": [{"text": "ok"}]}),
                       "application/json")
        else:
            self._send(404, "nf")

print(f"fake engine on :{PORT}, EngineCore child pid={child.pid}", flush=True)
try:
    HTTPServer(("127.0.0.1", PORT), H).serve_forever()
finally:
    child.kill()
