#!/usr/bin/env python3
"""Minimal /metrics stub: reports how many stub clients are 'generating'.

Gate 1 of the staggered-start design. The point is not to imitate vLLM but
to give the harness a running count it can sample, produced by processes
that alternate work and wait exactly the way a replay trajectory does. If
the analysis cannot separate lockstep from staggered here, it will not
separate them on hardware either.
"""
import http.server, threading, time, sys

RUNNING = 0
LOCK = threading.Lock()

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        with LOCK:
            body = f"vllm:num_requests_running {float(RUNNING)}\n".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a):
        pass

def worker(gen_s, tool_s, cycles):
    global RUNNING
    for _ in range(cycles):
        with LOCK:
            RUNNING += 1
        time.sleep(gen_s)      # generating
        with LOCK:
            RUNNING -= 1
        time.sleep(tool_s)     # waiting on a tool

if __name__ == "__main__":
    port = int(sys.argv[1]); offset = float(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    gen_s, tool_s, cycles = 1.5, 2.5, 8
    threading.Thread(target=lambda: http.server.HTTPServer(("127.0.0.1", port), H).serve_forever(),
                     daemon=True).start()
    time.sleep(0.3)
    import random
    rng = random.Random(42)
    arrival = sys.argv[4] if len(sys.argv) > 4 else "fixed"
    het = len(sys.argv) > 5 and sys.argv[5] == "het"
    ts = []
    for i in range(n):
        if i and offset > 0:
            time.sleep(rng.expovariate(1.0/offset) if arrival == "poisson" else offset)
        g = gen_s * (0.75, 1.0, 1.35, 1.0)[i % 4] if het else gen_s
        c = (cycles - 1, cycles, cycles + 1, cycles)[i % 4] if het else cycles
        t = threading.Thread(target=worker, args=(g, tool_s, c))
        t.start(); ts.append(t)
    for t in ts: t.join()
    time.sleep(0.5)
