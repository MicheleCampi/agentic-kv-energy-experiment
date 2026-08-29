"""Endpoint OpenAI-compatible che muore a comando.

Gate zero-cost per la campagna preemption: serve risposte valide finché
esiste il file /tmp/stage/UP, poi rifiuta ogni connessione. Verifica che il
driver ritenti col contesto intero invece di morire, e che prompt_tokens
cresca a ogni turno come fa una conversazione reale.
"""
import http.server, json, os, time

UP = "/tmp/stage/UP"

class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        if not os.path.exists(UP):
            self.close_connection = True
            return
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        # prompt_tokens cresce con la conversazione, come nel vero engine
        ptok = sum(len(m.get("content", "")) // 4 for m in body.get("messages", []))
        time.sleep(0.3)
        out = json.dumps({
            "id": "x", "object": "chat.completion", "created": 0,
            "model": body.get("model", "m"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok " * 40},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": ptok, "completion_tokens": 40,
                      "total_tokens": ptok + 40},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)
    def log_message(self, *a): pass

if __name__ == "__main__":
    open(UP, "w").close()
    http.server.HTTPServer(("127.0.0.1", 8555), H).serve_forever()
