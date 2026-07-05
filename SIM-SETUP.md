# llmd-sim setup required for calibration at realistic scale

The stock llm-d-inference-sim v0.8.2 deployments (kind cluster `llmd-sim`,
namespace `llmd-sim`) ship with defaults sized for toy validation. For the
calibration campaign (prefix ~15K tok + history, target context 32-48K),
BOTH the prefill and decode deployments need:

- `--max-model-len 65536` (default 1024 rejects the payload with HTTP 400)
- `--kv-cache-size 4096` blocks (x block-size 16 = 65K tokens; the default
  1024 blocks = 16K tokens barely covers the shared prefix -> thrashing)
- memory limit `1536Mi`, request `512Mi` (the stock 100Mi limit OOM-kills
  the pod under long-context load; observed: HTTP 503 -> 502 as the
  backend crashed mid-campaign)

Ports differ per role: prefill 8000, decode 8200. Args are a single shell
string in the pod spec; patch with a JSON-patch replace on
`/spec/template/spec/containers/0/args/0` (see git history of this file's
introducing commit for the exact kubectl invocations).

Measurement notes (encoded in run_experiment.py):
- scrape counters PER POD via the API-server pod proxy and sum
  prefill+decode; scraping through the EPP router samples one random
  backend per request.
- isolate campaigns via `--run-nonce` (seed offset): the sim's KV cache
  persists across runs and a warm cache from a previous campaign pollutes
  realized hit-rate.
