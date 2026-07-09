#!/usr/bin/env python3
"""Regenerate the failed smoke cell's prompts bit-identically (seed from
manifest) and compare approx_tokens vs real Qwen2.5-32B BPE counts.
Outputs the ratio that sizes the headroom margin. CPU-only, zero-cost."""
import sys, random
sys.path.insert(0, "/root/agentic-kv-energy-experiment")
from agentic_workload import (load_prefix, approx_tokens,
                              build_session_prompt, REGIMES, SHADE)
from transformers import AutoTokenizer

SEED = 61397384191643          # from smoke manifest
REGIME, CONDITION = "H0", "nominal"
N_SESSIONS, TARGET_CTX, PREFIX_V = 2, 32768, "v1"

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")
bpe = lambda s: len(tok.encode(s, add_special_tokens=False))

prefix, prefix_toks_auth = load_prefix(PREFIX_V)
rng = random.Random(SEED)
cfg = REGIMES[REGIME]

print(f"prefix: manifest={prefix_toks_auth} approx={approx_tokens(prefix)} bpe={bpe(prefix)}")
for sess in range(N_SESSIONS):
    prompt, turns = build_session_prompt(prefix, rng, SHADE, cfg,
                                         CONDITION, TARGET_CTX, sess, SEED)
    hist = prompt[len(prefix):] if prompt.startswith(prefix) else prompt
    a, b = approx_tokens(prompt), bpe(prompt)
    ha, hb = approx_tokens(hist), bpe(hist)
    print(f"sess{sess}: turns={turns} prompt approx={a} bpe={b} ratio={b/a:.4f} | "
          f"hist approx={ha} bpe={hb} ratio={hb/ha:.4f} | over_limit={b-32768:+d}")
