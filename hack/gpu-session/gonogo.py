#!/usr/bin/env python3
"""Go/no-go H1 calibration verdict (executable criterion, 2026-07-10).

Reads the calibration manifests and applies the frozen rules from
00-env.sh verbatim:
  median in [0.40,0.60]                     -> GREEN  (matrix)
  median in [0.30,0.40) or (0.60,0.70]     -> YELLOW (ONE adaptive run)
  median outside [0.30,0.70]               -> RED    (abort)
  intra spread = max-min > 0.04            -> RED regardless of median
    (empirical n=8 envelope: max-min 0.031 on 8 reps stays green; the
     2026-07-10 first-cell artefact produced 0.0689 and MUST stay red —
     the cure is the warm-up cell, not a wider gate)

Usage: gonogo.py <calib-out-dir>   # exit 0 GREEN, 2 YELLOW, 1 RED
"""
import json, glob, statistics, sys

d = sys.argv[1] if len(sys.argv) > 1 else "."
vals = []
for p in sorted(glob.glob(f"{d}/H1_nominal_rep*/manifest.json")):
    m = json.load(open(p))
    r = m.get("hitrate_realized")
    if r is None:
        print(f"RED: {p} has no realized hit-rate (scrape failed?)")
        sys.exit(1)
    vals.append((m["rep"], r))
    print(f"rep{m['rep']}: realized={r:.4f}")
if len(vals) < 3:
    print(f"RED: only {len(vals)} calibration manifests found, need 3")
    sys.exit(1)
rs = [r for _, r in vals]
med = statistics.median(rs)
spread = max(rs) - min(rs)
print(f"median={med:.4f} spread(max-min)={spread:.4f}")
if spread > 0.04:
    print(f"VERDICT: RED — intra spread {spread:.4f} > 0.04 (regardless of median)")
    sys.exit(1)
if 0.40 <= med <= 0.60:
    print("VERDICT: GREEN — proceed to matrix")
    sys.exit(0)
if 0.30 <= med < 0.40 or 0.60 < med <= 0.70:
    print("VERDICT: YELLOW — ONE adaptive run (linear history_shared_frac correction), document deviation")
    sys.exit(2)
print(f"VERDICT: RED — median {med:.4f} outside [0.30,0.70], abort, root cause off-node")
sys.exit(1)
