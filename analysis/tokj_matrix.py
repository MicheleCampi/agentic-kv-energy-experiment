#!/usr/bin/env python3
"""tok/J table from matrix evidence: manifest.json + inferscope.json per cell."""
import argparse, json, statistics as st
from pathlib import Path

def load_cell(d: Path):
    m = json.load(open(d / "manifest.json"))
    i = json.load(open(d / "inferscope.json"))
    pe = i["phase_energy"]
    prompt = pe["prompt_tokens_delta"]
    gen = pe["generation_tokens_delta"]
    e_window_j = i["gpu"]["energy_millijoules"] / 1000.0
    e_phase_j = (pe["energy_prefill_by_time_mj"] + pe["energy_decode_by_time_mj"]) / 1000.0
    return {
        "cell": d.name,
        "regime": m["regime"], "condition": m["condition"], "rep": m["rep"],
        "realized": m["hitrate_realized"],
        "growthstop": m["sessions_growth_stopped"],
        "wall_s": m["gpu"]["generator_wall_s"],
        "energy_source": i["gpu"]["energy_source"],
        "prompt_tok": prompt, "gen_tok": gen, "tot_tok": prompt + gen,
        "e_window_j": e_window_j, "e_phase_j": e_phase_j,
        "divergence": pe["phase_energy_divergence"],
        "tokj_tot_window": (prompt + gen) / e_window_j,
        "tokj_tot_phase": (prompt + gen) / e_phase_j,
        "tokj_gen_window": gen / e_window_j,
        "tokj_gen_phase": gen / e_phase_j,
    }

def fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix-dir", required=True)
    ap.add_argument("--exclude", action="append", default=[], help="cell dir name to exclude from aggregates")
    args = ap.parse_args()

    cells = sorted(p for p in Path(args.matrix_dir).iterdir()
                   if p.is_dir() and (p / "manifest.json").exists())
    rows = [load_cell(c) for c in cells]

    bad = [r for r in rows if r["energy_source"] != "counter"]
    if bad:
        print(f"WARNING: energy_source != counter in: {[r['cell'] for r in bad]}\n")

    hdr = ["cell", "realized", "gstop", "wall_s", "tot_tok", "gen_tok",
           "E_win_J", "E_phase_J", "div", "tokJ_tot_win", "tokJ_tot_ph",
           "tokJ_gen_win", "tokJ_gen_ph"]
    print("## Per-cell")
    print(" | ".join(hdr))
    for r in rows:
        excl = " *EXCL*" if r["cell"] in args.exclude else ""
        print(" | ".join([
            r["cell"] + excl, fmt(r["realized"], 4), str(r["growthstop"]),
            fmt(r["wall_s"], 1), str(r["tot_tok"]), str(r["gen_tok"]),
            fmt(r["e_window_j"], 1), fmt(r["e_phase_j"], 1), fmt(r["divergence"], 4),
            fmt(r["tokj_tot_window"]), fmt(r["tokj_tot_phase"]),
            fmt(r["tokj_gen_window"]), fmt(r["tokj_gen_phase"]),
        ]))

    print("\n## Aggregates (mean ± std over reps)")
    keys = ["tokj_tot_window", "tokj_tot_phase", "tokj_gen_window", "tokj_gen_phase"]
    groups = {}
    for r in rows:
        if r["cell"] in args.exclude:
            continue
        groups.setdefault((r["regime"], r["condition"]), []).append(r)
    print(" | ".join(["regime", "cond", "n"] + keys))
    for (reg, cond), rs in sorted(groups.items()):
        agg = []
        for k in keys:
            vs = [r[k] for r in rs]
            mu = st.mean(vs)
            sd = st.stdev(vs) if len(vs) > 1 else 0.0
            agg.append(f"{mu:.3f} ± {sd:.3f}")
        print(" | ".join([reg, cond, str(len(rs))] + agg))

if __name__ == "__main__":
    main()
