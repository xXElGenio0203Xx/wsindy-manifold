#!/usr/bin/env python3
"""Collect all per-model R² results from OSCAR experiments."""
import json, os, sys

base = os.path.expanduser("~/scratch/oscar_output")
rows = []

for exp_dir in sorted(os.listdir(base)):
    full = os.path.join(base, exp_dir)
    test_dir = os.path.join(full, "test")
    if not os.path.isdir(test_dir):
        continue
    if "thesis_final" not in exp_dir and "tier1" not in exp_dir:
        continue
    for tid in sorted(os.listdir(test_dir)):
        if not tid.startswith("test_"):
            continue
        td = os.path.join(test_dir, tid)
        if not os.path.isdir(td):
            continue
        for model in ["mvar", "lstm"]:
            mf = os.path.join(td, f"metrics_summary_{model}.json")
            if os.path.exists(mf):
                try:
                    d = json.load(open(mf))
                    rows.append({
                        "experiment": exp_dir,
                        "test_case": tid,
                        "model": model.upper(),
                        "r2_recon": round(d.get("r2_recon", float("nan")), 4),
                        "r2_1step": round(d.get("r2_1step", float("nan")), 4),
                        "neg_frac": round(d.get("negativity_frac", 0), 1),
                        "mass_pp": d.get("mass_postprocess", ""),
                    })
                except Exception as e:
                    print(f"ERROR: {mf}: {e}", file=sys.stderr)

# Print header
print(f"{'experiment':<40} {'test':>8} {'model':>5} {'r2_recon':>10} {'r2_1step':>10} {'neg_frac':>10} {'mass_pp':>8}")
print("-" * 95)
for r in rows:
    print(f"{r['experiment']:<40} {r['test_case']:>8} {r['model']:>5} {r['r2_recon']:>10.4f} {r['r2_1step']:>10.4f} {r['neg_frac']:>10.1f} {r['mass_pp']:>8}")

# Summary: mean R2 per experiment per model
print("\n\n=== SUMMARY: MEAN R² PER EXPERIMENT ===")
from collections import defaultdict
agg = defaultdict(list)
for r in rows:
    key = (r["experiment"], r["model"])
    agg[key].append(r["r2_recon"])

print(f"{'experiment':<40} {'model':>5} {'mean_r2':>10} {'min_r2':>10} {'max_r2':>10} {'n':>3}")
print("-" * 75)
for (exp, model), vals in sorted(agg.items()):
    n = len(vals)
    mn = sum(vals) / n
    lo = min(vals)
    hi = max(vals)
    flag = " *** BAD ***" if mn < 0.5 and model == "LSTM" else ""
    print(f"{exp:<40} {model:>5} {mn:>10.4f} {lo:>10.4f} {hi:>10.4f} {n:>3}{flag}")
