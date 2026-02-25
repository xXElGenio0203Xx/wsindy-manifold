#!/usr/bin/env python3
"""Analyze ABL v2 re-runs and completed DYN results."""
import pandas as pd
import numpy as np

print("=" * 80)
print("ABL v2 SIMPLEX RE-RUNS (with config fix)")
print("=" * 80)

originals = {
    "ABL1 raw/none/noAlign":    "oscar_output/ABL1_N200_raw_none_noAlign_H300/test/test_results.csv",
    "ABL2 raw/none/align":      "oscar_output/ABL2_N200_raw_none_align_H300/test/test_results.csv",
    "ABL5 sqrt/none/noAlign":   "oscar_output/ABL5_N200_sqrt_none_noAlign_H300/test/test_results.csv",
    "ABL6 sqrt/none/align":     "oscar_output/ABL6_N200_sqrt_none_align_H300/test/test_results.csv",
}
v2s = {
    "ABL3v2 raw/simplex/noAlign":  "oscar_output/ABL3_N200_raw_simplex_noAlign_H300_v2/test/test_results.csv",
    "ABL4v2 raw/simplex/align":    "oscar_output/ABL4_N200_raw_simplex_align_H300_v2/test/test_results.csv",
    "ABL7v2 sqrt/simplex/noAlign": "oscar_output/ABL7_N200_sqrt_simplex_noAlign_H300_v2/test/test_results.csv",
    "ABL8v2 sqrt/simplex/align":   "oscar_output/ABL8_N200_sqrt_simplex_align_H300_v2/test/test_results.csv",
}

header = f"{'Experiment':<35} {'R2':>8} {'neg%':>8} {'mass_pp':>10} {'RMSE':>10}"
print(header)
print("-" * 75)

all_data = {}
for label, path in {**originals, **v2s}.items():
    try:
        df = pd.read_csv(path)
        r2 = df["r2_reconstructed"].mean()
        neg = df["negativity_frac"].mean()
        rmse = df["rmse_recon"].mean()
        mp = df["mass_postprocess"].iloc[0]
        print(f"{label:<35} {r2:>+8.4f} {neg:>7.1f}% {mp:>10} {rmse:>10.4f}")
        all_data[label] = r2
    except Exception as e:
        print(f"{label:<35} ERROR: {e}")

# Factor analysis
print()
print("UPDATED FACTOR EFFECTS (with real simplex):")

align_yes = np.mean([v for k, v in all_data.items() if "align" in k.lower() and "noAlign" not in k])
align_no  = np.mean([v for k, v in all_data.items() if "noAlign" in k])
print(f"  align:     noAlign={align_no:+.4f}  align={align_yes:+.4f}  effect={align_yes - align_no:+.4f}")

simplex_yes = np.mean([v for k, v in all_data.items() if "simplex" in k.lower()])
simplex_no  = np.mean([v for k, v in all_data.items() if "none" in k.lower()])
print(f"  simplex:   none={simplex_no:+.4f}  simplex={simplex_yes:+.4f}  effect={simplex_yes - simplex_no:+.4f}")

raw_vals  = np.mean([v for k, v in all_data.items() if "raw" in k.lower()])
sqrt_vals = np.mean([v for k, v in all_data.items() if "sqrt" in k.lower()])
print(f"  transform: raw={raw_vals:+.4f}  sqrt={sqrt_vals:+.4f}  effect={sqrt_vals - raw_vals:+.4f}")

# Head-to-head: simplex vs none (same transform + align combo)
print()
print("HEAD-TO-HEAD: simplex vs none (same conditions):")
pairs = [
    ("ABL1 raw/none/noAlign",   "ABL3v2 raw/simplex/noAlign"),
    ("ABL2 raw/none/align",     "ABL4v2 raw/simplex/align"),
    ("ABL5 sqrt/none/noAlign",  "ABL7v2 sqrt/simplex/noAlign"),
    ("ABL6 sqrt/none/align",    "ABL8v2 sqrt/simplex/align"),
]
for none_k, simp_k in pairs:
    if none_k in all_data and simp_k in all_data:
        delta = all_data[simp_k] - all_data[none_k]
        winner = "simplex" if delta > 0 else "none"
        print(f"  {none_k:<30} vs {simp_k:<35} delta={delta:+.4f}  ({winner})")

print()
print("=" * 80)
print("DYN SUITE (completed runs)")
print("=" * 80)
dyns = {
    "DYN5 supernova":   "oscar_output/DYN5_supernova/test/test_results.csv",
    "DYN7 pure_vicsek": "oscar_output/DYN7_pure_vicsek/test/test_results.csv",
}
header2 = f"{'Experiment':<25} {'R2':>8} {'neg%':>8} {'RMSE':>10} {'n_tests':>8}"
print(header2)
print("-" * 65)
for label, path in dyns.items():
    try:
        df = pd.read_csv(path)
        r2 = df["r2_reconstructed"].mean()
        neg = df["negativity_frac"].mean()
        rmse = df["rmse_recon"].mean()
        r2_std = df["r2_reconstructed"].std()
        print(f"{label:<25} {r2:>+8.4f}  {neg:>6.1f}% {rmse:>10.4f} {len(df):>8}")
        print(f"{'  (std)':<25} {r2_std:>8.4f}")
    except Exception as e:
        print(f"{label:<25} ERROR: {e}")
