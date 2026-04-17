#!/usr/bin/env python3
"""Extract N-convergence R² from OSCAR output."""
import csv, statistics, os

base = os.path.expanduser("~/wsindy-manifold/oscar_output")

print("=== N-convergence ===")
for n in ["N0050", "N0100", "N0200", "N0300", "N0500", "N1000"]:
    d = os.path.join(base, f"NDYN08_pure_vicsek_{n}")
    mvar_f = os.path.join(d, "MVAR", "test_results.csv")
    lstm_f = os.path.join(d, "LSTM", "test_results.csv")
    mvar_r2, lstm_r2, mvar_g = "---", "---", "---"
    if os.path.exists(mvar_f):
        rows = list(csv.DictReader(open(mvar_f)))
        r2s = [float(r["r2_reconstructed"]) for r in rows]
        mvar_r2 = f"{statistics.mean(r2s):.3f}"
        if r2s:
            mvar_g = f"{r2s[0]:.3f}"
    if os.path.exists(lstm_f):
        rows = list(csv.DictReader(open(lstm_f)))
        r2s = [float(r["r2_reconstructed"]) for r in rows]
        lstm_r2 = f"{statistics.mean(r2s):.3f}"
    print(f"{n}  MVAR={mvar_r2:>8s}  LSTM={lstm_r2:>8s}  MVAR_gauss={mvar_g:>8s}")
