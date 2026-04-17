#!/usr/bin/env python3
"""Extract noise sweep R² from _slim and _ws dirs."""
import csv, statistics, os, glob

base = os.path.expanduser("~/wsindy-manifold/oscar_output")

# Find all relevant dirs
for pattern in ["*eta*_slim", "*eta*_ws"]:
    dirs = sorted(glob.glob(os.path.join(base, pattern)))
    for d in dirs:
        name = os.path.basename(d)
        mvar_f = os.path.join(d, "MVAR", "test_results.csv")
        lstm_f = os.path.join(d, "LSTM", "test_results.csv")
        mvar_r2 = "---"
        lstm_r2 = "---"
        if os.path.exists(mvar_f):
            rows = list(csv.DictReader(open(mvar_f)))
            r2s = [float(r["r2_reconstructed"]) for r in rows]
            mvar_r2 = f"{statistics.mean(r2s):.3f}"
        if os.path.exists(lstm_f):
            rows = list(csv.DictReader(open(lstm_f)))
            r2s = [float(r["r2_reconstructed"]) for r in rows]
            lstm_r2 = f"{statistics.mean(r2s):.3f}"
        if mvar_r2 != "---" or lstm_r2 != "---":
            print(f"{name:45s}  MVAR={mvar_r2:>8s}  LSTM={lstm_r2:>8s}")

# Also check N-convergence
print("\n=== N-convergence ===")
for n in ["N0050", "N0100", "N0200", "N0300", "N0500", "N1000"]:
    d = os.path.join(base, f"NDYN08_pure_vicsek_{n}")
    mvar_f = os.path.join(d, "MVAR", "test_results.csv")
    lstm_f = os.path.join(d, "LSTM", "test_results.csv")
    test_f = os.path.join(d, "test", "test_results.csv")
    mvar_r2 = "---"
    lstm_r2 = "---"
    gauss_r2 = "---"
    for f_path, label in [(mvar_f, "MVAR"), (test_f, "test")]:
        if os.path.exists(f_path):
            rows = list(csv.DictReader(open(f_path)))
            r2s = [float(r["r2_reconstructed"]) for r in rows]
            if label == "test":
                mvar_r2 = f"{statistics.mean(r2s):.3f}"
                if len(r2s) > 0:
                    gauss_r2 = f"{r2s[0]:.3f}"
            else:
                mvar_r2 = f"{statistics.mean(r2s):.3f}"
                if len(r2s) > 0:
                    gauss_r2 = f"{r2s[0]:.3f}"
    if os.path.exists(lstm_f):
        rows = list(csv.DictReader(open(lstm_f)))
        r2s = [float(r["r2_reconstructed"]) for r in rows]
        lstm_r2 = f"{statistics.mean(r2s):.3f}"
    print(f"{n}  MVAR={mvar_r2:>8s}  LSTM={lstm_r2:>8s}  Gauss={gauss_r2:>8s}")
