#!/usr/bin/env python3
"""Scan local oscar_output for R² from test_results.csv files."""
import csv, os, statistics

base = "oscar_output"

def scan_model(model_name):
    print(f"\n=== {model_name} — r2_reconstructed from test_results.csv ===")
    print(f"  {'Experiment':<40} {'Mean_R2':>8} {'Min_R2':>8} {'Max_R2':>8} {'N':>3}")
    print("  " + "-" * 72)
    for d in sorted(os.listdir(base)):
        csv_path = os.path.join(base, d, model_name, "test_results.csv")
        if not os.path.isfile(csv_path):
            continue
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        r2v = []
        for row in rows:
            v = row.get("r2_reconstructed")
            if v is not None:
                try:
                    r2v.append(float(v))
                except ValueError:
                    pass
        if r2v:
            m = statistics.mean(r2v)
            print(f"  {d:<40} {m:>8.4f} {min(r2v):>8.4f} {max(r2v):>8.4f} {len(r2v):>3}")

scan_model("MVAR")
scan_model("LSTM")
