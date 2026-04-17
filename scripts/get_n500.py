#!/usr/bin/env python3
import csv, statistics

f = '/users/emaciaso/wsindy-manifold/oscar_output/NDYN08_pure_vicsek_N0500/test/test_results.csv'
rows = list(csv.DictReader(open(f)))
r2s = [float(r['r2_reconstructed']) for r in rows]
print(f"N=500: mean_R2={statistics.mean(r2s):.3f}, gaussian_IC_R2={r2s[0]:.3f}")
print(f"  per-run: {[round(x,3) for x in r2s]}")
