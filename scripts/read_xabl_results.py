#!/usr/bin/env python3
"""Read XABL R² results from test_results.csv files."""
import json, os, glob, csv, statistics

for d in sorted(glob.glob(os.path.expanduser('~/wsindy-manifold/oscar_output/XABL*'))):
    name = os.path.basename(d)
    tr = os.path.join(d, 'test', 'test_results.csv')
    if os.path.exists(tr):
        with open(tr) as f:
            reader = csv.DictReader(f)
            r2s = [float(r['r2_reconstructed']) for r in reader if 'r2_reconstructed' in r]
        if r2s:
            m = statistics.mean(r2s)
            s = statistics.stdev(r2s) if len(r2s) > 1 else 0
            print(f'{name}: R2={m:.4f} +/- {s:.4f} ({len(r2s)} tests)')
        else:
            print(f'{name}: no R2 data in CSV')
    else:
        # try summary.json
        for sf in ['MVAR/summary.json', 'summary.json']:
            fp = os.path.join(d, sf)
            if os.path.exists(fp):
                data = json.load(open(fp))
                keys = list(data.keys())
                print(f'{name}: summary keys = {keys[:10]}')
                break
        else:
            print(f'{name}: no results found')
