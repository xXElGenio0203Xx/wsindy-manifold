#!/usr/bin/env python3
"""Collect all results from oscar_output — comprehensive dump."""
import csv, os, glob, json

base = '/users/emaciaso/scratch/oscar_output'

print("=== ALL EXPERIMENTS: MVAR/LSTM (non-tier1) ===")
for d in sorted(glob.glob(os.path.join(base, 'NDYN*'))):
    name = os.path.basename(d)
    if 'tier1' in name:
        continue
    for model in ['MVAR', 'LSTM']:
        csv_path = os.path.join(d, model, 'test_results.csv')
        if os.path.exists(csv_path):
            rows = list(csv.DictReader(open(csv_path)))
            vals = [float(r['r2_reconstructed']) for r in rows]
            mean_r2 = round(sum(vals)/len(vals), 4) if vals else None
            print(f'{name} | {model} | {mean_r2}')

print()
print("=== TIER1 (N=300): MVAR/LSTM ===")
for d in sorted(glob.glob(os.path.join(base, 'NDYN*tier1*'))):
    name = os.path.basename(d)
    for model in ['MVAR', 'LSTM']:
        csv_path = os.path.join(d, model, 'test_results.csv')
        if os.path.exists(csv_path):
            rows = list(csv.DictReader(open(csv_path)))
            vals = [float(r['r2_reconstructed']) for r in rows]
            mean_r2 = round(sum(vals)/len(vals), 4) if vals else None
            print(f'{name} | {model} | {mean_r2}')

print()
print("=== WSINDY V3 MODELS ===")
for d in sorted(glob.glob(os.path.join(base, 'NDYN*'))):
    name = os.path.basename(d)
    for wdir in ['wsindy_v3', 'wsindy']:
        model_path = os.path.join(d, wdir, 'multifield_model.json')
        if os.path.exists(model_path):
            with open(model_path) as f:
                mdata = json.load(f)
            r2 = mdata.get('r2_wf', {})
            n_active = mdata.get('n_active_terms', '?')
            r2_rho = r2.get('rho', '?')
            print(f'{name}/{wdir} | r2_wf_rho={r2_rho} | |A|={n_active}')

print()
print("=== REEVAL SCALE RESULTS ===")
for d in sorted(glob.glob(os.path.join(base, 'NDYN*'))):
    name = os.path.basename(d)
    scale_dir = os.path.join(d, 'reeval_scale')
    if not os.path.isdir(scale_dir):
        continue
    for model in ['MVAR', 'LSTM']:
        csv_path = os.path.join(scale_dir, model, 'test_results.csv')
        if os.path.exists(csv_path):
            rows = list(csv.DictReader(open(csv_path)))
            vals = [float(r['r2_reconstructed']) for r in rows]
            mean_r2 = round(sum(vals)/len(vals), 4) if vals else None
            print(f'{name}/reeval_scale | {model} | {mean_r2}')
