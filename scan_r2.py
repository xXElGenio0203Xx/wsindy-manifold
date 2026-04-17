#!/usr/bin/env python3
"""Scan all old experiments for R² values and compare against thesis."""
import csv, os, glob

THESIS = {
    'gas':     {'mvar': 0.995, 'lstm': 0.574, 'wsindy': 0.9998},
    'BH':      {'mvar': 0.990, 'lstm': 0.989, 'wsindy': 0.9995},
    'SN':      {'mvar': 0.545, 'lstm': 0.508, 'wsindy': None},
    'PV':      {'mvar': 0.465, 'lstm': None,  'wsindy': 0.99997},
    'gas_VS':  {'mvar': 0.558, 'lstm': None,  'wsindy': None},
    'BH_VS':   {'mvar': -0.433,'lstm': None,  'wsindy': 0.952},
    'SN_VS':   {'mvar': 0.120, 'lstm': None,  'wsindy': 0.997},
}

# Map experiment directory name -> thesis regime
REGIME_MAP = {}
# Gas (NDYN04)
for n in ['NDYN04_gas', 'DYN1_gentle', 'NDYN04_gas_thesis_final',
          'NDYN04_gas_tier1_w5', 'NDYN04_gas_tier1_bic']:
    REGIME_MAP[n] = 'gas'
# Blackhole (NDYN05)
for n in ['NDYN05_blackhole', 'DYN4_blackhole', 'DYN4_blackhole_v2',
          'NDYN05_blackhole_thesis_final', 'NDYN05_blackhole_tier1_w5',
          'NDYN05_blackhole_tier1_bic']:
    REGIME_MAP[n] = 'BH'
# Supernova (NDYN06)
for n in ['NDYN06_supernova', 'DYN5_supernova', 'NDYN06_supernova_thesis_final',
          'NDYN06_supernova_tier1_w5', 'NDYN06_supernova_tier1_bic']:
    REGIME_MAP[n] = 'SN'
# Pure Vicsek (NDYN08)
for n in ['NDYN08_pure_vicsek', 'DYN7_pure_vicsek',
          'NDYN08_pure_vicsek_thesis_final', 'NDYN08_pure_vicsek_tier1_w5',
          'NDYN08_pure_vicsek_tier1_bic']:
    REGIME_MAP[n] = 'PV'
# Gas VS
for n in ['NDYN04_gas_VS', 'NDYN04_gas_VS_thesis_final',
          'NDYN04_gas_VS_tier1_w5', 'NDYN04_gas_VS_tier1_bic',
          'VDYN1_gentle_varspeed']:
    REGIME_MAP[n] = 'gas_VS'
# BH VS
for n in ['NDYN05_blackhole_VS', 'NDYN05_blackhole_VS_thesis_final',
          'NDYN05_blackhole_VS_tier1_w5', 'NDYN05_blackhole_VS_tier1_bic',
          'VDYN4_blackhole_varspeed']:
    REGIME_MAP[n] = 'BH_VS'
# SN VS
for n in ['NDYN06_supernova_VS', 'NDYN06_supernova_VS_thesis_final',
          'NDYN06_supernova_VS_tier1_w5', 'NDYN06_supernova_VS_tier1_bic',
          'VDYN5_supernova_varspeed']:
    REGIME_MAP[n] = 'SN_VS'

# Potentially similar regimes (different params but same family)
# DYN2_hypervelocity = gas-like (v0=10)
# DYN3_hypernoisy = gas-like (high eta)
# VDYN7_pure_vicsek_varspeed = PV variant but with varspeed

results = []

for path in sorted(glob.glob('oscar_output/**/test_results.csv', recursive=True)):
    parts = path.split('/')
    exp_name = parts[1]
    if exp_name in ('systematics', 'results_9apr'):
        exp_name = parts[2] if len(parts) > 2 else exp_name

    # Determine model type from path
    model = None
    if '/MVAR/' in path:
        model = 'mvar'
    elif '/LSTM/' in path:
        model = 'lstm'
    elif '/WSINDy/' in path or '/wsindy/' in path:
        model = 'wsindy'
    elif '/test/' in path:
        model = 'legacy'
    else:
        model = 'unknown'

    regime = REGIME_MAP.get(exp_name)

    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            continue

        r2_vals = []
        for r in rows:
            for col in ['r2_reconstructed', 'r2_recon', 'R2', 'r2']:
                if col in r and r[col] not in ('', 'nan', 'None', None):
                    try:
                        val = float(r[col])
                        r2_vals.append(val)
                    except:
                        pass
                    break

        if r2_vals:
            mean_r2 = sum(r2_vals) / len(r2_vals)
            max_r2 = max(r2_vals)
            results.append({
                'path': path, 'exp': exp_name, 'model': model,
                'regime': regime, 'mean_r2': mean_r2, 'max_r2': max_r2,
                'n_tests': len(r2_vals),
                'cols': list(rows[0].keys()) if rows else [],
            })
    except Exception as e:
        pass

# Print results grouped by thesis regime
hdr = f"{'Regime':<10} {'Experiment':<45} {'Model':<8} {'Mean R2':>10} {'Max R2':>10} {'N':>4}  {'Thesis':>10}  {'Better?':>8}"
print('=' * len(hdr))
print(hdr)
print('=' * len(hdr))

for regime in ['gas', 'BH', 'SN', 'PV', 'gas_VS', 'BH_VS', 'SN_VS']:
    matching = [r for r in results if r['regime'] == regime]
    matching.sort(key=lambda x: -x['mean_r2'])
    thesis_vals = THESIS[regime]

    for r in matching:
        model = r['model']
        thesis_r2 = thesis_vals.get(model)
        if thesis_r2 is None and model == 'legacy':
            thesis_r2 = thesis_vals.get('mvar')

        better = ''
        if thesis_r2 is not None and r['mean_r2'] > thesis_r2:
            better = '*** YES'
        elif thesis_r2 is not None and r['mean_r2'] > thesis_r2 - 0.01:
            better = '~same'

        thesis_str = f"{thesis_r2:.4f}" if thesis_r2 is not None else '---'
        print(f"{regime:<10} {r['exp']:<45} {model:<8} {r['mean_r2']:>10.4f} {r['max_r2']:>10.4f} {r['n_tests']:>4}  {thesis_str:>10}  {better:>8}")

    if matching:
        print('-' * len(hdr))

# Unmatched experiments
print()
print('=' * len(hdr))
print('UNMATCHED experiments (not mapped to thesis regime):')
print('=' * len(hdr))
unmatched = [r for r in results if r['regime'] is None]
unmatched.sort(key=lambda x: (x['exp'], x['model']))
for r in unmatched:
    print(f"{r['exp']:<45} {r['model']:<8} {r['mean_r2']:>10.4f} {r['max_r2']:>10.4f} {r['n_tests']:>4}  {r['path']}")
