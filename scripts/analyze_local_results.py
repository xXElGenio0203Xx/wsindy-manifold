#!/usr/bin/env python3
"""Comprehensive R² analysis from downloaded OSCAR results.
Builds thesis-inclusion decision table."""
import json, os, glob
import numpy as np

base = 'oscar_output/results_9apr'

def parse_experiment(exp):
    """Return (regime, config) from experiment name."""
    for prefix in ['NDYN04_', 'NDYN05_', 'NDYN06_', 'NDYN07_', 'NDYN08_']:
        if exp.startswith(prefix):
            rest = exp[len(prefix):]
            break
    else:
        return None, None
    if rest.endswith('_thesis_final'):
        return rest[:-len('_thesis_final')], 'thesis_final'
    elif rest.endswith('_tier1_w5'):
        return rest[:-len('_tier1_w5')], 'tier1_w5'
    elif rest.endswith('_tier1_bic'):
        return rest[:-len('_tier1_bic')], 'tier1_bic'
    elif 'wsindy_v3' in rest:
        return None, None
    return None, None

# Collect ALL per-test R2 values
data = {}  # (regime, model) -> [(config, [r2 per test])]
for exp_dir in sorted(glob.glob(os.path.join(base, 'NDYN*'))):
    exp = os.path.basename(exp_dir)
    regime, config = parse_experiment(exp)
    if regime is None:
        continue
    for model in ['mvar', 'lstm']:
        r2vals = []
        for tid in ['test_000', 'test_001', 'test_002', 'test_003']:
            p = os.path.join(exp_dir, 'test', tid, f'metrics_summary_{model}.json')
            if os.path.exists(p):
                with open(p) as f:
                    d = json.load(f)
                v = d.get('r2_recon', d.get('r2_reconstruction', None))
                if v is not None:
                    r2vals.append(v)
        if r2vals:
            key = (regime, model)
            if key not in data:
                data[key] = []
            data[key].append((config, r2vals))

# Print raw experiment table
print(f"{'Experiment':<50s} {'Model':<5s} {'Mean':>8s} {'Min':>8s} {'Max':>8s} {'N':>3s}")
print('-' * 85)
for exp_dir in sorted(glob.glob(os.path.join(base, 'NDYN*'))):
    exp = os.path.basename(exp_dir)
    regime, config = parse_experiment(exp)
    if regime is None:
        continue
    for model in ['mvar', 'lstm']:
        key = (regime, model)
        if key not in data:
            continue
        for cfg, vals in data[key]:
            if cfg == config:
                m = np.mean(vals)
                flag = ' ***BAD***' if model == 'lstm' and m < 0.5 else ''
                print(f"{exp:<50s} {model:<5s} {m:8.4f} {min(vals):8.4f} {max(vals):8.4f} {len(vals):3d}{flag}")
                break

# Best config per regime
regime_order = ['gas', 'blackhole', 'supernova', 'pure_vicsek', 'crystal',
                'gas_VS', 'blackhole_VS', 'supernova_VS', 'crystal_VS']

print("\n\n" + "=" * 105)
print(f"{'Regime':<20s} {'Model':<6s} {'Best Config':<15s} {'Mean R2':>8s} {'Min':>8s} {'Max':>8s} {'N':>3s}  {'Thesis?':<10s}")
print("=" * 105)

for regime in regime_order:
    for model in ['mvar', 'lstm']:
        key = (regime, model)
        if key not in data:
            print(f"{regime:<20s} {model:<6s} {'---':<15s} {'N/A':>8s} {'':>8s} {'':>8s} {'':>3s}  {'NO DATA':<10s}")
            continue
        best_mean = -999
        best_config = None
        best_vals = None
        for config, vals in data[key]:
            m = np.mean(vals)
            if m > best_mean:
                best_mean = m
                best_config = config
                best_vals = vals
        if model == 'lstm' and best_mean < 0.40:
            verdict = "REMOVE"
        elif model == 'mvar' and best_mean < 0.0:
            verdict = "REMOVE"
        elif model == 'lstm' and best_mean < 0.50:
            verdict = "MARGINAL"
        elif model == 'mvar' and best_mean < 0.40:
            verdict = "MARGINAL"
        else:
            verdict = "INCLUDE"
        print(f"{regime:<20s} {model:<6s} {best_config:<15s} {best_mean:8.4f} {min(best_vals):8.4f} {max(best_vals):8.4f} {len(best_vals):3d}  {verdict:<10s}")
    print()

# Final thesis decision table
print("\n" + "=" * 75)
print("THESIS DECISION TABLE (final values for tables)")
print("=" * 75)
print(f"{'Regime':<20s} {'MVAR R2':>10s} {'LSTM R2':>10s} {'Action'}")
print("-" * 75)
for regime in regime_order:
    mvar_key = (regime, 'mvar')
    lstm_key = (regime, 'lstm')
    if mvar_key in data:
        mvar_best = max(np.mean(v) for _, v in data[mvar_key])
    else:
        mvar_best = None
    if lstm_key in data:
        lstm_best = max(np.mean(v) for _, v in data[lstm_key])
    else:
        lstm_best = None
    mvar_str = f"{mvar_best:.3f}" if mvar_best is not None else "---"
    lstm_str = f"{lstm_best:.3f}" if lstm_best is not None else "---"
    actions = []
    if mvar_best is not None and mvar_best < 0:
        actions.append("MVAR->---")
    if lstm_best is not None and lstm_best < 0.4:
        actions.append("LSTM->---")
    if not actions:
        if mvar_best is None and lstm_best is None:
            actions.append("PENDING (no data)")
        else:
            actions.append("OK")
    print(f"{regime:<20s} {mvar_str:>10s} {lstm_str:>10s}  {'; '.join(actions)}")
