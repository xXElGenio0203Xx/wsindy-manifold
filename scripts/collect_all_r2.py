#!/usr/bin/env python3
"""Collect density R², latent R², and 1-step R² for all experiments."""
import csv, statistics, os

experiments = [
    ('DYN1_gentle_v2', 'test', 'MVAR'),
    ('DYN2_hypervelocity_v2', 'test', 'MVAR'),
    ('DYN3_hypernoisy_v2', 'test', 'MVAR'),
    ('DYN4_blackhole_v2', 'test', 'MVAR'),
    ('DYN5_supernova', 'test', 'MVAR'),
    ('DYN6_varspeed_v2', 'test', 'MVAR'),
    ('DYN7_pure_vicsek', 'test', 'MVAR'),
    ('LST4_sqrt_simplex_align_h64_L2', 'MVAR', 'MVAR'),
    ('LST4_sqrt_simplex_align_h64_L2', 'LSTM', 'LSTM'),
    ('LST7_raw_none_align_h128_L2', 'MVAR', 'MVAR'),
    ('LST7_raw_none_align_h128_L2', 'LSTM', 'LSTM'),
    ('DEG1_long_horizon_200s', 'MVAR', 'MVAR'),
]

print(f"{'Experiment':<40s} {'Model':<6s} {'Density R²':>12s} {'Latent R²':>12s} {'1-step R²':>12s} {'n':>4s}")
print("-" * 92)

for exp, sub, model in experiments:
    csv_path = f'oscar_output/{exp}/{sub}/test_results.csv'
    if not os.path.exists(csv_path):
        # Try alternate location
        csv_path = f'oscar_output/{exp}/test/test_results.csv'
    if not os.path.exists(csv_path):
        print(f'{exp:<40s} {model:<6s} {"MISSING":>12s}')
        continue
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    r2d = [float(r['r2_reconstructed']) for r in rows]
    r2l = [float(r['r2_latent']) for r in rows]
    r2_1s = []
    for r in rows:
        v = r.get('r2_1step', '')
        if v and v != 'nan':
            try:
                r2_1s.append(float(v))
            except ValueError:
                pass

    md = statistics.mean(r2d)
    ml = statistics.mean(r2l)
    sd = statistics.stdev(r2d) if len(r2d) > 1 else 0
    sl = statistics.stdev(r2l) if len(r2l) > 1 else 0
    m1 = statistics.mean(r2_1s) if r2_1s else float('nan')

    d_str = f"{md:.4f}±{sd:.4f}"
    l_str = f"{ml:.4f}±{sl:.4f}"
    s_str = f"{m1:.4f}" if r2_1s else "N/A"
    print(f'{exp:<40s} {model:<6s} {d_str:>12s} {l_str:>12s} {s_str:>12s} {len(r2d):>4d}')
