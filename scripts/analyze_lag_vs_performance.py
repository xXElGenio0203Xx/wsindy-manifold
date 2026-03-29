#!/usr/bin/env python3
"""
Unified analysis: BIC/AIC lag selection vs empirical LSTM/MVAR performance.

Reads:
  1. Alvarez BIC/AIC/HQIC results (rom_hyperparameters/results_oscar_aligned/)
  2. Oscar experiment results (oscar_output/*/)
  3. Produces a combined table showing:
     - Information-theoretic optimal lag per experiment
     - Actual lag used → observed R²
     - Whether lag=5 was appropriate or catastrophically wrong

Usage:
    python scripts/analyze_lag_vs_performance.py
"""
import json
import os
import glob
import csv


def load_alvarez_summaries(results_dir):
    """Load all Alvarez lag selection summaries."""
    summaries = {}
    for d in sorted(glob.glob(os.path.join(results_dir, '*/'))):
        name = os.path.basename(d.rstrip('/'))
        sf = os.path.join(d, 'summary.json')
        if os.path.exists(sf):
            with open(sf) as f:
                summaries[name] = json.load(f)
    return summaries


def load_experiment_results(oscar_output_dir, patterns=None):
    """Load experiment summaries from oscar_output."""
    if patterns is None:
        patterns = ['DO_*', 'LST*', 'VLST*', 'LAG*', 'DYN*', 'NDYN*']
    
    results = {}
    for pat in patterns:
        for d in sorted(glob.glob(os.path.join(oscar_output_dir, pat, ''))):
            name = os.path.basename(d.rstrip('/'))
            sf = os.path.join(d, 'summary.json')
            if os.path.exists(sf):
                with open(sf) as f:
                    results[name] = json.load(f)
    return results


def extract_per_test_r2(oscar_output_dir, experiment_name, model='LSTM'):
    """Read per-test R² from test_results.csv."""
    csv_path = os.path.join(oscar_output_dir, experiment_name, model, 'test_results.csv')
    if not os.path.exists(csv_path):
        return None
    
    per_test = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_id = row.get('test_id', row.get('test_idx', '?'))
            ic_type = row.get('ic_type', row.get('distribution', '?'))
            r2 = float(row.get('r2_reconstructed', row.get('r2_test', 'nan')))
            per_test[f"{test_id}_{ic_type}"] = r2
    return per_test


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    alvarez_dir = os.path.join(base_dir, 'rom_hyperparameters', 'results_oscar_aligned')
    oscar_dir = os.path.join(base_dir, 'oscar_output')
    
    # 1. Load Alvarez lag selections
    alvarez = load_alvarez_summaries(alvarez_dir)
    
    # 2. Load experiment results
    experiments = load_experiment_results(oscar_dir)
    
    # =========================================================================
    # TABLE 1: BIC/AIC Lag Selection for All 32 Experiments
    # =========================================================================
    print("=" * 90)
    print("TABLE 1: Alvarez-Style BIC/AIC/HQIC Lag Selection (Aligned, w_max=50)")
    print("=" * 90)
    hdr = f"{'Experiment':<36} {'BIC':>4} {'HQIC':>5} {'AIC':>4} {'Stat':>6} {'Lag Used':>8} {'Status':>10}"
    print(hdr)
    print("-" * 90)
    
    for name, s in sorted(alvarez.items()):
        bic = s.get('lag_bic', '?')
        aic = s.get('lag_aic', '?')
        hqic = s.get('lag_hqic', '?')
        n_stat = s.get('n_stationary', '?')
        n_modes = s.get('n_modes', '?')
        
        # Check if we have experiment results with lag info
        lag_used = '5'  # default
        if name in experiments:
            lag_used = str(experiments[name].get('lag', 5))
        elif f'{name}' in experiments:
            lag_used = str(experiments[name].get('lag', 5))
        
        # Status: is lag=5 appropriate?
        if isinstance(bic, (int, float)):
            if int(lag_used) >= bic:
                status = 'OK'
            elif bic <= 5:
                status = 'OK'
            else:
                status = f'UNDER({bic})'
        else:
            status = '?'
        
        stat_str = f"{n_stat}/{n_modes}"
        print(f"{name:<36} {bic!s:>4} {hqic!s:>5} {aic!s:>4} {stat_str:>6} {lag_used:>8} {status:>10}")
    
    # =========================================================================
    # TABLE 2: LSTM Performance vs Lag
    # =========================================================================
    print()
    print("=" * 90)
    print("TABLE 2: LSTM Performance vs Lag (All LSTM Experiments)")
    print("=" * 90)
    
    # Collect all experiments with LSTM data
    lstm_results = []
    for name, s in sorted(experiments.items()):
        lstm = s.get('lstm', {})
        mvar = s.get('mvar', {})
        lstm_r2 = lstm.get('mean_r2_test', lstm.get('r2_test'))
        mvar_r2 = mvar.get('mean_r2_test', mvar.get('r2_test'))
        
        if lstm_r2 is not None:
            lag = s.get('lag', 5)
            n_test = s.get('n_test', '?')
            test_T = '?'
            
            # Get BIC recommendation if available
            base_name = name.replace('_VS', '')
            bic_lag = alvarez.get(base_name, {}).get('lag_bic', '?')
            
            lstm_results.append({
                'name': name,
                'lag': lag,
                'n_test': n_test,
                'lstm_r2': lstm_r2,
                'mvar_r2': mvar_r2,
                'bic_lag': bic_lag,
            })
    
    hdr = f"{'Experiment':<45} {'Lag':>4} {'LSTM R²':>12} {'MVAR R²':>12} {'BIC Lag':>8} {'Gap':>5}"
    print(hdr)
    print("-" * 90)
    
    for r in sorted(lstm_results, key=lambda x: x['lstm_r2'] if isinstance(x['lstm_r2'], (int, float)) else -1e10):
        gap = ''
        if isinstance(r['bic_lag'], (int, float)) and isinstance(r['lag'], (int, float)):
            gap = str(r['bic_lag'] - r['lag'])
            if r['bic_lag'] > r['lag']:
                gap = f'+{gap}'
        
        mvar_str = f"{r['mvar_r2']:.4f}" if isinstance(r['mvar_r2'], (int, float)) else '?'
        lstm_str = f"{r['lstm_r2']:.4f}" if isinstance(r['lstm_r2'], (int, float)) else '?'
        
        print(f"{r['name']:<45} {r['lag']!s:>4} {lstm_str:>12} {mvar_str:>12} {r['bic_lag']!s:>8} {gap:>5}")
    
    # =========================================================================
    # TABLE 3: CS01 Regime Deep Dive (our key experiment)
    # =========================================================================
    print()
    print("=" * 90)
    print("TABLE 3: CS01 Regime — Lag Impact on LSTM (BIC=23, HQIC=37, AIC=50)")
    print("=" * 90)
    
    cs01_experiments = {
        'DO_CS01_swarm_C01_l05': {'note': 'BUGGY pipeline (LSTM got lag=5 data despite config=20)'},
        'LAG_CS01_mvar8_lstm20': {'note': 'FIXED pipeline (LSTM got its own lag=20 data)'},
    }
    
    for ename, meta in cs01_experiments.items():
        print(f"\n--- {ename} ---")
        print(f"    Note: {meta['note']}")
        
        if ename in experiments:
            s = experiments[ename]
            print(f"    Summary lag (MVAR): {s.get('lag', '?')}")
            
            lstm = s.get('lstm', {})
            mvar = s.get('mvar', {})
            print(f"    MVAR mean R²: {mvar.get('mean_r2_test', '?')}")
            print(f"    LSTM mean R²: {lstm.get('mean_r2_test', '?')}")
        
        # Per-test breakdown
        for model in ['LSTM', 'MVAR']:
            per_test = extract_per_test_r2(oscar_dir, ename, model)
            if per_test:
                print(f"    {model} per-test R²:")
                for k, v in sorted(per_test.items()):
                    print(f"      {k}: {v:.4f}")
    
    # =========================================================================
    # TABLE 4: How Far Off Is Lag=5 from BIC Optimal?
    # =========================================================================
    print()
    print("=" * 90)
    print("TABLE 4: Lag=5 vs BIC Optimal — Distribution of Under-specification")
    print("=" * 90)
    
    bic_values = []
    hqic_values = []
    for name, s in alvarez.items():
        bic = s.get('lag_bic')
        hqic = s.get('lag_hqic')
        if isinstance(bic, (int, float)):
            bic_values.append(bic)
        if isinstance(hqic, (int, float)):
            hqic_values.append(hqic)
    
    if bic_values:
        bic_values.sort()
        print(f"\nBIC optimal lag distribution (n={len(bic_values)}):")
        print(f"  min={min(bic_values)}, median={bic_values[len(bic_values)//2]}, max={max(bic_values)}")
        print(f"  mean={sum(bic_values)/len(bic_values):.1f}")
        n_above_5 = sum(1 for v in bic_values if v > 5)
        print(f"  BIC > 5 (lag=5 insufficient): {n_above_5}/{len(bic_values)} ({100*n_above_5/len(bic_values):.0f}%)")
        n_above_10 = sum(1 for v in bic_values if v > 10)
        print(f"  BIC > 10: {n_above_10}/{len(bic_values)} ({100*n_above_10/len(bic_values):.0f}%)")
        n_above_20 = sum(1 for v in bic_values if v > 20)
        print(f"  BIC > 20: {n_above_20}/{len(bic_values)} ({100*n_above_20/len(bic_values):.0f}%)")
    
    if hqic_values:
        hqic_values.sort()
        print(f"\nHQIC optimal lag distribution (n={len(hqic_values)}):")
        print(f"  min={min(hqic_values)}, median={hqic_values[len(hqic_values)//2]}, max={max(hqic_values)}")
        print(f"  mean={sum(hqic_values)/len(hqic_values):.1f}")
        n_above_5 = sum(1 for v in hqic_values if v > 5)
        print(f"  HQIC > 5: {n_above_5}/{len(hqic_values)} ({100*n_above_5/len(hqic_values):.0f}%)")
    
    # =========================================================================
    # KEY FINDING
    # =========================================================================
    print()
    print("=" * 90)
    print("KEY FINDINGS")
    print("=" * 90)
    print("""
1. CS01 (collective swarm) has BIC=23, the HIGHEST among all 32 experiments.
   This regime needs >4x the lag we've been using (lag=5).

2. The only successful LSTM experiments (LST4/LST7, R²≈0.97) used:
   - Short test horizons (301 steps vs 1662 for DO_CS01)
   - Only uniform test ICs (no ring/gaussian/two_clusters)
   - lag=5 was shared correctly (no bug) since LSTM and MVAR both used lag=5

3. LAG_CS01 (lag=20, fixed pipeline) improved LSTM from R²=-909 → R²=-25:
   - 3/4 tests now excellent (gaussian=0.94, uniform=0.91, two_clusters=0.86)
   - ring still diverges (R²=-103)
   - But lag=20 is still BELOW the BIC optimum of 23!

4. PROPOSED: Run LSTM lag sweep {5, 10, 15, 20, 25, 30, 40} on CS01
   to find the empirical optimum and validate against BIC=23/HQIC=37.
""")


if __name__ == '__main__':
    main()
