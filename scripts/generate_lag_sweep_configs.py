#!/usr/bin/env python3
"""
Generate LSTM lag sweep configs for CS01 regime.

Creates configs for LSTM lags: {5, 10, 15, 20, 25, 30, 40}
MVAR fixed at lag=5 (its empirical optimum).
Based on lag_experiment_CS01.yaml.

BIC optimal for CS01 = 23, HQIC = 37, AIC = 50.

Usage:
    python scripts/generate_lag_sweep_configs.py
"""
import yaml
import os
import copy

BASE_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'configs', 'lag_experiment_CS01.yaml')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'configs', 'lag_sweep')

LSTM_LAGS = [5, 10, 15, 20, 25, 30, 40]
MVAR_LAG = 5  # Fixed — BIC=23 is for the system, but linear MVAR at lag=5 already R²=0.95


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(BASE_CONFIG) as f:
        base = yaml.safe_load(f)
    
    configs_created = []
    
    for lstm_lag in LSTM_LAGS:
        config = copy.deepcopy(base)
        
        # Set experiment name
        exp_name = f"LSWEEP_CS01_mvar{MVAR_LAG}_lstm{lstm_lag}"
        config['experiment_name'] = exp_name
        
        # Set MVAR lag
        config['rom']['models']['mvar']['lag'] = MVAR_LAG
        config['rom']['models']['mvar']['ridge_alpha'] = 0.0001
        config['rom']['models']['mvar']['enabled'] = True
        
        # Set LSTM lag
        config['rom']['models']['lstm']['lag'] = lstm_lag
        config['rom']['models']['lstm']['enabled'] = True
        config['rom']['models']['lstm']['hidden_units'] = 128
        config['rom']['models']['lstm']['num_layers'] = 2
        config['rom']['models']['lstm']['dropout'] = 0.1
        config['rom']['models']['lstm']['residual'] = True
        config['rom']['models']['lstm']['use_layer_norm'] = True
        config['rom']['models']['lstm']['max_epochs'] = 300
        config['rom']['models']['lstm']['patience'] = 30
        config['rom']['models']['lstm']['lr'] = 0.001
        config['rom']['models']['lstm']['batch_size'] = 256
        config['rom']['models']['lstm']['grad_clip'] = 1.0
        
        # Header comment
        header = (
            f"# LSTM LAG SWEEP: CS01 regime, MVAR lag={MVAR_LAG}, LSTM lag={lstm_lag}\n"
            f"# BIC optimal lag=23, HQIC=37, AIC=50 for this regime\n"
            f"# Purpose: Empirical validation of information-theoretic lag selection\n"
        )
        
        out_path = os.path.join(OUTPUT_DIR, f"{exp_name}.yaml")
        with open(out_path, 'w') as f:
            f.write(header)
            f.write("---\n")
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        configs_created.append((exp_name, lstm_lag, out_path))
        print(f"  Created: {exp_name} (LSTM lag={lstm_lag})")
    
    print(f"\n{len(configs_created)} configs created in {OUTPUT_DIR}/")
    print(f"\nLag sweep: {LSTM_LAGS}")
    print(f"MVAR fixed at lag={MVAR_LAG}")
    print(f"BIC=23, HQIC=37, AIC=50 for CS01")
    
    return configs_created


if __name__ == '__main__':
    main()
