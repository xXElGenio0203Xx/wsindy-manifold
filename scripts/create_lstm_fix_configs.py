#!/usr/bin/env python3
"""
Generate LFIX configs: Apply LST4's proven LSTM architecture + sqrt/simplex
to the 6 hardest DO_ regimes.

Key changes from DO_ baseline:
  ROM:
    density_transform: raw → sqrt
    mass_postprocess: none → simplex
  LSTM (match LST4):
    hidden_units: 128 → 64
    learning_rate: 0.001 → 0.0007
    batch_size: 256 → 512
    gradient_clip: 1.0 → 5.0
    patience: 30 → 40
    weight_decay: 1e-5 (add)
    normalize_input: true (explicit, already default)

Note: Keep the default LSTM lag at 5 unless a regime-specific sweep justifies
something larger. Oscar results showed that transformed densities helped much
more consistently than a global lag=20 default.
"""

import yaml
import os
import copy

# Base template from DO_CS01 (all DO_ share identical structure)
BASE_TEMPLATE = {
    'sim': {
        'N': 300,
        'T': 20.0,
        'dt': 0.04,
        'Lx': 25.0,
        'Ly': 25.0,
        'bc': 'periodic',
    },
    'test_sim': {'T': 200.0},
    'model': {
        'type': 'discrete',
        'speed': 3.0,
        'speed_mode': 'constant_with_forces',
    },
    'params': {'R': 4.0},
    'noise': {
        'kind': 'gaussian',
        'eta': 0.2,
        'match_variance': True,
    },
    'forces': {
        'enabled': True,
        'type': 'morse',
        'params': {
            'Ca': None,  # filled per-experiment
            'Cr': None,
            'la': 1.0,
            'lr': None,
            'mu_t': 0.3,
            'rcut_factor': 5.0,
        },
    },
    'alignment': {'enabled': True},
    'density': {'nx': 64, 'ny': 64, 'bandwidth': 5.0},
    'outputs': {'density_resolution': 64, 'density_bandwidth': 5.0},
    'train_ic': {
        'type': 'mixed_comprehensive',
        'gaussian': {
            'enabled': True,
            'n_runs': 80,
            'positions_x': [5.0, 10.0, 15.0, 20.0],
            'positions_y': [5.0, 10.0, 15.0, 20.0],
            'variances': [1.5, 3.0, 5.0],
            'n_samples_per_config': 1,
        },
        'uniform': {'enabled': True, 'n_runs': 40, 'n_samples': 40},
        'two_clusters': {
            'enabled': True,
            'n_runs': 24,
            'separations': [4.0, 7.0, 10.0],
            'sigmas': [1.5, 3.0],
            'n_samples_per_config': 4,
        },
        'ring': {
            'enabled': True,
            'n_runs': 24,
            'radii': [3.0, 5.0, 8.0],
            'widths': [0.5, 1.0],
            'n_samples_per_config': 4,
        },
    },
    'test_ic': {
        'type': 'mixed_test_comprehensive',
        'gaussian': {
            'enabled': True,
            'n_runs': 1,
            'test_positions_x': [12.5],
            'test_positions_y': [12.5],
            'test_variances': [2.5],
            'n_samples_per_config': 1,
        },
        'uniform': {'enabled': True, 'n_runs': 1},
        'two_clusters': {
            'enabled': True,
            'n_runs': 1,
            'test_separations': [7.0],
            'test_sigmas': [2.0],
            'n_samples_per_config': 1,
        },
        'ring': {
            'enabled': True,
            'n_runs': 1,
            'test_radii': [5.0],
            'test_widths': [0.8],
            'n_samples_per_config': 1,
        },
    },
    'rom': {
        'subsample': 3,
        'fixed_modes': 19,
        'density_transform': 'sqrt',           # ← changed from raw
        'density_transform_eps': 1e-10,
        'shift_align': True,
        'shift_align_ref': 'mean',
        'save_unaligned_pod': True,
        'mass_postprocess': 'simplex',          # ← changed from none
        'models': {
            'mvar': {
                'enabled': True,
                'lag': 5,
                'ridge_alpha': 0.0001,
            },
            'lstm': {
                'enabled': True,
                'lag': 5,
                'hidden_units': 64,             # ← LST4 (was 128)
                'num_layers': 2,
                'dropout': 0.1,
                'residual': True,
                'use_layer_norm': True,
                'normalize_input': True,         # ← explicit
                'max_epochs': 300,
                'patience': 40,                  # ← LST4 (was 30)
                'learning_rate': 0.0007,         # ← LST4 (was 0.001)
                'batch_size': 512,               # ← LST4 (was 256)
                'gradient_clip': 5.0,            # ← LST4 (was 1.0)
                'weight_decay': 1e-5,            # ← LST4 (was absent)
            },
        },
    },
    'eval': {
        'metrics': ['r2', 'rmse'],
        'save_forecasts': True,
        'save_time_resolved': True,
        'forecast_start': 0.6,
        'clamp_mode': 'C0',
    },
}

# The 6 target regimes
EXPERIMENTS = [
    {
        'name': 'LFIX_CS01',
        'base': 'DO_CS01_swarm_C01_l05',
        'desc': 'Collective swarm (Ca=1.0, Cr=0.1, lr=0.5) — BIC=23, hardest regime',
        'Ca': 1.0, 'Cr': 0.1, 'lr_morse': 0.5,
    },
    {
        'name': 'LFIX_DR01',
        'base': 'DO_DR01_dring_C01_l01',
        'desc': 'Double ring (Ca=1.0, Cr=0.1, lr=0.1)',
        'Ca': 1.0, 'Cr': 0.1, 'lr_morse': 0.1,
    },
    {
        'name': 'LFIX_EC02',
        'base': 'DO_EC02_esccol_C3_l05',
        'desc': 'Escape collapse (Ca=1.0, Cr=3.0, lr=0.5)',
        'Ca': 1.0, 'Cr': 3.0, 'lr_morse': 0.5,
    },
    {
        'name': 'LFIX_ES01',
        'base': 'DO_ES01_escsym_C3_l09',
        'desc': 'Escape symmetric (Ca=1.0, Cr=3.0, lr=0.9)',
        'Ca': 1.0, 'Cr': 3.0, 'lr_morse': 0.9,
    },
    {
        'name': 'LFIX_DM01',
        'base': 'DO_DM01_dmill_C09_l05',
        'desc': 'Double mill (Ca=1.0, Cr=0.9, lr=0.5)',
        'Ca': 1.0, 'Cr': 0.9, 'lr_morse': 0.5,
    },
    {
        'name': 'LFIX_SM01',
        'base': 'DO_SM01_mill_C05_l01',
        'desc': 'Single mill (Ca=1.0, Cr=0.5, lr=0.1)',
        'Ca': 1.0, 'Cr': 0.5, 'lr_morse': 0.1,
    },
]


def generate_config(exp):
    cfg = copy.deepcopy(BASE_TEMPLATE)
    cfg['forces']['params']['Ca'] = exp['Ca']
    cfg['forces']['params']['Cr'] = exp['Cr']
    cfg['forces']['params']['lr'] = exp['lr_morse']
    cfg['experiment_name'] = exp['name']
    return cfg


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'configs', 'lstm_fix')
    os.makedirs(out_dir, exist_ok=True)

    for exp in EXPERIMENTS:
        cfg = generate_config(exp)
        path = os.path.join(out_dir, f"{exp['name']}.yaml")
        with open(path, 'w') as f:
            f.write(f"---\n# {exp['name']}: {exp['desc']}\n")
            f.write(f"# Based on {exp['base']} with LST4 LSTM arch + sqrt/simplex\n")
            f.write(f"# Auto-generated by scripts/create_lstm_fix_configs.py\n\n")
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"Created: {path}")


if __name__ == '__main__':
    main()
