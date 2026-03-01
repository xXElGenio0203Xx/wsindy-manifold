#!/usr/bin/env python3
"""
Generate VDYN Clamp-Mode Comparison Configs
============================================

Creates 14 YAML configs: 7 VDYN dynamics × 2 clamp modes (C0 vs C2).

Purpose: determine whether clamp+renormalize (C2) improves forecasts over
         no clamping (C0) before running the full 54-config systematic sweep.

All configs use the updated systematic base template:
  - N=300, T_train=20s, T_test=200s, dt=0.04
  - 168 training ICs (80 gaussian + 40 uniform + 24 two-cluster + 24 ring)
  - 4 test ICs (1 gaussian + 1 uniform + 1 two-cluster + 1 ring)
  - raw density transform, shift alignment, save_unaligned_pod
  - MVAR(lag=5, alpha=1e-4) + LSTM
  - speed_mode = "variable" for all VDYN

Usage:
    python generate_vdyn_clamp_configs.py [--output_dir configs/vdyn_clamp]
"""

import yaml
from pathlib import Path
import argparse


def base_config():
    """Shared base config — identical to systematic template."""
    return {
        'sim': {
            'N': 300,
            'T': 20.0,
            'dt': 0.04,
            'Lx': 25.0,
            'Ly': 25.0,
            'bc': 'periodic',
        },
        'test_sim': {
            'T': 200.0,
        },
        'model': {
            'type': 'discrete',
            'speed': 4.0,
            'speed_mode': 'variable',
        },
        'params': {
            'R': 4.0,
        },
        'noise': {
            'kind': 'gaussian',
            'eta': 0.2,
            'match_variance': True,
        },
        'forces': {
            'enabled': True,
            'type': 'morse',
            'params': {
                'Ca': 1.5,
                'Cr': 0.5,
                'la': 1.5,
                'lr': 0.5,
                'mu_t': 0.3,
                'rcut_factor': 5.0,
            },
        },
        'alignment': {
            'enabled': True,
        },
        'density': {
            'nx': 64,
            'ny': 64,
            'bandwidth': 5.0,
        },
        'outputs': {
            'density_resolution': 64,
            'density_bandwidth': 5.0,
        },
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
            'uniform': {
                'enabled': True,
                'n_runs': 40,
                'n_samples': 40,
            },
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
            'uniform': {
                'enabled': True,
                'n_runs': 1,
            },
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
            'density_transform': 'raw',
            'density_transform_eps': 1.0e-10,
            'shift_align': True,
            'shift_align_ref': 'mean',
            'save_unaligned_pod': True,
            'mass_postprocess': 'none',
            'models': {
                'mvar': {
                    'enabled': True,
                    'lag': 5,
                    'ridge_alpha': 1.0e-4,
                },
                'lstm': {
                    'enabled': True,
                    'lag': 20,
                    'hidden_units': 128,
                    'num_layers': 2,
                    'dropout': 0.1,
                    'residual': True,
                    'use_layer_norm': True,
                    'max_epochs': 300,
                    'patience': 30,
                    'lr': 1.0e-3,
                    'batch_size': 256,
                    'grad_clip': 1.0,
                },
            },
        },
        'eval': {
            'metrics': ['r2', 'rmse'],
            'save_forecasts': True,
            'save_time_resolved': True,
            'forecast_start': 0.60,
            'clamp_mode': 'C0',
        },
    }


def deep_update(d, overrides):
    """Recursively update dict d with overrides."""
    for k, v in overrides.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


# ── VDYN dynamics definitions ──────────────────────────────────────────────
VDYN_REGIMES = [
    {
        'id': 'VDYN1', 'tag': 'gentle',
        'comment': 'VDYN1 — Gentle (low speed, balanced forces, low noise)',
        'overrides': {
            'model': {'speed': 1.5},
            'noise': {'eta': 0.1},
            'forces': {'params': {'Ca': 0.8, 'Cr': 0.3}},
        },
    },
    {
        'id': 'VDYN2', 'tag': 'hypervelocity',
        'comment': 'VDYN2 — Hypervelocity (very high speed)',
        'overrides': {
            'model': {'speed': 15.0},
        },
    },
    {
        'id': 'VDYN3', 'tag': 'hypernoisy',
        'comment': 'VDYN3 — Hyper-noisy (very high noise)',
        'overrides': {
            'noise': {'eta': 1.5},
        },
    },
    {
        'id': 'VDYN4', 'tag': 'blackhole',
        'comment': 'VDYN4 — Blackhole (extremely strong attraction)',
        'overrides': {
            'forces': {'params': {'Ca': 12.0, 'Cr': 0.1}},
        },
    },
    {
        'id': 'VDYN5', 'tag': 'supernova',
        'comment': 'VDYN5 — Supernova (extremely strong repulsion)',
        'overrides': {
            'forces': {'params': {'Ca': 0.1, 'Cr': 10.0}},
        },
    },
    {
        'id': 'VDYN6', 'tag': 'baseline',
        'comment': 'VDYN6 — Baseline (moderate everything)',
        'overrides': {},  # pure base config
    },
    {
        'id': 'VDYN7', 'tag': 'pure_vicsek',
        'comment': 'VDYN7 — Pure Vicsek (no Morse forces)',
        'overrides': {
            'forces': {'enabled': False},
        },
    },
]

CLAMP_MODES = ['C0', 'C2']


def write_config(cfg, path):
    """Write config dict to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write('---\n')
        f.write(f'# {cfg.get("_comment", "")}\n')
        f.write('# Auto-generated by generate_vdyn_clamp_configs.py\n\n')
        out = {k: v for k, v in cfg.items() if not k.startswith('_')}
        yaml.dump(out, f, default_flow_style=False, sort_keys=False)


def generate(output_dir: Path):
    """Generate all 14 configs and the manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for regime in VDYN_REGIMES:
        for clamp in CLAMP_MODES:
            name = f"{regime['id']}_{regime['tag']}_{clamp}"
            comment = f"{regime['comment']} — clamp_mode={clamp}"

            cfg = base_config()
            deep_update(cfg, regime['overrides'])
            cfg['experiment_name'] = name
            cfg['eval']['clamp_mode'] = clamp
            cfg['_comment'] = comment

            # For VDYN7 (pure Vicsek), remove force params entirely
            if not cfg['forces'].get('enabled', True):
                cfg['forces'] = {'enabled': False}

            fname = f"{name}.yaml"
            write_config(cfg, output_dir / fname)
            manifest.append(fname)
            print(f"  ✓ {fname}")

    # Write manifest
    manifest_path = output_dir / 'manifest.txt'
    with open(manifest_path, 'w') as f:
        for m in manifest:
            f.write(m + '\n')
    print(f"\n  Manifest: {manifest_path}  ({len(manifest)} configs)")
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='configs/vdyn_clamp',
                        help='Directory for generated configs')
    args = parser.parse_args()
    generate(Path(args.output_dir))
