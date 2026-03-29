#!/usr/bin/env python3
"""
Generate XABL (Extended Ablation) config files.

Tests all combinations of:
  - alignment: always ON (the interesting case per user request)
  - density_transform: raw vs sqrt  (2)
  - mass_postprocess: none vs simplex (2)
  - eigenvalue_threshold: None vs 1.0 (2)
  
= 8 configs total (XABL1-XABL8)

Naming: XABL{n}_align_{transform}_{postprocess}_{scaling}
"""

import yaml
from pathlib import Path

# Base config (same physics/ICs as ABL suite)
BASE = {
    'sim': {
        'N': 200,
        'T': 20.0,
        'dt': 0.04,
        'Lx': 25.0,
        'Ly': 25.0,
        'bc': 'periodic',
    },
    'test_sim': {'T': 36.6},
    'model': {
        'type': 'discrete',
        'speed': 4.0,
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
            'Ca': 1.5,
            'Cr': 0.5,
            'la': 1.5,
            'lr': 0.5,
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
        'ring': {'enabled': False},
    },
    'test_ic': {
        'type': 'mixed_test_comprehensive',
        'gaussian': {'enabled': False},
        'uniform': {'enabled': True, 'n_runs': 20},
        'two_clusters': {'enabled': False},
        'ring': {'enabled': False},
    },
    'eval': {
        'metrics': ['r2', 'rmse'],
        'save_forecasts': True,
        'save_time_resolved': True,
        'forecast_start': 0.60,
        'clamp_mode': 'C0',
    },
}

# Experiment grid: all combinations with alignment=ON
#   transform:    raw | sqrt
#   postprocess:  none | simplex
#   scaling:      none | 1.0 (eigenvalue_threshold)
combos = [
    # (n, transform, postprocess, eigenvalue_threshold, short_label)
    (1, 'raw',  'none',    None, 'raw_none_noScale'),
    (2, 'raw',  'none',    1.0,  'raw_none_scale1'),
    (3, 'raw',  'simplex', None, 'raw_simplex_noScale'),
    (4, 'raw',  'simplex', 1.0,  'raw_simplex_scale1'),
    (5, 'sqrt', 'none',    None, 'sqrt_none_noScale'),
    (6, 'sqrt', 'none',    1.0,  'sqrt_none_scale1'),
    (7, 'sqrt', 'simplex', None, 'sqrt_simplex_noScale'),
    (8, 'sqrt', 'simplex', 1.0,  'sqrt_simplex_scale1'),
]

configs_dir = Path('configs')

for n, transform, postprocess, eig_thresh, short_label in combos:
    name = f"XABL{n}_align_{short_label}"
    
    cfg = yaml.safe_load(yaml.dump(BASE))  # deep copy
    cfg['experiment_name'] = name
    
    # ROM config
    rom = {
        'subsample': 3,
        'fixed_modes': 19,
        'density_transform': transform,
        'density_transform_eps': 1.0e-10,
        'shift_align': True,
        'shift_align_ref': 'mean',
        'mass_postprocess': postprocess,
        'models': {
            'mvar': {
                'enabled': True,
                'lag': 5,
                'ridge_alpha': 1.0e-4,
            },
            'lstm': {'enabled': False},
        },
    }
    
    if eig_thresh is not None:
        rom['models']['mvar']['eigenvalue_threshold'] = eig_thresh
    
    cfg['rom'] = rom
    
    # Header comment
    scaling_str = f"scale to {eig_thresh}" if eig_thresh else "no scaling"
    header = f"""---
# ============================================================================
# {name}
# ============================================================================
# Extended ablation: alignment=ON, transform={transform}, postprocess={postprocess}, {scaling_str}
# Base: N=200, Ca=1.5, Cr=0.5, speed=4.0, R=4.0, eta=0.2
# ROM: d=19, MVAR(p=5, alpha=1e-4), H=300 (~36.6s forecast)
# ============================================================================
"""
    
    out_path = configs_dir / f"{name}.yaml"
    with open(out_path, 'w') as f:
        f.write(header)
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    
    print(f"  ✓ {name}.yaml  (transform={transform}, postprocess={postprocess}, scaling={eig_thresh})")

print(f"\nGenerated {len(combos)} configs in {configs_dir}/")
