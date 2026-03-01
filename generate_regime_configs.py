#!/usr/bin/env python3
"""
Generate Systematic Experiment Configs
=======================================

Creates YAML configs for the systematic regime comparison:

1. NDYN suite: Vicsek/force regimes covering diverse + extreme behaviors
2. DORSOGNA suite: d'Orsogna model regimes from Bhaskar & Ziegelmeier taxonomy

All configs share identical:
  - N=300 particles, Lx=Ly=25, dt=0.04, T=20s train / T=200s test
  - 64x64 KDE density, bandwidth=5.0
  - 19 POD modes, shift alignment (raw, no sqrt/simplex), save unaligned POD
  - MVAR(p=5, alpha=1e-4) AND LSTM enabled
  - Training ICs: 80 gaussian + 40 uniform + 24 two-cluster + 24 ring
  - Test ICs: 4 tests (1 gaussian, 1 uniform, 1 two-cluster, 1 ring)
  - Same eval settings

Usage:
    python generate_regime_configs.py [--output_dir configs/systematic]
"""

import yaml
from pathlib import Path
import argparse


def base_config():
    """Return the shared base config dict."""
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
            'speed': 3.0,
            'speed_mode': 'constant_with_forces',
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


def write_config(cfg, path):
    """Write config dict to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write('---\n')
        f.write(f'# {cfg.get("_comment", "")}\n')
        f.write(f'# Auto-generated by generate_regime_configs.py\n\n')
        # Remove internal keys
        out = {k: v for k, v in cfg.items() if not k.startswith('_')}
        yaml.dump(out, f, default_flow_style=False, sort_keys=False)
    print(f"  {path.name}")


# ============================================================================
# NDYN SUITE: Each config occupies a distinct corner of behavior space
# ============================================================================

def new_dyn_configs():
    """
    DYN suite: every config produces qualitatively DIFFERENT dynamics.

    No parameter sweeps — each experiment is a distinct physical regime.

    Axes of variation:
      speed × noise × attraction × repulsion × force range × speed coupling

    Selection criteria: at least 2 parameter axes differ between any pair.
    """
    configs = []

    # 01: Crawling deterministic flock — very slow, very quiet, weak forces
    #     → smooth, slowly-evolving density, easy baseline for POD
    configs.append(('NDYN01_crawl', {
        '_comment': 'NDYN01: Slow deterministic flock. Easy POD baseline.',
        'model': {'speed': 0.5},
        'noise': {'eta': 0.03},
        'forces': {'params': {'Ca': 0.8, 'Cr': 0.3, 'la': 1.5, 'lr': 0.5}},
    }))

    # 02: Standard ordered flock — moderate speed, low noise, moderate attraction
    #     → coherent traveling cluster
    configs.append(('NDYN02_flock', {
        '_comment': 'NDYN02: Ordered coherent flock, traveling cluster.',
        'model': {'speed': 3.0},
        'noise': {'eta': 0.1},
        'forces': {'params': {'Ca': 2.0, 'Cr': 0.5, 'la': 1.5, 'lr': 0.5}},
    }))

    # 03: Fast transport — rapid cluster motion, tests shift alignment
    #     → same physics as 02 but density moves across domain much faster
    configs.append(('NDYN03_sprint', {
        '_comment': 'NDYN03: Fast transport, rapid density translation.',
        'model': {'speed': 10.0},
        'noise': {'eta': 0.1},
        'forces': {'params': {'Ca': 2.0, 'Cr': 0.5, 'la': 1.5, 'lr': 0.5}},
    }))

    # 04: Disordered gas — high noise destroys alignment → near-uniform density
    #     → POD should struggle (low-rank structure disappears)
    configs.append(('NDYN04_gas', {
        '_comment': 'NDYN04: Disordered gas, noise destroys all structure.',
        'model': {'speed': 3.0},
        'noise': {'eta': 1.5},
        'forces': {'params': {'Ca': 0.5, 'Cr': 0.2, 'la': 1.5, 'lr': 0.5}},
    }))

    # 05: Black hole — extreme attraction collapses swarm to sharp spike
    #     → challenging for KDE (very narrow peak), sharp singular values
    configs.append(('NDYN05_blackhole', {
        '_comment': 'NDYN05: Extreme attraction, near-singular density spike.',
        'model': {'speed': 4.0},
        'noise': {'eta': 0.15},
        'forces': {'params': {'Ca': 20.0, 'Cr': 0.05, 'la': 1.5, 'lr': 0.3}},
    }))

    # 06: Supernova — extreme repulsion drives explosive dispersal
    #     → density spreads thin, periodic boundary wrapping matters
    configs.append(('NDYN06_supernova', {
        '_comment': 'NDYN06: Explosive repulsion, violent dispersal.',
        'model': {'speed': 3.0},
        'noise': {'eta': 0.15},
        'forces': {'params': {'Ca': 0.05, 'Cr': 15.0, 'la': 1.5, 'lr': 0.5}},
    }))

    # 07: Crystal lattice — balanced attract=repulse, ring/crystal equilibrium
    #     → stable structure, quasi-static density after transient
    configs.append(('NDYN07_crystal', {
        '_comment': 'NDYN07: Balanced forces, crystal/ring equilibrium.',
        'model': {'speed': 2.0},
        'noise': {'eta': 0.1},
        'forces': {'params': {'Ca': 3.0, 'Cr': 3.0, 'la': 2.0, 'lr': 0.5}},
    }))

    # 08: Pure Vicsek — no forces, alignment only
    #     → band/wave patterns, classic order-disorder transition
    configs.append(('NDYN08_pure_vicsek', {
        '_comment': 'NDYN08: Pure Vicsek, alignment only, no forces.',
        'model': {'speed': 3.0},
        'noise': {'eta': 0.3},
        'forces': {'enabled': False},
    }))

    # 09: Long-range soup — forces reach across domain, global coupling
    #     → density modes are collective, smooth large-scale patterns
    configs.append(('NDYN09_longrange', {
        '_comment': 'NDYN09: Long-range forces, global collective modes.',
        'model': {'speed': 2.0},
        'noise': {'eta': 0.2},
        'forces': {'params': {'Ca': 1.5, 'Cr': 0.6, 'la': 6.0, 'lr': 3.0}},
    }))

    # 10: Short-range spikes — very localized sharp forces, clumpy dynamics
    #     → multi-cluster fragmentation, complex topology
    configs.append(('NDYN10_shortrange', {
        '_comment': 'NDYN10: Short-range spike forces, fragmented clusters.',
        'model': {'speed': 3.0},
        'noise': {'eta': 0.15},
        'forces': {'params': {'Ca': 4.0, 'Cr': 2.0, 'la': 0.3, 'lr': 0.1}},
    }))

    # 11: Noisy collapse — strong attraction fights high noise
    #     → intermittent clustering, stochastic density bursts
    configs.append(('NDYN11_noisy_collapse', {
        '_comment': 'NDYN11: Strong attraction vs high noise, intermittent clusters.',
        'model': {'speed': 3.0},
        'noise': {'eta': 0.8},
        'forces': {'params': {'Ca': 12.0, 'Cr': 0.1, 'la': 1.5, 'lr': 0.5}},
    }))

    # 12: Fast explosion — high speed + strong repulsion
    #     → rapid dispersal, tests long-horizon degradation
    configs.append(('NDYN12_fast_explosion', {
        '_comment': 'NDYN12: Fast speed + strong repulsion, rapid dispersal.',
        'model': {'speed': 8.0},
        'noise': {'eta': 0.2},
        'forces': {'params': {'Ca': 0.1, 'Cr': 10.0, 'la': 1.5, 'lr': 0.5}},
    }))

    # 13: Hyperfast noisy — extreme speed + extreme noise, hardest case
    #     → near-unpredictable, lower bound on ROM performance
    configs.append(('NDYN13_chaos', {
        '_comment': 'NDYN13: Extreme speed + noise, ROM performance lower bound.',
        'model': {'speed': 12.0},
        'noise': {'eta': 1.2},
        'forces': {'params': {'Ca': 1.0, 'Cr': 0.5, 'la': 1.5, 'lr': 0.5}},
    }))

    # 14: Variable speed baseline — standard flock but speed coupled to force
    #     → non-constant flux, different transport structure
    configs.append(('NDYN14_varspeed', {
        '_comment': 'NDYN14: Variable speed coupled to forces, non-constant flux.',
        'model': {'speed': 3.0, 'speed_mode': 'variable'},
        'noise': {'eta': 0.2},
        'forces': {'params': {'Ca': 2.0, 'Cr': 0.8, 'la': 1.5, 'lr': 0.5}},
    }))

    return configs


# ============================================================================
# D'ORSOGNA SUITE: Regimes from Bhaskar & Ziegelmeier taxonomy
# ============================================================================

def dorsogna_configs():
    """
    d'Orsogna model regimes parametrized by (C, l) = (Cr/Ca, lr/la).

    We fix Ca=1.0 and la=1.0 as reference scale, then:
        Cr = C * Ca = C
        lr = l * la = l

    Pruned to one or two representatives per distinct regime region,
    avoiding near-duplicate dynamics. Selection picks points that are
    maximally separated within each regime boundary.

    Regime classification (from Bhaskar & Ziegelmeier):
        Single mill (2):    (0.5,0.1), (3,0.1) — weak vs strong C, both short-range repulse
        Single mill+:       (2,0.5) — unique: strong C with medium range
        Double mill (1):    (0.9,0.5) — only known point
        Double ring (2):    (0.1,0.1), (0.9,0.9) — extremes of the diagonal
        Collective swarm(3):(0.1,0.5), (0.5,3), (0.9,3) — low/med/high C with long range
        Escape sym. (1):    (3,0.9) — strongest symmetric escape
        Escape unsym.(2):   (2,2), (3,3) — corners of the region
        Escape coll. (2):   (2,3), (3,0.5) — both points (very different l)
    """
    Ca_ref = 1.0
    la_ref = 1.0

    regimes = [
        # Single mill — rotational vortex
        ('DO_SM01_mill_C05_l01',   'Single mill',          0.5, 0.1),
        ('DO_SM02_mill_C3_l01',    'Single mill',          3.0, 0.1),
        ('DO_SM03_mill_C2_l05',    'Single mill',          2.0, 0.5),
        # Double mill — counter-rotating double vortex
        ('DO_DM01_dmill_C09_l05',  'Double mill',          0.9, 0.5),
        # Double ring — nested annular structures
        ('DO_DR01_dring_C01_l01',  'Double ring',          0.1, 0.1),
        ('DO_DR02_dring_C09_l09',  'Double ring',          0.9, 0.9),
        # Collective swarm — cohesive traveling group
        ('DO_CS01_swarm_C01_l05',  'Collective swarm',     0.1, 0.5),
        ('DO_CS02_swarm_C05_l3',   'Collective swarm',     0.5, 3.0),
        ('DO_CS03_swarm_C09_l3',   'Collective swarm',     0.9, 3.0),
        # Escape (symmetric) — particles flee symmetrically
        ('DO_ES01_escsym_C3_l09',  'Escape (symmetric)',   3.0, 0.9),
        # Escape (unsymmetric) — asymmetric dispersal
        ('DO_EU01_escuns_C2_l2',   'Escape (unsymmetric)', 2.0, 2.0),
        ('DO_EU02_escuns_C3_l3',   'Escape (unsymmetric)', 3.0, 3.0),
        # Escape (collective) — group escape
        ('DO_EC01_esccol_C2_l3',   'Escape (collective)',  2.0, 3.0),
        ('DO_EC02_esccol_C3_l05',  'Escape (collective)',  3.0, 0.5),
    ]

    configs = []
    for name, regime_label, C_ratio, l_ratio in regimes:
        Cr = C_ratio * Ca_ref
        lr = l_ratio * la_ref
        configs.append((name, {
            '_comment': f'{name}: {regime_label} (C={C_ratio}, l={l_ratio})',
            'model': {'speed': 3.0},
            'noise': {'eta': 0.2},
            'forces': {'params': {
                'Ca': Ca_ref,
                'Cr': Cr,
                'la': la_ref,
                'lr': lr,
            }},
        }))

    return configs


import copy


def duplicate_with_varspeed(configs):
    """
    Given a list of (name, overrides) configs, return the original list
    PLUS a '_VS' variant for each one where speed_mode='variable'.

    Skips configs that already have speed_mode='variable' or forces disabled.
    """
    out = list(configs)
    for name, overrides in configs:
        # Skip if already variable speed
        if overrides.get('model', {}).get('speed_mode') == 'variable':
            continue
        # Skip if forces disabled
        if overrides.get('forces', {}).get('enabled') is False:
            continue

        vs_overrides = copy.deepcopy(overrides)
        vs_name = name + '_VS'
        orig_comment = vs_overrides.get('_comment', name)
        vs_overrides['_comment'] = orig_comment + ' [variable speed]'
        vs_overrides.setdefault('model', {})['speed_mode'] = 'variable'
        out.append((vs_name, vs_overrides))
    return out


def main():
    parser = argparse.ArgumentParser(description='Generate systematic experiment configs')
    parser.add_argument('--output_dir', type=str, default='configs/systematic',
                        help='Output directory for generated configs')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Systematic Regime Configs")
    print("=" * 60)

    # NDYN suite (constant speed) + variable-speed duplicates
    ndyn_all = duplicate_with_varspeed(new_dyn_configs())
    print(f"\nNDYN suite ({len(ndyn_all)} configs: constant + variable speed):")
    for name, overrides in ndyn_all:
        cfg = base_config()
        cfg['experiment_name'] = name
        deep_update(cfg, overrides)
        write_config(cfg, output_dir / f'{name}.yaml')

    # d'Orsogna suite (constant speed) + variable-speed duplicates
    do_all = duplicate_with_varspeed(dorsogna_configs())
    print(f"\nd'Orsogna suite ({len(do_all)} configs: constant + variable speed):")
    for name, overrides in do_all:
        cfg = base_config()
        cfg['experiment_name'] = name
        deep_update(cfg, overrides)
        write_config(cfg, output_dir / f'{name}.yaml')

    total = len(ndyn_all) + len(do_all)
    print(f"\nGenerated {total} configs in {output_dir}/")
    print("Done.")


if __name__ == '__main__':
    main()
