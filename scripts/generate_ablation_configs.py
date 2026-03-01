#!/usr/bin/env python3
"""Generate all 8 ablation configs for the preprocessing ablation suite.

2³ factorial design:
  Factor 1: density_transform  ∈ {raw, sqrt}
  Factor 2: mass_postprocess   ∈ {none, simplex}
  Factor 3: shift_align        ∈ {false, true}

Base parameters from CUR regime:
  N=200, speed=4.0, Ca=1.5, Cr=0.5, R=4.0, η=0.2
  d=19, MVAR(p=5, α=1e-4), H=300 (~36.6s forecast)
"""
import itertools
from pathlib import Path

CONFIGS_DIR = Path("configs")

# Base template
TEMPLATE = """---
# ============================================================================
# {title}
# ============================================================================
# Preprocessing ablation: sqrt={sqrt_label}, simplex={simplex_label}, align={align_label}
# Base: N=200, Ca=1.5, Cr=0.5, speed=4.0, R=4.0, eta=0.2
# ROM: d=19, MVAR(p=5, alpha=1e-4), H=300 (~36.6s forecast)
# ============================================================================

experiment_name: "{exp_name}"

sim:
  N: 200
  T: 20.0
  dt: 0.04
  Lx: 25.0
  Ly: 25.0
  bc: "periodic"

test_sim:
  T: 36.6

model:
  type: "discrete"
  speed: 4.0
  speed_mode: "constant_with_forces"

params:
  R: 4.0

noise:
  kind: "gaussian"
  eta: 0.2
  match_variance: true

forces:
  enabled: true
  type: "morse"
  params:
    Ca: 1.5
    Cr: 0.5
    la: 1.5
    lr: 0.5
    mu_t: 0.3
    rcut_factor: 5.0

alignment:
  enabled: true

density:
  nx: 64
  ny: 64
  bandwidth: 5.0

outputs:
  density_resolution: 64
  density_bandwidth: 5.0

train_ic:
  type: "mixed_comprehensive"
  gaussian:
    enabled: true
    n_runs: 80
    positions_x: [5.0, 10.0, 15.0, 20.0]
    positions_y: [5.0, 10.0, 15.0, 20.0]
    variances: [1.5, 3.0, 5.0]
    n_samples_per_config: 1
  uniform:
    enabled: true
    n_runs: 40
    n_samples: 40
  two_clusters:
    enabled: true
    n_runs: 24
    separations: [4.0, 7.0, 10.0]
    sigmas: [1.5, 3.0]
    n_samples_per_config: 4
  ring:
    enabled: false

test_ic:
  type: "mixed_test_comprehensive"
  gaussian:
    enabled: false
  uniform:
    enabled: true
    n_runs: 20
  two_clusters:
    enabled: false
  ring:
    enabled: false

rom:
  subsample: 3
  fixed_modes: 19
  density_transform: "{density_transform}"
  density_transform_eps: 1.0e-10
  shift_align: {shift_align}
  shift_align_ref: "mean"
  mass_postprocess: "{mass_postprocess}"

  models:
    mvar:
      enabled: true
      lag: 5
      ridge_alpha: 1.0e-4
    lstm:
      enabled: false

eval:
  metrics: ["r2", "rmse"]
  save_forecasts: true
  save_time_resolved: true
  forecast_start: 0.60
  clamp_mode: "C0"
"""

# Factor levels
factors = {
    'sqrt':    [False, True],
    'simplex': [False, True],
    'align':   [False, True],
}

configs = []
for i, (use_sqrt, use_simplex, use_align) in enumerate(
    itertools.product(factors['sqrt'], factors['simplex'], factors['align']), 1
):
    # Build labels
    sqrt_label = "sqrt" if use_sqrt else "raw"
    simplex_label = "simplex" if use_simplex else "none"
    align_label = "align" if use_align else "noAlign"

    short = f"ABL{i}"
    exp_name = f"ABL{i}_N200_{sqrt_label}_{simplex_label}_{align_label}_H300"
    title = f"ABL{i} — Ablation: {sqrt_label}, {simplex_label}, {align_label}"

    content = TEMPLATE.format(
        title=title,
        sqrt_label=sqrt_label,
        simplex_label=simplex_label,
        align_label=align_label,
        exp_name=exp_name,
        density_transform="sqrt" if use_sqrt else "raw",
        shift_align="true" if use_align else "false",
        mass_postprocess="simplex" if use_simplex else "none",
    )

    filename = f"{exp_name}.yaml"
    filepath = CONFIGS_DIR / filename
    filepath.write_text(content)
    configs.append((i, exp_name, filename, use_sqrt, use_simplex, use_align))
    print(f"  Created: {filename}")

print(f"\n  Total: {len(configs)} configs")
print("\n  Ablation matrix:")
print(f"  {'Exp':>5s}  {'sqrt':>5s}  {'simplex':>8s}  {'align':>6s}  Name")
print(f"  {'─'*60}")
for i, name, fn, sq, si, al in configs:
    print(f"  ABL{i:d}  {'  ✓' if sq else '  ✗'}  {'     ✓' if si else '     ✗'}  {'   ✓' if al else '   ✗'}  {name}")
