#!/usr/bin/env python3
"""
Generate VARCH (Variable-speed ARCHitecture ablation) LSTM configs.

Thesis-friendly ablation suite — 6 configurations × 7 regimes = 42 jobs.

Suite A — Capacity sweep (same training tricks):
  A16:   Nh=16, L=1  (Alvarez-spirit, ~2,723 params)
  A32:   Nh=32, L=1  (modest scale-up, ~7,475 params)
  A32x2: Nh=32, L=2  (reasonable depth, ~15,923 params)
  All with: residual=true, layer_norm=true, multistep=false, SS=false

Suite B — Training-trick sweep at Nh=32, L=1:
  B1: residual OFF      (is Δy formulation needed?)
  B2: layer_norm OFF    (is LN needed for small models?)
  B3: multistep ON      (does rollout loss help once capacity is sane?)

All configs use sqrt + simplex + C0 (matching VLST_A suite) and LSTM-only.

Usage:
    python scripts/generate_varch_lstm_configs.py
"""
import os

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")

# ─── Per-regime parameters (same 7 VDYN regimes) ────────────────────────
REGIMES = [
    {
        "num": 1, "tag": "gentle", "label": "GENTLE",
        "desc": "Low speed, balanced forces, low noise. Easy regime.",
        "speed": 1.5, "eta": 0.1,
        "forces": True, "Ca": 0.8, "Cr": 0.3,
    },
    {
        "num": 2, "tag": "hypervelocity", "label": "HYPER VELOCITY",
        "desc": "speed=15 base + force-driven. Agents zip across domain.",
        "speed": 15.0, "eta": 0.2,
        "forces": True, "Ca": 1.5, "Cr": 0.5,
    },
    {
        "num": 3, "tag": "hypernoisy", "label": "HYPER NOISY",
        "desc": "Strong noise eta=1.5. Near-random heading perturbation.",
        "speed": 4.0, "eta": 1.5,
        "forces": True, "Ca": 1.5, "Cr": 0.5,
    },
    {
        "num": 4, "tag": "blackhole", "label": "BLACK HOLE",
        "desc": "Extreme attraction Ca=12, minimal repulsion Cr=0.1.",
        "speed": 4.0, "eta": 0.2,
        "forces": True, "Ca": 12.0, "Cr": 0.1,
    },
    {
        "num": 5, "tag": "supernova", "label": "SUPERNOVA",
        "desc": "Extreme repulsion Cr=10, minimal attraction Ca=0.1.",
        "speed": 4.0, "eta": 0.2,
        "forces": True, "Ca": 0.1, "Cr": 10.0,
    },
    {
        "num": 6, "tag": "baseline", "label": "BASELINE",
        "desc": "Balanced Vicsek + Morse. Standard variable-speed reference.",
        "speed": 4.0, "eta": 0.2,
        "forces": True, "Ca": 1.5, "Cr": 0.5,
    },
    {
        "num": 7, "tag": "pure_vicsek", "label": "PURE VICSEK",
        "desc": "No Morse forces — pure alignment + variable speed.",
        "speed": 4.0, "eta": 0.2,
        "forces": False,
    },
]

# ─── Architecture / trick configurations ────────────────────────────────
VARIANTS = [
    # Suite A: Capacity sweep
    {
        "suite": "A", "id": "A16", "suffix": "Nh16_L1",
        "desc": "Alvarez-spirit: Nh=16, L=1 (~2,723 params)",
        "hidden_units": 16, "num_layers": 1,
        "residual": True, "use_layer_norm": True,
        "multistep_loss": False, "scheduled_sampling": False,
    },
    {
        "suite": "A", "id": "A32", "suffix": "Nh32_L1",
        "desc": "Modest scale-up: Nh=32, L=1 (~7,475 params)",
        "hidden_units": 32, "num_layers": 1,
        "residual": True, "use_layer_norm": True,
        "multistep_loss": False, "scheduled_sampling": False,
    },
    {
        "suite": "A", "id": "A32x2", "suffix": "Nh32_L2",
        "desc": "Reasonable depth: Nh=32, L=2 (~15,923 params)",
        "hidden_units": 32, "num_layers": 2,
        "residual": True, "use_layer_norm": True,
        "multistep_loss": False, "scheduled_sampling": False,
    },
    # Suite B: Training-trick sweep at Nh=32, L=1
    {
        "suite": "B", "id": "B1", "suffix": "Nh32_noRes",
        "desc": "No residual: residual=false, LN on (Nh=32, L=1)",
        "hidden_units": 32, "num_layers": 1,
        "residual": False, "use_layer_norm": True,
        "multistep_loss": False, "scheduled_sampling": False,
    },
    {
        "suite": "B", "id": "B2", "suffix": "Nh32_noLN",
        "desc": "No LayerNorm: use_layer_norm=false, residual on (Nh=32, L=1)",
        "hidden_units": 32, "num_layers": 1,
        "residual": True, "use_layer_norm": False,
        "multistep_loss": False, "scheduled_sampling": False,
    },
    {
        "suite": "B", "id": "B3", "suffix": "Nh32_multistep",
        "desc": "Multistep loss: k=5, alpha=0.3, no SS (Nh=32, L=1)",
        "hidden_units": 32, "num_layers": 1,
        "residual": True, "use_layer_norm": True,
        "multistep_loss": True, "scheduled_sampling": False,
    },
]


def make_forces_block(regime):
    if not regime["forces"]:
        return """forces:
  enabled: false"""
    return f"""forces:
  enabled: true
  type: "morse"
  params:
    Ca: {regime['Ca']}
    Cr: {regime['Cr']}
    la: 1.5
    lr: 0.5
    mu_t: 0.3
    rcut_factor: 5.0"""


def make_multistep_block(variant):
    if variant["multistep_loss"]:
        return """      multistep_loss: true
      multistep_k: 5
      multistep_alpha: 0.3"""
    else:
        return "      multistep_loss: false"


def generate_config(regime, variant):
    num = regime["num"]
    tag = regime["tag"]
    label = regime["label"]
    vid = variant["id"]
    exp_name = f"VARCH_{vid}_{num}_{tag}"

    residual_str = "true" if variant["residual"] else "false"
    ln_str = "true" if variant["use_layer_norm"] else "false"
    ss_str = "true" if variant["scheduled_sampling"] else "false"

    yaml = f"""---
# ============================================================================
# VARCH_{vid}_{num} — {label} + VARIABLE SPEED — LSTM ARCHITECTURE ABLATION
# ============================================================================
# Regime: {regime['desc']}
# Variant {vid}: {variant['desc']}
# Suite {variant['suite']}: {'Capacity sweep' if variant['suite'] == 'A' else 'Training-trick sweep at Nh=32, L=1'}
# LSTM-only (MVAR disabled). sqrt + simplex + C0.
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
  T: 50.0

model:
  type: "discrete"
  speed: {regime['speed']}
  speed_mode: "variable"

params:
  R: 4.0

noise:
  kind: "gaussian"
  eta: {regime['eta']}
  match_variance: true

{make_forces_block(regime)}

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
  density_transform: "sqrt"
  density_transform_eps: 1.0e-10
  shift_align: true
  shift_align_ref: "mean"
  mass_postprocess: "simplex"
  models:
    mvar:
      enabled: false
    lstm:
      enabled: true
      hidden_units: {variant['hidden_units']}
      num_layers: {variant['num_layers']}
      max_epochs: 200
      learning_rate: 0.001
      batch_size: 32
      lag: 5
      dropout: 0.0
      gradient_clip: 5.0
      residual: {residual_str}
      use_layer_norm: {ln_str}
      normalize_input: true
      patience: 40
{make_multistep_block(variant)}
      scheduled_sampling: {ss_str}

eval:
  metrics: ["r2", "rmse"]
  save_forecasts: true
  save_time_resolved: true
  forecast_start: 0.60
  clamp_mode: "C0"
"""
    return exp_name, yaml


if __name__ == "__main__":
    created = []
    for variant in VARIANTS:
        for regime in REGIMES:
            exp_name, yaml_content = generate_config(regime, variant)
            path = os.path.join(CONFIGS_DIR, f"{exp_name}.yaml")
            with open(path, "w") as f:
                f.write(yaml_content)
            created.append((exp_name, path, variant))
            print(f"  Created: {os.path.basename(path)}")

    print(f"\n{'='*60}")
    print(f"  Total: {len(created)} configs generated")
    suite_a = [n for n, _, v in created if v['suite'] == 'A']
    suite_b = [n for n, _, v in created if v['suite'] == 'B']
    print(f"  Suite A (capacity sweep):      {len(suite_a)} configs")
    for vid in ['A16', 'A32', 'A32x2']:
        count = sum(1 for n in suite_a if f'_{vid}_' in n)
        print(f"    {vid}: {count} regimes")
    print(f"  Suite B (trick sweep, Nh=32):  {len(suite_b)} configs")
    for vid in ['B1', 'B2', 'B3']:
        count = sum(1 for n in suite_b if f'_{vid}_' in n)
        print(f"    {vid}: {count} regimes")

    # Print parameter counts
    print(f"\n  Parameter counts (d=19):")
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    try:
        import torch
        from rom.lstm_rom import LatentLSTMROM
        seen = set()
        for v in VARIANTS:
            key = (v['hidden_units'], v['num_layers'])
            if key not in seen:
                seen.add(key)
                m = LatentLSTMROM(d=19, hidden_units=v['hidden_units'],
                                  num_layers=v['num_layers'],
                                  residual=v['residual'],
                                  use_layer_norm=v['use_layer_norm'])
                n = sum(p.numel() for p in m.parameters())
                print(f"    Nh={v['hidden_units']:>3}, L={v['num_layers']}  =>  {n:>8,} params")
    except ImportError:
        print("    (torch not available, skipping param counts)")
