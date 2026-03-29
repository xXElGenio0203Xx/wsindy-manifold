#!/usr/bin/env python3
"""
Generate VFIX (Variable-speed FIX) LSTM configs — the "everything that works" suite.

Key insight from 56 prior experiments:
  - Removing residual (Δy) improved rollout R² by +53 points on average
  - Raw + C2 beat sqrt+simplex+C0 in 6/7 regimes for R²
  - LayerNorm ON helps vs OFF
  - Multistep loss and scheduled sampling hurt or didn't help
  - 1-step R² is ~0.95 everywhere — the bottleneck is autoregressive rollout

Design: 8 variants × 7 regimes = 56 jobs.
  ALL variants use: residual=OFF, LayerNorm=ON, dropout=0.0

  Suite A — Width sweep (L=1, raw+C2):
    F16:   Nh=16,  L=1  (~2,723 params)  — Alvarez-scale
    F32:   Nh=32,  L=1  (~7,475 params)  — B1 sweet spot ★ REF
    F64:   Nh=64,  L=1  (~23,123 params) — wider
    F128:  Nh=128, L=1  (~78,995 params) — wide (half original VDYN)

  Suite B — Depth sweep (raw+C2):
    F32x2: Nh=32,  L=2  (~15,923 params) — deeper
    F64x2: Nh=64,  L=2  (~56,403 params) — deep + wide

  Suite C — Trick / transform ablation at Nh=32, L=1:
    FSS:   raw+C2  + scheduled_sampling=ON  — test if SS helps no-residual
    FSQRT: sqrt+simplex+C0 + SS=OFF         — transform control

Usage:
    python scripts/generate_vfix_lstm_configs.py
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

# ─── Variant configurations ─────────────────────────────────────────────
# ALL variants: residual=OFF, LayerNorm=ON, dropout=0.0
VARIANTS = [
    # Suite A: Width sweep (L=1, raw+C2)
    {
        "suite": "A", "id": "F16", "suffix": "Nh16_L1",
        "desc": "Alvarez-scale: Nh=16, L=1 (~2,723 params)",
        "hidden_units": 16, "num_layers": 1,
        "density_transform": "raw", "mass_postprocess": "none", "clamp_mode": "C2",
        "scheduled_sampling": False,
    },
    {
        "suite": "A", "id": "F32", "suffix": "Nh32_L1",
        "desc": "Sweet spot: Nh=32, L=1 (~7,475 params) ★ REF",
        "hidden_units": 32, "num_layers": 1,
        "density_transform": "raw", "mass_postprocess": "none", "clamp_mode": "C2",
        "scheduled_sampling": False,
    },
    {
        "suite": "A", "id": "F64", "suffix": "Nh64_L1",
        "desc": "Wider: Nh=64, L=1 (~23,123 params)",
        "hidden_units": 64, "num_layers": 1,
        "density_transform": "raw", "mass_postprocess": "none", "clamp_mode": "C2",
        "scheduled_sampling": False,
    },
    {
        "suite": "A", "id": "F128", "suffix": "Nh128_L1",
        "desc": "Wide: Nh=128, L=1 (~78,995 params)",
        "hidden_units": 128, "num_layers": 1,
        "density_transform": "raw", "mass_postprocess": "none", "clamp_mode": "C2",
        "scheduled_sampling": False,
    },
    # Suite B: Depth sweep (raw+C2)
    {
        "suite": "B", "id": "F32x2", "suffix": "Nh32_L2",
        "desc": "Deeper: Nh=32, L=2 (~15,923 params)",
        "hidden_units": 32, "num_layers": 2,
        "density_transform": "raw", "mass_postprocess": "none", "clamp_mode": "C2",
        "scheduled_sampling": False,
    },
    {
        "suite": "B", "id": "F64x2", "suffix": "Nh64_L2",
        "desc": "Deep+wide: Nh=64, L=2 (~56,403 params)",
        "hidden_units": 64, "num_layers": 2,
        "density_transform": "raw", "mass_postprocess": "none", "clamp_mode": "C2",
        "scheduled_sampling": False,
    },
    # Suite C: Trick / transform ablation at Nh=32, L=1
    {
        "suite": "C", "id": "FSS", "suffix": "Nh32_SS",
        "desc": "Scheduled sampling ON: test if SS helps no-residual (Nh=32, L=1)",
        "hidden_units": 32, "num_layers": 1,
        "density_transform": "raw", "mass_postprocess": "none", "clamp_mode": "C2",
        "scheduled_sampling": True,
    },
    {
        "suite": "C", "id": "FSQRT", "suffix": "Nh32_sqrt",
        "desc": "Transform control: sqrt+simplex+C0, no residual (Nh=32, L=1)",
        "hidden_units": 32, "num_layers": 1,
        "density_transform": "sqrt", "mass_postprocess": "simplex", "clamp_mode": "C0",
        "scheduled_sampling": False,
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


def generate_config(regime, variant):
    num = regime["num"]
    tag = regime["tag"]
    label = regime["label"]
    vid = variant["id"]
    exp_name = f"VFIX_{vid}_{num}_{tag}"

    ss_str = "true" if variant["scheduled_sampling"] else "false"
    dt = variant["density_transform"]
    mp = variant["mass_postprocess"]
    cm = variant["clamp_mode"]

    # Scheduled sampling block (only when enabled)
    if variant["scheduled_sampling"]:
        ss_block = """      scheduled_sampling: true
      ss_warmup: 20
      ss_phase1_end: 200
      ss_phase1_ratio: 0.3
      ss_phase2_end: 400
      ss_max_ratio: 0.5"""
    else:
        ss_block = "      scheduled_sampling: false"

    suite_labels = {
        "A": "Width sweep (L=1, raw+C2)",
        "B": "Depth sweep (raw+C2)",
        "C": "Trick / transform ablation at Nh=32, L=1",
    }

    yaml = f"""---
# ============================================================================
# VFIX_{vid}_{num} — {label} + VARIABLE SPEED — LSTM FIX SUITE
# ============================================================================
# Regime: {regime['desc']}
# Variant {vid}: {variant['desc']}
# Suite {variant['suite']}: {suite_labels[variant['suite']]}
#
# ALL VFIX: residual=OFF (key finding from 56-experiment ablation),
#           LayerNorm=ON, dropout=0.0, multistep=OFF.
#           Transform: {dt} + {mp} + {cm}.
# LSTM-only (MVAR disabled).
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
  density_transform: "{dt}"
  density_transform_eps: 1.0e-10
  shift_align: true
  shift_align_ref: "mean"
  mass_postprocess: "{mp}"
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
      residual: false
      use_layer_norm: true
      normalize_input: true
      patience: 40
      multistep_loss: false
{ss_block}

eval:
  metrics: ["r2", "rmse"]
  save_forecasts: true
  save_time_resolved: true
  forecast_start: 0.60
  clamp_mode: "{cm}"
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

    print(f"\n{'='*70}")
    print(f"  VFIX — LSTM Fix Suite: {len(created)} configs generated")
    print(f"  ALL variants: residual=OFF, LayerNorm=ON, dropout=0, multistep=OFF")
    print(f"{'='*70}")

    suite_labels = {"A": "Width sweep", "B": "Depth sweep", "C": "Trick/transform"}
    for suite_id in ["A", "B", "C"]:
        suite_items = [(n, v) for n, _, v in created if v['suite'] == suite_id]
        if suite_items:
            print(f"\n  Suite {suite_id} ({suite_labels[suite_id]}): {len(suite_items)} configs")
            vids_seen = []
            for n, v in suite_items:
                if v['id'] not in vids_seen:
                    vids_seen.append(v['id'])
                    count = sum(1 for n2, v2 in suite_items if v2['id'] == v['id'])
                    dt = v['density_transform']
                    mp = v['mass_postprocess']
                    cm = v.get('clamp_mode', 'C2')
                    ss = "SS=ON" if v['scheduled_sampling'] else "SS=OFF"
                    print(f"    {v['id']:6s}: Nh={v['hidden_units']}, L={v['num_layers']}, "
                          f"{dt}+{mp}+{cm}, {ss}  ({count} regimes)")

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
                                  residual=False, use_layer_norm=True)
                n = sum(p.numel() for p in m.parameters())
                print(f"    Nh={v['hidden_units']:>3}, L={v['num_layers']}  =>  {n:>8,} params")
    except ImportError:
        print("    (torch not available, skipping param counts)")
