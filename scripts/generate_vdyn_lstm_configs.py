#!/usr/bin/env python3
"""
Generate VDYN LSTM-only configs for two ablation suites:
  Suite A (VLST_A*): sqrt + simplex + C0  — test if transform prevents divergence
  Suite B (VLST_B*): raw + none + C2      — test if clamping alone prevents divergence
Both suites disable MVAR (LSTM-only to save compute).
"""
import os

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")

# ─── Per-regime parameters ──────────────────────────────────────────────
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

# ─── Suite definitions ──────────────────────────────────────────────────
SUITES = {
    "A": {
        "suffix": "sqrtSimplex_C0",
        "suite_tag": "VLST_A",
        "density_transform": "sqrt",
        "mass_postprocess": "simplex",
        "clamp_mode": "C0",
        "desc_extra": "sqrt transform + simplex projection + C0 clamp",
    },
    "B": {
        "suffix": "raw_C2",
        "suite_tag": "VLST_B",
        "density_transform": "raw",
        "mass_postprocess": "none",
        "clamp_mode": "C2",
        "desc_extra": "raw densities + C2 clamp (clamp + renormalize)",
    },
}


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


def generate_config(regime, suite):
    num = regime["num"]
    tag = regime["tag"]
    label = regime["label"]
    suite_key = suite["suite_tag"]
    exp_name = f"{suite_key}{num}_{tag}_{suite['suffix']}"

    yaml = f"""---
# ============================================================================
# {suite_key}{num} — {label} + VARIABLE SPEED — LSTM ONLY
# ============================================================================
# {regime['desc']}
# Suite {suite_key[-1]}: {suite['desc_extra']}
# LSTM-only (MVAR disabled) to test whether transform/clamping fixes rollout.
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
  density_transform: "{suite['density_transform']}"
  density_transform_eps: 1.0e-10
  shift_align: true
  shift_align_ref: "mean"
  mass_postprocess: "{suite['mass_postprocess']}"
  models:
    mvar:
      enabled: false
    lstm:
      enabled: true
      hidden_units: 128
      num_layers: 2
      max_epochs: 200
      learning_rate: 0.001
      batch_size: 32
      lag: 5
      dropout: 0.15
      gradient_clip: 5.0
      residual: true
      use_layer_norm: true
      normalize_input: true
      patience: 40
      multistep_loss: false
      scheduled_sampling: false

eval:
  metrics: ["r2", "rmse"]
  save_forecasts: true
  save_time_resolved: true
  forecast_start: 0.60
  clamp_mode: "{suite['clamp_mode']}"
"""
    return exp_name, yaml


if __name__ == "__main__":
    created = []
    for suite_key, suite in SUITES.items():
        for regime in REGIMES:
            exp_name, yaml_content = generate_config(regime, suite)
            path = os.path.join(CONFIGS_DIR, f"{exp_name}.yaml")
            with open(path, "w") as f:
                f.write(yaml_content)
            created.append((exp_name, path))
            print(f"  Created: {os.path.basename(path)}")

    print(f"\n  Total: {len(created)} configs generated")
    print(f"  Suite A (sqrt+simplex+C0): {sum(1 for n,_ in created if 'VLST_A' in n)}")
    print(f"  Suite B (raw+C2):          {sum(1 for n,_ in created if 'VLST_B' in n)}")
