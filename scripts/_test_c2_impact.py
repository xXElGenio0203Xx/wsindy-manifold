#!/usr/bin/env python3
"""Quick test: what happens to R² if we apply C2 clamping to ABL2 predictions?

Loads saved POD + MVAR artifacts from ABL2 (raw/none/align) and re-evaluates
with C0 vs C2 clamping to quantify the exact R² impact.
"""
import numpy as np
import pandas as pd
from pathlib import Path

# We don't have saved raw predictions, but we CAN simulate C2's effect:
# C2 = clamp negatives to 0, then rescale to preserve total mass.
#
# Since neg_frac ~ 9.4% of pixels and mass_violation ~ 1e-16,
# the negative pixels have near-zero total mass.
# Let's compute EXACTLY how much R² changes by running the full pipeline
# on one test sim with C0 vs C2.

import sys
sys.path.insert(0, "src")

from rectsim.config_loader import load_config
from rectsim.ROM_pipeline import ROM_pipeline

# Load ABL2 config but override clamp_mode
import yaml

config_path = "configs/ABL2_N200_raw_none_align_H300.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

print("Current clamp_mode:", config.get("eval", {}).get("clamp_mode", "C0"))
print("Current mass_postprocess:", config.get("rom", {}).get("mass_postprocess", "none"))

# Run with C0 (current)
print("\n--- Running with C0 (no clamping) ---")
config["eval"]["clamp_mode"] = "C0"
config["eval"]["n_test_sims"] = 3  # just 3 sims for speed

with open("/tmp/test_c0.yaml", "w") as f:
    yaml.dump(config, f)

result_c0 = load_config("/tmp/test_c0.yaml")
pod_data, metrics_c0 = ROM_pipeline(*result_c0)

# Run with C2
print("\n--- Running with C2 (clamp + renorm) ---")
config["eval"]["clamp_mode"] = "C2"

with open("/tmp/test_c2.yaml", "w") as f:
    yaml.dump(config, f)

result_c2 = load_config("/tmp/test_c2.yaml")
pod_data2, metrics_c2 = ROM_pipeline(*result_c2)

# Compare
print("\n" + "=" * 70)
print("C0 vs C2 COMPARISON (3 test sims)")
print("=" * 70)

for i, (m0, m2) in enumerate(zip(metrics_c0, metrics_c2)):
    r2_c0 = m0.get("r2_reconstructed", m0.get("r2_recon", "N/A"))
    r2_c2 = m2.get("r2_reconstructed", m2.get("r2_recon", "N/A"))
    neg_c0 = m0.get("negativity_frac", "N/A")
    neg_c2 = m2.get("negativity_frac", "N/A")
    print(f"  Sim {i}: C0 R²={r2_c0:.6f} neg={neg_c0:.1f}%  |  C2 R²={r2_c2:.6f} neg={neg_c2:.1f}%  |  ΔR²={r2_c2-r2_c0:+.8f}")
