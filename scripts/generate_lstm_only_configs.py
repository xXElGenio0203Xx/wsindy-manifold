#!/usr/bin/env python3
"""Generate LSTM-only YAML configs from existing experiment configs.

Each config sets resume_from_latent: true so that the pipeline
re-uses cached rom_common/ data and only trains the LSTM.
"""

import yaml
import copy
from pathlib import Path

# Mapping: (source config path, source_dir for rom_common symlink, new experiment name)
REGIMES = [
    ("/tmp/oscar_output/NDYN06_supernova_thesis_final/config_used.yaml",
     "NDYN06_supernova_thesis_final", "NDYN06_supernova_lstm"),
    ("/tmp/oscar_output/NDYN07_crystal_wsindy_v3/config_used.yaml",
     "NDYN07_crystal_wsindy_v3", "NDYN07_crystal_lstm"),
    ("/tmp/oscar_output/NDYN08_pure_vicsek_thesis_final/config_used.yaml",
     "NDYN08_pure_vicsek_thesis_final", "NDYN08_pure_vicsek_lstm"),
    ("/tmp/oscar_output/NDYN04_gas_VS_thesis_final/config_used.yaml",
     "NDYN04_gas_VS_thesis_final", "NDYN04_gas_VS_lstm"),
    ("/tmp/oscar_output/NDYN05_blackhole_VS_thesis_final/config_used.yaml",
     "NDYN05_blackhole_VS_thesis_final", "NDYN05_blackhole_VS_lstm"),
    ("/tmp/oscar_output/NDYN06_supernova_VS_thesis_final/config_used.yaml",
     "NDYN06_supernova_VS_thesis_final", "NDYN06_supernova_VS_lstm"),
    ("/tmp/oscar_output/NDYN07_crystal_VS_wsindy_v3/config_used.yaml",
     "NDYN07_crystal_VS_wsindy_v3", "NDYN07_crystal_VS_lstm"),
]

OUT_DIR = Path("configs/lstm_only")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for src_path, source_dir, new_name in REGIMES:
    with open(src_path) as f:
        cfg = yaml.safe_load(f)

    # Add resume_from_latent
    cfg["resume_from_latent"] = True

    # Disable MVAR, ensure LSTM enabled
    models = cfg.get("rom", {}).get("models", {})
    models["mvar"]["enabled"] = False
    models["lstm"]["enabled"] = True

    # Remove WSINDy section entirely
    cfg.pop("wsindy", None)

    # Update experiment name
    cfg["experiment_name"] = new_name

    # Add comment about source dir for the SLURM script
    cfg["_rom_source_dir"] = source_dir

    out_path = OUT_DIR / f"{new_name}.yaml"
    with open(out_path, "w") as f:
        f.write(f"---\n# LSTM-only config for {new_name}\n")
        f.write(f"# ROM data sourced from: oscar_output/{source_dir}/rom_common\n\n")
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"  Created: {out_path}")

# Also write the manifest for the SLURM array job
manifest_path = OUT_DIR / "manifest.txt"
with open(manifest_path, "w") as f:
    for _, source_dir, new_name in REGIMES:
        f.write(f"{new_name} {source_dir}\n")
print(f"\n  Manifest: {manifest_path}")
print(f"  Total: {len(REGIMES)} LSTM-only jobs")
