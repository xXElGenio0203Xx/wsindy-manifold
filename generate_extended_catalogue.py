#!/usr/bin/env python3
"""
Generate Extended Catalogue Configs (Tier 3)
=============================================

Creates YAML configs for the remaining NDYN + all d'Orsogna (DO) regimes
not covered by Tier 1 (main regimes).

Tier 3 runs MVAR + LSTM only (no WSINDy) at w=5.

NDYN residuals (after removing main regimes and duplicates):
  NDYN01_gentle, NDYN02_flock, NDYN09_long_range, NDYN10_sparse,
  NDYN13_drift, NDYN14_clustered
  (NDYN03/11/12 removed as duplicates of NDYN02/05/06)

DO suite: all 14 d'Orsogna configs from generate_regime_configs.py

Usage:
    python generate_extended_catalogue.py [--output_dir configs/extended_catalogue]
"""

import yaml
from pathlib import Path
import argparse

# NDYN regimes not in Tier 1 (04-08) and not removed as duplicates (03/11/12)
NDYN_RESIDUAL = [
    "NDYN01_crawl",
    "NDYN02_flock",
    "NDYN09_longrange",
    "NDYN10_shortrange",
    "NDYN13_chaos",
]

# All DO configs are Tier 3 (full filenames)
DO_CONFIGS = [
    "DO_CS01_swarm_C01_l05",
    "DO_CS02_swarm_C05_l3",
    "DO_CS03_swarm_C09_l3",
    "DO_DM01_dmill_C09_l05",
    "DO_DR01_dring_C01_l01",
    "DO_DR02_dring_C09_l09",
    "DO_EC01_esccol_C2_l3",
    "DO_EC02_esccol_C3_l05",
    "DO_ES01_escsym_C3_l09",
    "DO_EU01_escuns_C2_l2",
    "DO_EU02_escuns_C3_l3",
    "DO_SM01_mill_C05_l01",
    "DO_SM02_mill_C3_l01",
    "DO_SM03_mill_C2_l05",
]


def find_config(name: str) -> Path:
    """Find a config file by regime name."""
    candidates = [
        Path(f"configs/systematic/{name}.yaml"),
        Path(f"configs/{name}.yaml"),
    ]
    # Also search subdirectories
    for base in [Path("configs/systematic"), Path("configs")]:
        if base.exists():
            matches = list(base.rglob(f"{name}.yaml"))
            candidates.extend(matches)

    for p in candidates:
        if p.exists():
            return p
    return None


def generate_extended_configs(output_dir: Path):
    """Generate Tier 3 extended catalogue configs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    missing = []
    generated = []

    all_regimes = NDYN_RESIDUAL + DO_CONFIGS

    for regime in all_regimes:
        src = find_config(regime)
        if src is None:
            missing.append(regime)
            continue

        with open(src) as f:
            cfg = yaml.safe_load(f)

        # Apply Tier 1 baseline settings
        cfg.setdefault("sim", {})["N"] = 300
        rom = cfg.setdefault("rom", {})
        rom["mass_postprocess"] = "none"
        cfg["cleanup_train_after_pod"] = True
        cfg["experiment_name"] = regime

        # Tier 3: MVAR + LSTM only, no WSINDy
        models = rom.setdefault("models", {})
        if "wsindy" in cfg:
            cfg["wsindy"]["enabled"] = False
        if "wsindy" in models:
            models["wsindy"]["enabled"] = False

        # Ensure lag=5 (baseline)
        if "mvar" in models:
            models["mvar"]["lag"] = 5
        if "lstm" in models:
            models["lstm"]["lag"] = 5
            models["lstm"]["hidden_units"] = 128
            models["lstm"]["residual"] = False      # ablation: +53 pts R²
            models["lstm"]["dropout"] = 0.0          # ablation: no benefit
            models["lstm"]["use_layer_norm"] = True  # ablation: +36 pts R²

        # Tag for provenance
        cfg.setdefault("meta", {})
        cfg["meta"]["tier"] = 3
        cfg["meta"]["suite"] = "NDYN" if regime.startswith("NDYN") else "DO"

        out_path = output_dir / f"{regime}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        count += 1
        generated.append(f"{regime}.yaml")
        print(f"  {out_path.name}")

    # Write slurm-compatible manifest
    manifest = output_dir / "manifest.txt"
    with open(manifest, "w") as f:
        for g in generated:
            f.write(f"{output_dir}/{g}\n")

    if missing:
        print(f"\nWARNING: {len(missing)} configs not found: {missing}")
    print(f"\nGenerated {count} extended-catalogue configs in {output_dir}/")
    print(f"  Manifest: {manifest}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Tier 3 extended catalogue configs")
    parser.add_argument("--output_dir", type=str,
                        default="configs/extended_catalogue",
                        help="Output directory for generated configs")
    args = parser.parse_args()
    generate_extended_configs(Path(args.output_dir))


if __name__ == "__main__":
    main()
