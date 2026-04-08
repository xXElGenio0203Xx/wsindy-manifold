#!/usr/bin/env python3
"""
Generate Noise Sweep Configs (Tier 2)
======================================

Creates YAML configs varying observation noise η for 4 force-bearing regimes.
Each config is derived from the corresponding Tier 1 w=5 YAML with only the
noise level overridden.

Sweep specification (from EXPERIMENTAL_PLAN_FINAL.md):
  - NDYN04_gas:       η ∈ {0.15, 0.5, 1.0, 1.5*, 2.0}   (* = native)
  - NDYN05_blackhole: η ∈ {0.05, 0.15*, 0.5, 1.0, 1.5}
  - NDYN06_supernova: η ∈ {0.05, 0.15*, 0.5, 1.0, 1.5}
  - NDYN07_crystal:   η ∈ {0.05, 0.1*, 0.5, 1.0, 1.5}

20 configs total (4 regimes × 5 noise levels).
Lag: w=5 only (isolate noise effect from lag effect).
Native noise configs duplicate Tier 1 w=5 for completeness.

Usage:
    python generate_noise_sweep.py [--output_dir configs/noise_sweep]
"""

import yaml
from pathlib import Path
import argparse

NOISE_SWEEPS = {
    "NDYN04_gas": {
        "native": 1.5,
        "sweep": [0.15, 0.5, 1.0, 1.5, 2.0],
    },
    "NDYN05_blackhole": {
        "native": 0.15,
        "sweep": [0.05, 0.15, 0.5, 1.0, 1.5],
    },
    "NDYN06_supernova": {
        "native": 0.15,
        "sweep": [0.05, 0.15, 0.5, 1.0, 1.5],
    },
    "NDYN07_crystal": {
        "native": 0.1,
        "sweep": [0.05, 0.1, 0.5, 1.0, 1.5],
    },
}


def load_base_config(regime_name: str) -> dict:
    """Load the Tier 1 w=5 config for a regime (N=300 + all Tier 1 settings)."""
    candidates = [
        Path(f"configs/systematic/tier1/{regime_name}_tier1_w5.yaml"),
        Path(f"configs/systematic/{regime_name}.yaml"),
        Path(f"configs/{regime_name}.yaml"),
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(
        f"No base config found for {regime_name} in {[str(c) for c in candidates]}"
    )


def generate_noise_configs(output_dir: Path):
    """Generate all noise-sweep YAML configs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    generated = []

    for regime, spec in NOISE_SWEEPS.items():
        base = load_base_config(regime)

        for eta in spec["sweep"]:
            # Format eta for filename: 0.15 -> "0p15"
            eta_str = f"{eta:.2f}".replace(".", "p")
            name = f"{regime}_eta{eta_str}"

            cfg = yaml.safe_load(yaml.dump(base))  # deep copy
            cfg["noise"]["eta"] = float(eta)

            # Ensure Tier 1 settings are present (in case base wasn't from tier1/)
            cfg.setdefault("sim", {})["N"] = 300
            cfg.setdefault("rom", {})["mass_postprocess"] = "none"
            cfg["cleanup_train_after_pod"] = True
            cfg["experiment_name"] = name

            # Tag the config for provenance
            cfg.setdefault("meta", {})
            cfg["meta"]["tier"] = 2
            cfg["meta"]["sweep"] = "noise"
            cfg["meta"]["native_eta"] = spec["native"]
            cfg["meta"]["is_native"] = abs(eta - spec["native"]) < 1e-9

            out_path = output_dir / f"{name}.yaml"
            with open(out_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            count += 1
            generated.append(f"{name}.yaml")
            marker = " (native)" if cfg["meta"]["is_native"] else ""
            print(f"  {out_path.name}{marker}")

    # Write slurm-compatible manifest
    manifest = output_dir / "manifest.txt"
    with open(manifest, "w") as f:
        for g in generated:
            f.write(f"{output_dir}/{g}\n")

    print(f"\nGenerated {count} noise-sweep configs in {output_dir}/")
    print(f"  Manifest: {manifest}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Tier 2 noise-sweep configs")
    parser.add_argument("--output_dir", type=str,
                        default="configs/noise_sweep",
                        help="Output directory for generated configs")
    args = parser.parse_args()
    generate_noise_configs(Path(args.output_dir))


if __name__ == "__main__":
    main()
