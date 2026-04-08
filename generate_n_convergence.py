#!/usr/bin/env python3
"""
Generate N-Convergence Configs (Tier 4)
========================================

Creates YAML configs sweeping particle count N for the pure Vicsek regime.
Used to verify the Messenger & Bortz O(N^{-1/2}) convergence rate.

Sweep: N ∈ {50, 100, 200, 300, 500, 1000}
Fixed: grid 64×64, bandwidth h=5, lag w=5, all models enabled.

6 configs total.

Usage:
    python generate_n_convergence.py [--output_dir configs/n_convergence]
"""

import yaml
from pathlib import Path
import argparse

REGIME = "NDYN08_pure_vicsek"
N_VALUES = [50, 100, 200, 300, 500, 1000]


def load_base_config() -> dict:
    """Load the Tier 1 w=5 config for pure Vicsek (N=300 + all Tier 1 settings)."""
    candidates = [
        Path(f"configs/systematic/tier1/{REGIME}_tier1_w5.yaml"),
        Path(f"configs/systematic/{REGIME}.yaml"),
        Path(f"configs/{REGIME}.yaml"),
        Path(f"configs/DYN7_pure_vicsek.yaml"),
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(
        f"No base config found for {REGIME} in {[str(c) for c in candidates]}"
    )


def generate_n_configs(output_dir: Path):
    """Generate all N-convergence YAML configs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    generated = []

    base = load_base_config()

    for n in N_VALUES:
        name = f"{REGIME}_N{n:04d}"

        cfg = yaml.safe_load(yaml.dump(base))  # deep copy
        cfg["sim"]["N"] = n

        # Ensure Tier 1 settings
        cfg.setdefault("rom", {})["mass_postprocess"] = "none"
        cfg["cleanup_train_after_pod"] = True
        cfg["experiment_name"] = name

        models = cfg.get("rom", {}).get("models", {})
        if "lstm" in models:
            models["lstm"]["hidden_units"] = 128
            models["lstm"]["residual"] = False      # ablation: +53 pts R²
            models["lstm"]["dropout"] = 0.0          # ablation: no benefit
            models["lstm"]["use_layer_norm"] = True  # ablation: +36 pts R²

        # Tag for provenance
        cfg.setdefault("meta", {})
        cfg["meta"]["tier"] = 4
        cfg["meta"]["sweep"] = "n_convergence"
        cfg["meta"]["is_baseline"] = (n == 300)

        out_path = output_dir / f"{name}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        count += 1
        generated.append(f"{name}.yaml")
        marker = " (baseline)" if n == 300 else ""
        print(f"  {out_path.name}  N={n}{marker}")

    # Write slurm-compatible manifest
    manifest = output_dir / "manifest.txt"
    with open(manifest, "w") as f:
        for g in generated:
            f.write(f"{output_dir}/{g}\n")

    print(f"\nGenerated {count} N-convergence configs in {output_dir}/")
    print(f"  Manifest: {manifest}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Tier 4 N-convergence configs")
    parser.add_argument("--output_dir", type=str,
                        default="configs/n_convergence",
                        help="Output directory for generated configs")
    args = parser.parse_args()
    generate_n_configs(Path(args.output_dir))


if __name__ == "__main__":
    main()
