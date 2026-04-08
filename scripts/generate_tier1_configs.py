#!/usr/bin/env python3
"""
Generate Tier 1 Experiment Configs
====================================

Creates 18 YAML configs for the Tier 1 systematic experiment:
  9 regimes × 2 lag variants (w=5 baseline, w=w*_BIC data-driven)

Output directory: configs/systematic/tier1/

Input base configs:
  configs/systematic/main_regimes/          gas, blackhole, supernova, pure_vicsek (CS + VS)
  configs/systematic/NDYN07_crystal.yaml    crystal CS
  configs/systematic/variable_speed/
    NDYN07_crystal_VS.yaml                  crystal VS

Changes applied to every Tier 1 config:
  sim.N = 300
  rom.mass_postprocess = "none"
  rom.models.lstm.hidden_units = 128  (up from 64; matches Phase B)
  eval.clamp_mode = "C0"              (no post-inverse clamping)
  wsindy.bootstrap.enabled = true, B = 200, ci_alpha = 0.05
  wsindy.model_selection.n_ell = 20   (rich enough for Pareto selection)

BIC lags (from Phase B Alvarez at N=100 — valid as N=300 proxies):
  gas CS=2, blackhole CS=2, supernova CS=2, crystal CS=3, pure_vicsek CS=2
  VS variants: lag=2 (conservative estimate; should match Alvarez on N=300 Phase A data)
  NOTE: update VS BIC lags once Tier 1 Phase A Alvarez jobs complete.

Usage:
    python scripts/generate_tier1_configs.py [--output_dir configs/systematic/tier1]
"""

import copy
import yaml
from pathlib import Path
import argparse


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# (short_key, source_path_relative_to_repo_root, variant)
REGIMES = [
    ("NDYN04_gas",          "configs/systematic/main_regimes/NDYN04_gas.yaml",          "CS"),
    ("NDYN04_gas_VS",       "configs/systematic/main_regimes/NDYN04_gas_VS.yaml",        "VS"),
    ("NDYN05_blackhole",    "configs/systematic/main_regimes/NDYN05_blackhole.yaml",     "CS"),
    ("NDYN05_blackhole_VS", "configs/systematic/main_regimes/NDYN05_blackhole_VS.yaml",  "VS"),
    ("NDYN06_supernova",    "configs/systematic/main_regimes/NDYN06_supernova.yaml",     "CS"),
    ("NDYN06_supernova_VS", "configs/systematic/main_regimes/NDYN06_supernova_VS.yaml",  "VS"),
    ("NDYN07_crystal",      "configs/systematic/NDYN07_crystal.yaml",                   "CS"),
    ("NDYN07_crystal_VS",   "configs/systematic/variable_speed/NDYN07_crystal_VS.yaml", "VS"),
    ("NDYN08_pure_vicsek",  "configs/systematic/main_regimes/NDYN08_pure_vicsek.yaml",  "CS"),
]

# BIC-optimal lags from Phase B Alvarez (N=100).
# These are the w*_BIC values per regime; CS values are confirmed, VS are estimates.
# All CS BIC=2 except crystal (3). VS estimates: same as CS — update after N=300 Alvarez.
BIC_LAGS = {
    "NDYN04_gas":          2,
    "NDYN04_gas_VS":       2,   # estimate — confirm after Phase A Alvarez
    "NDYN05_blackhole":    2,
    "NDYN05_blackhole_VS": 2,   # estimate
    "NDYN06_supernova":    2,
    "NDYN06_supernova_VS": 2,   # estimate
    "NDYN07_crystal":      3,
    "NDYN07_crystal_VS":   3,   # estimate — crystal forces identical, BIC likely same
    "NDYN08_pure_vicsek":  2,
}

# Canonical WSINDy section for all Tier 1 configs.
# n_ell=20: richer model selection; bootstrap B=200: stable CI for thesis tables.
WSINDY_TEMPLATE = {
    "enabled": True,
    "mode": "multifield",
    "subsample": 3,
    "seed": 42,
    "multifield_library": {
        "morse": True,
        "rich": True,
        "rho_strategy": "continuity_first",
    },
    "model_selection": {
        "n_ell": 20,
        "p": [3, 5, 5],
        "stride": [2, 2, 2],
    },
    "lambdas": {
        "log_min": -5,
        "log_max": 2,
        "n_points": 60,
    },
    "bootstrap": {
        "enabled": True,
        "B": 200,
        "ci_alpha": 0.05,
    },
    "forecast": {
        "clip_negative": True,
        "mass_conserve": True,
        "method": "auto",
    },
    "n_train": 340,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_tier1_settings(cfg: dict, key: str, lag: int, lag_variant: str) -> dict:
    """
    Apply all Tier 1 modifications to a loaded config dict (deep copy first).

    Parameters
    ----------
    cfg         : raw dict loaded from the base YAML
    key         : regime short key, e.g. "NDYN04_gas"
    lag         : resolved lag integer (5 for w5, BIC value for bic)
    lag_variant : "w5" or "bic" — used to set experiment_name
    """
    c = copy.deepcopy(cfg)

    # 1. Particle count
    c.setdefault("sim", {})["N"] = 300

    # 2. ROM section
    rom = c.setdefault("rom", {})
    rom["mass_postprocess"] = "none"

    # Delete training densities after POD build to avoid disk-quota crash
    c["cleanup_train_after_pod"] = True

    models = rom.setdefault("models", {})

    mvar = models.setdefault("mvar", {})
    mvar["enabled"] = True
    mvar["lag"] = lag
    mvar.setdefault("ridge_alpha", 1e-4)

    lstm = models.setdefault("lstm", {})
    lstm["enabled"] = True
    lstm["lag"] = lag
    lstm["hidden_units"] = 128   # upgrade from legacy 64
    lstm.setdefault("num_layers", 2)
    lstm["dropout"] = 0.0        # ablation: no benefit from dropout
    lstm["residual"] = False     # ablation: +53 pts R² when OFF
    lstm["use_layer_norm"] = True   # ablation: +36 pts R² when ON
    lstm.setdefault("normalize_input", True)
    lstm.setdefault("max_epochs", 300)
    lstm.setdefault("patience", 40)
    lstm.setdefault("learning_rate", 7e-4)
    lstm.setdefault("batch_size", 512)
    lstm.setdefault("gradient_clip", 5.0)
    lstm.setdefault("weight_decay", 1e-5)

    # 3. Eval section
    c.setdefault("eval", {})["clamp_mode"] = "C0"

    # 4. WSINDy section (full canonical template)
    c["wsindy"] = copy.deepcopy(WSINDY_TEMPLATE)

    # 5. Experiment name
    c["experiment_name"] = f"{key}_tier1_{lag_variant}"

    return c


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        default="configs/systematic/tier1",
        help="Output directory for generated configs (default: configs/systematic/tier1)",
    )
    parser.add_argument(
        "--repo_root",
        default=".",
        help="Path to repository root (default: current directory)",
    )
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = repo / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = []

    for key, src_rel, variant in REGIMES:
        src_path = repo / src_rel
        if not src_path.exists():
            print(f"  [SKIP] Source not found: {src_path}")
            continue

        with open(src_path) as f:
            base_cfg = yaml.safe_load(f)

        bic_lag = BIC_LAGS[key]

        for lag_variant, lag_int in [("w5", 5), ("bic", bic_lag)]:
            cfg = apply_tier1_settings(base_cfg, key, lag_int, lag_variant)

            out_name = f"{key}_tier1_{lag_variant}.yaml"
            out_path = out_dir / out_name

            # Write header comment + YAML body
            is_vs = variant == "VS"
            bic_note = (
                f"  # BIC lag: confirmed from Phase B Alvarez"
                if variant == "CS"
                else f"  # BIC lag: estimated (update after Tier 1 Phase A Alvarez)"
            )
            header_lines = [
                "---",
                f"# Tier 1 — {key}  [{variant}]  lag={lag_int} ({lag_variant})",
                f"# N=300, sqrt+none preprocessing, WSINDy+MVAR+LSTM",
            ]
            if lag_variant == "bic":
                header_lines.append(bic_note)

            with open(out_path, "w") as f:
                f.write("\n".join(header_lines) + "\n")
                # Serialise without the leading "---" that yaml.dump may add
                body = yaml.dump(cfg, default_flow_style=False, sort_keys=False)
                # Remove yaml.dump's own leading "---\n" if present
                body = body.lstrip("---\n")
                f.write(body)

            generated.append(out_name)
            lag_src = "confirmed" if variant == "CS" else "estimate"
            print(f"  {out_name:55s}  (lag={lag_int}, {lag_src})")

    print(f"\nGenerated {len(generated)} configs → {out_dir}/")
    if len(generated) == 18:
        print("  ✓ All 18 Tier 1 configs created.")
    else:
        missing = 18 - len(generated)
        print(f"  ⚠  Expected 18, got {len(generated)} ({missing} skipped — check source paths).")

    # Write a descriptive manifest
    manifest_path = out_dir / "manifest_info.txt"
    with open(manifest_path, "w") as f:
        f.write("# Tier 1 config manifest\n")
        f.write("# 9 regimes × 2 lag variants = 18 tasks\n")
        f.write("#\n")
        f.write("# Columns: config_file  lag_variant  bic_lag  variant\n")
        for key, _, variant in REGIMES:
            bic_lag = BIC_LAGS[key]
            for lag_variant, lag_int in [("w5", 5), ("bic", bic_lag)]:
                bic_src = "confirmed" if variant == "CS" else "estimate"
                f.write(f"{key}_tier1_{lag_variant}.yaml  {lag_variant}  {lag_int}  {variant}  {bic_src}\n")

    # Write slurm-compatible manifest (plain paths, one per line)
    slurm_manifest = out_dir / "manifest.txt"
    rel_dir = out_dir.relative_to(repo)
    with open(slurm_manifest, "w") as f:
        for name in generated:
            f.write(f"{rel_dir}/{name}\n")
    print(f"  Manifest: {slurm_manifest} ({len(generated)} entries)")
    print(f"  Info:     {manifest_path}")


if __name__ == "__main__":
    main()
