#!/usr/bin/env python3
"""Generate lean WSINDy-only configs from existing noise_sweep and n_convergence configs."""
import yaml
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "configs"

def make_lean(src_dir, dst_dir, suffix="_ws"):
    """Copy configs from src_dir to dst_dir, disabling LSTM and MVAR."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_dir.glob("*.yaml")):
        with open(src) as f:
            cfg = yaml.safe_load(f)

        # Disable LSTM
        models = cfg.get("rom", {}).get("models", {})
        if "lstm" in models:
            models["lstm"]["enabled"] = False
        # Disable MVAR
        if "mvar" in models:
            models["mvar"]["enabled"] = False

        # Update experiment name to avoid clobbering existing outputs
        old_name = cfg.get("experiment_name", src.stem)
        cfg["experiment_name"] = old_name + suffix

        # Ensure WSINDy is enabled with bootstrap
        ws = cfg.get("wsindy", {})
        ws["enabled"] = True
        if "bootstrap" in ws:
            ws["bootstrap"]["enabled"] = True
            ws["bootstrap"]["B"] = 200
        cfg["wsindy"] = ws

        # Cleanup to save time
        cfg["cleanup_train_after_pod"] = True

        dst = dst_dir / src.name
        with open(dst, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=True)
        print(f"  {src.name} -> {dst}")


print("=== Noise sweep (WSINDy-only) ===")
make_lean(BASE / "noise_sweep", BASE / "noise_sweep_ws", suffix="_ws")

print("\n=== N-convergence (WSINDy-only) ===")
make_lean(BASE / "n_convergence", BASE / "n_convergence_ws", suffix="_ws")

print("\nDone. Configs written to:")
print(f"  {BASE / 'noise_sweep_ws'}")
print(f"  {BASE / 'n_convergence_ws'}")
