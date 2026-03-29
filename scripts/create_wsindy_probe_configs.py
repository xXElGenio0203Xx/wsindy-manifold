#!/usr/bin/env python3
"""Generate a small WSINDy-only probe suite from systematic configs."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rectsim.ic_generator import generate_training_configs

SYSTEMATIC_DIR = REPO_ROOT / "configs" / "systematic"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "configs" / "wsindy_probe"

# Pick the most stable completed systematic regimes while covering
# distinct qualitative behaviors: ring, mill, and swarm.
SELECTED_REGIMES = [
    "DO_DR02_dring_C09_l09",
    "DO_SM02_mill_C3_l01",
    "DO_CS01_swarm_C01_l05",
]

BASE_WSINDY_TEMPLATE = {
    "enabled": True,
    "mode": "multifield",
    "subsample": 3,
    "seed": 42,
    "multifield_library": {
        "morse": True,
        "rich": False,
    },
    "model_selection": {
        "n_ell": 12,
        "p": [2, 2, 2],
        "stride": [2, 2, 2],
    },
    "lambdas": {
        "log_min": -5,
        "log_max": 2,
        "n_points": 60,
    },
    # Bootstrap does not affect forecast R^2, so keep the probe cheap.
    "bootstrap": {
        "enabled": False,
        "B": 0,
        "ci_alpha": 0.05,
    },
    "forecast": {
        "clip_negative": True,
        "mass_conserve": True,
        "method": "auto",
    },
}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def build_probe_config(
    base_name: str,
    *,
    name_prefix: str,
    rich: bool,
) -> tuple[str, dict]:
    base_path = SYSTEMATIC_DIR / f"{base_name}.yaml"
    config = load_yaml(base_path)
    full_train_count = len(generate_training_configs(config["train_ic"], config))

    probe_name = f"{name_prefix}_{base_name}"
    config["experiment_name"] = probe_name

    rom_models = config.setdefault("rom", {}).setdefault("models", {})
    rom_models.setdefault("mvar", {})["enabled"] = False
    rom_models.setdefault("lstm", {})["enabled"] = False

    config["wsindy"] = deepcopy(BASE_WSINDY_TEMPLATE)
    config["wsindy"]["multifield_library"]["rich"] = rich
    config["wsindy"]["n_train"] = full_train_count
    return probe_name, config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate WSINDy-only probe configs from systematic configs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where generated configs and manifest are written.",
    )
    parser.add_argument(
        "--name-prefix",
        default="WSY",
        help="Experiment-name prefix for the generated configs.",
    )
    parser.add_argument(
        "--rich",
        action="store_true",
        help="Enable the richer multifield WSINDy library.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[str] = []
    for base_name in SELECTED_REGIMES:
        probe_name, config = build_probe_config(
            base_name,
            name_prefix=args.name_prefix,
            rich=args.rich,
        )
        out_path = output_dir / f"{probe_name}.yaml"
        dump_yaml(out_path, config)
        manifest_entries.append(out_path.name)

    manifest_path = output_dir / "manifest.txt"
    manifest_path.write_text("\n".join(manifest_entries) + "\n", encoding="utf-8")

    library_mode = "rich" if args.rich else "baseline"
    print(
        f"Wrote {len(manifest_entries)} {library_mode} WSINDy probe configs to {output_dir}"
    )
    for entry in manifest_entries:
        print(f"  - {entry}")


if __name__ == "__main__":
    main()
