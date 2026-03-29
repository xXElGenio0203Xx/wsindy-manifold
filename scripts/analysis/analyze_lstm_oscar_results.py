#!/usr/bin/env python3
"""
Analyze LSTM experiment outcomes from Oscar output directories.

This script compares LSTM rollout R^2 across experiment families, config
signatures, matched experiment pairs, and per-test IC distributions.

Usage:
    python scripts/analysis/analyze_lstm_oscar_results.py
    python scripts/analysis/analyze_lstm_oscar_results.py --base-dir .
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Experiment:
    root_name: str
    name: str
    exp_dir: Path
    mean_r2_lstm: float | None
    mean_r2_1step_lstm: float | None
    mean_r2_mvar: float | None
    n_train: int | None
    n_test: int | None
    test_T: float | None
    density_transform: str | None
    mass_postprocess: str | None
    shift_align: bool | None
    lag_lstm: int | None
    hidden_units: int | None
    num_layers: int | None
    dropout: float | None
    learning_rate: float | None
    batch_size: int | None
    gradient_clip: float | None
    weight_decay: float | None
    patience: int | None
    normalize_input: bool | None


def fmt(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def discover_roots(base_dir: Path) -> list[tuple[str, Path]]:
    roots = []
    root = base_dir / "oscar_output"
    if root.exists():
        roots.append(("oscar_output", root))
    systematics = root / "systematics"
    if systematics.exists():
        roots.append(("oscar_output/systematics", systematics))
    return roots


def load_yaml(path: Path) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def load_experiments(base_dir: Path) -> list[Experiment]:
    experiments: list[Experiment] = []
    for root_name, root in discover_roots(base_dir):
        for summary_path in sorted(root.glob("*/summary.json")):
            with open(summary_path) as handle:
                summary = json.load(handle)

            cfg_path = summary_path.parent / "config_used.yaml"
            cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
            rom = cfg.get("rom", {})
            lstm_cfg = rom.get("models", {}).get("lstm", {})
            lstm = summary.get("lstm") or {}
            mvar = summary.get("mvar") or {}

            if "mean_r2_test" not in lstm:
                continue

            experiments.append(
                Experiment(
                    root_name=root_name,
                    name=summary_path.parent.name,
                    exp_dir=summary_path.parent,
                    mean_r2_lstm=lstm.get("mean_r2_test"),
                    mean_r2_1step_lstm=lstm.get("mean_r2_1step_test"),
                    mean_r2_mvar=mvar.get("mean_r2_test"),
                    n_train=summary.get("n_train"),
                    n_test=summary.get("n_test"),
                    test_T=(cfg.get("test_sim") or {}).get("T"),
                    density_transform=rom.get("density_transform"),
                    mass_postprocess=rom.get("mass_postprocess"),
                    shift_align=rom.get("shift_align"),
                    lag_lstm=lstm_cfg.get("lag"),
                    hidden_units=lstm_cfg.get("hidden_units", lstm.get("hidden_units")),
                    num_layers=lstm_cfg.get("num_layers", lstm.get("num_layers")),
                    dropout=lstm_cfg.get("dropout"),
                    learning_rate=lstm_cfg.get("learning_rate", lstm_cfg.get("lr")),
                    batch_size=lstm_cfg.get("batch_size"),
                    gradient_clip=lstm_cfg.get("gradient_clip", lstm_cfg.get("grad_clip")),
                    weight_decay=lstm_cfg.get("weight_decay"),
                    patience=lstm_cfg.get("patience"),
                    normalize_input=lstm_cfg.get("normalize_input"),
                )
            )
    return experiments


def load_distribution_map(exp_dir: Path) -> dict[str, str]:
    meta_path = exp_dir / "test" / "metadata.json"
    if not meta_path.exists():
        return {}

    with open(meta_path) as handle:
        metadata = json.load(handle)

    mapping: dict[str, str] = {}
    if isinstance(metadata, list):
        for item in metadata:
            mapping[str(item.get("run_id"))] = item.get("distribution") or item.get("label") or "unknown"
    return mapping


def find_lstm_results_csv(exp_dir: Path) -> Path | None:
    candidates = [
        exp_dir / "LSTM" / "test_results.csv",
        exp_dir / "test" / "test_results.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_per_test_rows(experiments: list[Experiment]) -> list[dict]:
    rows: list[dict] = []
    for exp in experiments:
        csv_path = find_lstm_results_csv(exp.exp_dir)
        if csv_path is None:
            continue

        distribution_map = load_distribution_map(exp.exp_dir)
        with open(csv_path) as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    r2 = float(row.get("r2_reconstructed") or row.get("r2_test") or "nan")
                except ValueError:
                    continue
                try:
                    r2_1step = float(row.get("r2_1step") or "nan")
                except ValueError:
                    r2_1step = float("nan")

                test_id = str(row.get("test_id") or row.get("test_idx") or "")
                distribution = (
                    row.get("distribution")
                    or row.get("ic_type")
                    or distribution_map.get(test_id)
                    or "unknown"
                )
                rows.append(
                    {
                        "exp_name": exp.name,
                        "distribution": distribution,
                        "r2": r2,
                        "r2_1step": r2_1step,
                    }
                )
    return rows


def print_top_bottom(experiments: list[Experiment], top_k: int = 10) -> None:
    ranked = sorted(experiments, key=lambda exp: exp.mean_r2_lstm or -float("inf"), reverse=True)
    print("Top LSTM experiments by rollout R^2")
    print("-" * 80)
    for exp in ranked[:top_k]:
        print(
            f"{exp.root_name:<22} {exp.name:<30} "
            f"r2={fmt(exp.mean_r2_lstm):>8} mvar={fmt(exp.mean_r2_mvar):>8} "
            f"lag={fmt(exp.lag_lstm):>4} h={fmt(exp.hidden_units):>3} "
            f"dt={exp.density_transform} mass={exp.mass_postprocess}"
        )

    print("\nBottom LSTM experiments by rollout R^2")
    print("-" * 80)
    for exp in ranked[-top_k:]:
        print(
            f"{exp.root_name:<22} {exp.name:<30} "
            f"r2={fmt(exp.mean_r2_lstm):>8} mvar={fmt(exp.mean_r2_mvar):>8} "
            f"lag={fmt(exp.lag_lstm):>4} h={fmt(exp.hidden_units):>3} "
            f"dt={exp.density_transform} mass={exp.mass_postprocess}"
        )


def print_signature_stats(experiments: list[Experiment]) -> None:
    groups: dict[tuple, list[Experiment]] = defaultdict(list)
    for exp in experiments:
        key = (
            exp.lag_lstm,
            exp.hidden_units,
            exp.num_layers,
            exp.dropout,
            exp.density_transform,
            exp.mass_postprocess,
            exp.learning_rate,
            exp.batch_size,
            exp.gradient_clip,
            exp.weight_decay,
            exp.patience,
            exp.normalize_input,
        )
        groups[key].append(exp)

    print("\nArchitecture signature summary")
    print("-" * 80)
    ordered = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            -(sum(exp.mean_r2_lstm for exp in item[1] if exp.mean_r2_lstm is not None) / len(item[1])),
        ),
    )
    for key, group in ordered:
        values = [exp.mean_r2_lstm for exp in group if exp.mean_r2_lstm is not None]
        good = sum(value >= 0.9 for value in values)
        bad = sum(value < 0.5 for value in values)
        print(
            f"count={len(group):>2} mean={sum(values)/len(values):>9.4f} "
            f"median={median(values):>9.4f} good>=0.9={good:>2} bad<0.5={bad:>2} "
            f"sig={key}"
        )


def print_matched_pairs(experiments: list[Experiment]) -> None:
    by_root: dict[str, dict[str, Experiment]] = defaultdict(dict)
    for exp in experiments:
        by_root[exp.root_name][exp.name] = exp

    root = by_root.get("oscar_output", {})
    systematics = by_root.get("oscar_output/systematics", {})
    common_names = sorted(set(root) & set(systematics))
    if not common_names:
        return

    print("\nMatched root vs systematics experiments")
    print("-" * 80)
    for name in common_names:
        root_exp = root[name]
        sys_exp = systematics[name]
        delta = (sys_exp.mean_r2_lstm or 0.0) - (root_exp.mean_r2_lstm or 0.0)
        print(
            f"{name:<28} root={fmt(root_exp.mean_r2_lstm):>10} "
            f"sys={fmt(sys_exp.mean_r2_lstm):>10} delta={delta:>10.4f}"
        )
        print(
            f"  root: lag={fmt(root_exp.lag_lstm)} h={fmt(root_exp.hidden_units)} "
            f"dt={root_exp.density_transform} mass={root_exp.mass_postprocess}"
        )
        print(
            f"  sys : lag={fmt(sys_exp.lag_lstm)} h={fmt(sys_exp.hidden_units)} "
            f"dt={sys_exp.density_transform} mass={sys_exp.mass_postprocess}"
        )


def print_distribution_stats(per_test_rows: list[dict]) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in per_test_rows:
        groups[row["distribution"]].append(row)

    print("\nPer-test LSTM rollout by IC distribution")
    print("-" * 80)
    for distribution, rows in sorted(groups.items()):
        values = [row["r2"] for row in rows]
        one_step = [row["r2_1step"] for row in rows if not math.isnan(row["r2_1step"])]
        good = sum(value >= 0.9 for value in values)
        neg = sum(value < 0 for value in values)
        one_step_mean = sum(one_step) / len(one_step) if one_step else float("nan")
        print(
            f"{distribution:<18} n={len(rows):>3} mean={sum(values)/len(values):>10.3f} "
            f"median={median(values):>8.3f} good>=0.9={good:>3} "
            f"neg={neg:>3} mean_1step={one_step_mean:>8.3f}"
        )


def print_rollout_vs_onestep(experiments: list[Experiment]) -> None:
    unstable = [
        exp
        for exp in experiments
        if exp.mean_r2_lstm is not None
        and exp.mean_r2_lstm < 0
        and exp.mean_r2_1step_lstm is not None
        and exp.mean_r2_1step_lstm > 0.8
    ]

    print("\nNegative rollout R^2 despite strong one-step fit")
    print("-" * 80)
    print(f"count={len(unstable)}")
    for exp in sorted(unstable, key=lambda item: item.mean_r2_lstm or 0.0)[:15]:
        print(
            f"{exp.name:<30} rollout={fmt(exp.mean_r2_lstm):>10} "
            f"one_step={fmt(exp.mean_r2_1step_lstm):>8} "
            f"dt={exp.density_transform} mass={exp.mass_postprocess} lag={fmt(exp.lag_lstm)}"
        )


def print_findings(experiments: list[Experiment], per_test_rows: list[dict]) -> None:
    by_pair: dict[tuple[str | None, str | None], list[Experiment]] = defaultdict(list)
    for exp in experiments:
        by_pair[(exp.density_transform, exp.mass_postprocess)].append(exp)

    print("\nKey findings")
    print("-" * 80)

    raw_none = by_pair.get(("raw", "none"), [])
    if raw_none:
        values = [exp.mean_r2_lstm for exp in raw_none if exp.mean_r2_lstm is not None]
        print(
            f"raw + none: n={len(raw_none)}, mean rollout R^2={sum(values)/len(values):.3f}, "
            f"median={median(values):.3f}"
        )

    sqrt_scale = by_pair.get(("sqrt", "scale"), [])
    if sqrt_scale:
        values = [exp.mean_r2_lstm for exp in sqrt_scale if exp.mean_r2_lstm is not None]
        print(
            f"sqrt + scale: n={len(sqrt_scale)}, mean rollout R^2={sum(values)/len(values):.3f}, "
            f"median={median(values):.3f}"
        )

    sqrt_simplex = by_pair.get(("sqrt", "simplex"), [])
    if sqrt_simplex:
        values = [exp.mean_r2_lstm for exp in sqrt_simplex if exp.mean_r2_lstm is not None]
        print(
            f"sqrt + simplex: n={len(sqrt_simplex)}, mean rollout R^2={sum(values)/len(values):.3f}, "
            f"median={median(values):.3f}"
        )

    unstable = [
        exp for exp in experiments
        if exp.mean_r2_lstm is not None
        and exp.mean_r2_lstm < 0
        and exp.mean_r2_1step_lstm is not None
        and exp.mean_r2_1step_lstm > 0.8
    ]
    print(f"negative-rollout / high-one-step cases: {len(unstable)}")

    by_distribution: dict[str, list[float]] = defaultdict(list)
    for row in per_test_rows:
        by_distribution[row["distribution"]].append(row["r2"])
    hardest = sorted(
        ((distribution, median(values), sum(values) / len(values)) for distribution, values in by_distribution.items()),
        key=lambda item: item[1],
    )
    for distribution, dist_median, dist_mean in hardest:
        print(f"{distribution}: median per-test rollout R^2={dist_median:.3f}, mean={dist_mean:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Oscar LSTM result patterns")
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Repository root containing oscar_output/",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    experiments = load_experiments(base_dir)
    if not experiments:
        raise SystemExit(f"No LSTM summary files found under {base_dir / 'oscar_output'}")

    per_test_rows = load_per_test_rows(experiments)

    print(f"Loaded {len(experiments)} LSTM experiment summaries from Oscar outputs.")
    print_top_bottom(experiments)
    print_signature_stats(experiments)
    print_matched_pairs(experiments)
    if per_test_rows:
        print_distribution_stats(per_test_rows)
    print_rollout_vs_onestep(experiments)
    print_findings(experiments, per_test_rows)


if __name__ == "__main__":
    main()
