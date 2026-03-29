#!/usr/bin/env python3
"""
Create a compact local audit archive from OSCAR experiment outputs.

This script reads the real OSCAR scratch tree over SSH, mirrors only compact
metadata locally, and builds tables/notes that can be reviewed before any
remote deletion happens.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
from typing import Iterable

import yaml


DEFAULT_REMOTE_ROOT = "/oscar/scratch/emaciaso/oscar_output"
DEFAULT_DEST = Path("~/wsindy-oscar-audit") / date.today().isoformat()
TARGET_FAMILIES = ["DO", "NDYN", "LPREP", "PSW", "VFIX", "WSY", "WSYR"]

REMOTE_INVENTORY_SCRIPT = r"""
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for exp in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
    def has_file(*parts):
        return (exp.joinpath(*parts)).is_file()

    match = re.match(r"[A-Za-z]+", exp.name)
    family = match.group(0) if match else exp.name.split("_", 1)[0]

    rows.append({
        "name": exp.name,
        "family": family,
        "remote_dir": str(exp),
        "has_summary": has_file("summary.json"),
        "has_config": has_file("config_used.yaml"),
        "has_mvar_results": has_file("MVAR", "test_results.csv"),
        "has_lstm_results": has_file("LSTM", "test_results.csv"),
        "has_wsindy_results": has_file("WSINDy", "test_results.csv"),
        "has_fallback_test_results": has_file("test", "test_results.csv"),
    })

print(json.dumps(rows))
"""

REMOTE_COUNTS_SCRIPT = r"""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
dirs = [p for p in root.iterdir() if p.is_dir()]
payload = {
    "total_dirs": len(dirs),
    "with_summary": sum((p / "summary.json").is_file() for p in dirs),
    "with_config": sum((p / "config_used.yaml").is_file() for p in dirs),
}
print(json.dumps(payload))
"""


@dataclass
class MetricValue:
    mean_r2: float | None
    mean_r2_1step: float | None
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive compact OSCAR metadata and summarize MVAR/LSTM design outcomes."
    )
    parser.add_argument("--remote-host", default="oscar")
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--dest", default=str(DEFAULT_DEST))
    return parser.parse_args()


def run_command(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        capture_output=True,
        text=True,
    )


def run_ssh(remote_host: str, remote_script: str, *script_args: str) -> str:
    encoded = base64.b64encode(remote_script.encode("utf-8")).decode("ascii")
    quoted_args = " ".join(shlex.quote(arg) for arg in script_args)
    launcher = (
        "import base64; exec(base64.b64decode("
        + repr(encoded)
        + ").decode('utf-8'))"
    )
    remote_cmd = f"python3 -c {shlex.quote(launcher)} {quoted_args}"
    result = run_command(["ssh", remote_host, remote_cmd])
    return result.stdout


def collect_remote_inventory(remote_host: str, remote_root: str) -> list[dict]:
    stdout = run_ssh(remote_host, REMOTE_INVENTORY_SCRIPT, remote_root)
    return json.loads(stdout)


def collect_remote_counts(remote_host: str, remote_root: str) -> dict:
    stdout = run_ssh(remote_host, REMOTE_COUNTS_SCRIPT, remote_root)
    return json.loads(stdout)


def relative_archive_paths(row: dict) -> list[str]:
    name = row["name"]
    wanted: list[str] = []
    if row["has_config"]:
        wanted.append(f"{name}/config_used.yaml")
    if row["has_summary"]:
        wanted.append(f"{name}/summary.json")
    if row["has_mvar_results"]:
        wanted.append(f"{name}/MVAR/test_results.csv")
    if row["has_lstm_results"]:
        wanted.append(f"{name}/LSTM/test_results.csv")
    if row["has_wsindy_results"]:
        wanted.append(f"{name}/WSINDy/test_results.csv")
    if (
        row["has_fallback_test_results"]
        and not row["has_mvar_results"]
        and not row["has_lstm_results"]
        and not row["has_wsindy_results"]
    ):
        wanted.append(f"{name}/test/test_results.csv")
    return wanted


def mirror_compact_archive(
    inventory_rows: list[dict],
    *,
    remote_host: str,
    remote_root: str,
    archive_root: Path,
) -> int:
    file_list = sorted(
        {
            relative_path
            for row in inventory_rows
            for relative_path in relative_archive_paths(row)
        }
    )
    archive_root.mkdir(parents=True, exist_ok=True)
    if not file_list:
        return 0

    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
        for relative_path in file_list:
            handle.write(relative_path + "\n")
        file_list_path = handle.name

    try:
        run_command(
            [
                "rsync",
                "-avz",
                "--files-from",
                file_list_path,
                f"{remote_host}:{remote_root.rstrip('/')}/",
                str(archive_root),
            ]
        )
    finally:
        os.unlink(file_list_path)

    return len(file_list)


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(value):
            return float(value)
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def mean(values: Iterable[float]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def extract_metric_from_summary(summary: dict, model_key: str) -> MetricValue:
    model_summary = summary.get(model_key)
    if model_summary is None and model_key.lower() == "wsindy":
        model_summary = summary.get("WSINDy")
    if not isinstance(model_summary, dict):
        return MetricValue(None, None, "missing")

    rollout_keys = ["mean_r2_test", "r2_test", "mean_r2", "rollout_r2", "r2_rollout"]
    one_step_keys = ["mean_r2_1step_test", "mean_r2_1step", "r2_1step", "one_step_r2"]

    rollout = next((parse_float(model_summary.get(key)) for key in rollout_keys if parse_float(model_summary.get(key)) is not None), None)
    one_step = next((parse_float(model_summary.get(key)) for key in one_step_keys if parse_float(model_summary.get(key)) is not None), None)
    if rollout is None and one_step is None:
        return MetricValue(None, None, "missing")
    return MetricValue(rollout, one_step, "summary")


def extract_metric_from_results_csv(path: Path) -> MetricValue:
    if not path.exists():
        return MetricValue(None, None, "missing")

    rollout_values: list[float] = []
    one_step_values: list[float] = []
    with open(path, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rollout = (
                parse_float(row.get("r2_reconstructed"))
                or parse_float(row.get("r2_test"))
                or parse_float(row.get("mean_r2_test"))
            )
            one_step = (
                parse_float(row.get("r2_1step"))
                or parse_float(row.get("mean_r2_1step_test"))
            )
            if rollout is not None:
                rollout_values.append(rollout)
            if one_step is not None:
                one_step_values.append(one_step)

    return MetricValue(mean(rollout_values), mean(one_step_values), "test_results")


def best_metric_value(*values: MetricValue) -> MetricValue:
    for value in values:
        if value.mean_r2 is not None or value.mean_r2_1step is not None:
            return value
    return MetricValue(None, None, "missing")


def exp_archive_dir(archive_root: Path, exp_name: str) -> Path:
    return archive_root / exp_name


def format_metric(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "nan"
    if value == 0:
        return "0.0000"
    if abs(value) >= 1e4:
        return f"{value:.3e}"
    return f"{value:.4f}"


def stringify_signature(parts: dict[str, object], keys: list[str]) -> str:
    return json.dumps({key: parts.get(key) for key in keys}, sort_keys=True)


def family_balanced_mean(rows: list[dict], metric_key: str) -> float | None:
    per_family: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = parse_float(row.get(metric_key))
        if value is None:
            continue
        per_family[row["family"]].append(value)
    family_means = [sum(values) / len(values) for values in per_family.values() if values]
    return mean(family_means)


def summarize_groups(rows: list[dict], *, metric_key: str, signature_keys: list[str]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        value = parse_float(row.get(metric_key))
        if value is None:
            continue
        signature = stringify_signature(row, signature_keys)
        grouped[signature].append(row)

    summaries: list[dict] = []
    for signature, members in grouped.items():
        values = [parse_float(member.get(metric_key)) for member in members]
        valid = [value for value in values if value is not None]
        if not valid:
            continue
        families = sorted({member["family"] for member in members})
        summaries.append(
            {
                "signature": signature,
                "count": len(valid),
                "family_count": len(families),
                "mean_r2": float(sum(valid) / len(valid)),
                "median_r2": float(statistics.median(valid)),
                "family_balanced_r2": family_balanced_mean(members, metric_key),
                "families": ",".join(families),
            }
        )
    return summaries


def top_bottom(entries: list[dict], metric_key: str, top_n: int = 5) -> tuple[list[dict], list[dict]]:
    ranked = [entry for entry in entries if parse_float(entry.get(metric_key)) is not None]
    ranked.sort(key=lambda entry: parse_float(entry.get(metric_key)) or -math.inf, reverse=True)
    return ranked[:top_n], list(reversed(ranked[-top_n:]))


def choose_recommendation(
    entries: list[dict],
    *,
    metric_key: str,
    min_families: int = 2,
    require_positive: bool = True,
) -> tuple[dict | None, str]:
    ranked = [entry for entry in entries if parse_float(entry.get(metric_key)) is not None]
    ranked.sort(key=lambda entry: parse_float(entry.get(metric_key)) or -math.inf, reverse=True)

    robust = [entry for entry in ranked if entry.get("family_count", 0) >= min_families]
    if require_positive:
        robust = [entry for entry in robust if (parse_float(entry.get(metric_key)) or -math.inf) > 0]
    if robust:
        return robust[0], f"family_balanced_{min_families}plus"

    if require_positive:
        positive = [entry for entry in ranked if (parse_float(entry.get(metric_key)) or -math.inf) > 0]
        if positive:
            return positive[0], "positive_fallback"

    return (ranked[0], "unfiltered_fallback") if ranked else (None, "missing")


def detect_known_issues(row: dict, cfg: dict | None) -> list[str]:
    issues: list[str] = []
    if not row["has_summary"]:
        issues.append("missing_summary")
    if not row["has_config"]:
        issues.append("missing_config")
    if row["has_fallback_test_results"] and not any(
        row[key] for key in ("has_mvar_results", "has_lstm_results", "has_wsindy_results")
    ):
        issues.append("aggregate_test_results_only")
    if row["family"] in {"WSY", "WSYR"} and not row["has_summary"]:
        issues.append("wsindy_export_incomplete")
    if cfg:
        density_transform = ((cfg.get("rom") or {}).get("density_transform"))
        if density_transform == "meansub":
            issues.append("meansub_metrics_invalid")
    return issues


def build_enriched_rows(inventory_rows: list[dict], archive_root: Path) -> list[dict]:
    enriched_rows: list[dict] = []
    for base_row in inventory_rows:
        row = dict(base_row)
        archive_dir = exp_archive_dir(archive_root, row["name"])
        cfg_path = archive_dir / "config_used.yaml"
        summary_path = archive_dir / "summary.json"

        cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
        summary = load_json(summary_path) if summary_path.exists() else {}

        rom = cfg.get("rom") or {}
        models = rom.get("models") or {}
        mvar_cfg = models.get("mvar") or {}
        lstm_cfg = models.get("lstm") or {}

        row["completion_status"] = "completed" if row["has_summary"] else "partial"
        row["known_issues"] = ";".join(detect_known_issues(row, cfg))
        row["exclude_from_rankings"] = "meansub_metrics_invalid" in row["known_issues"]

        row["density_transform"] = rom.get("density_transform")
        row["mass_postprocess"] = rom.get("mass_postprocess")
        row["shift_align"] = rom.get("shift_align")
        row["shift_align_ref"] = rom.get("shift_align_ref")
        row["fixed_modes"] = rom.get("fixed_modes")
        row["subsample"] = rom.get("subsample")
        row["forecast_start"] = (cfg.get("eval") or {}).get("forecast_start")
        row["test_T"] = (cfg.get("test_sim") or {}).get("T")
        row["timestamp"] = summary.get("timestamp")
        row["n_train"] = summary.get("n_train")
        row["n_test"] = summary.get("n_test")

        row["mvar_lag"] = mvar_cfg.get("lag", summary.get("lag"))
        row["mvar_ridge_alpha"] = mvar_cfg.get("ridge_alpha")
        row["mvar_enabled"] = bool(mvar_cfg.get("enabled")) if mvar_cfg else None

        row["lstm_enabled"] = bool(lstm_cfg.get("enabled")) if lstm_cfg else None
        row["lstm_lag"] = lstm_cfg.get("lag")
        row["lstm_hidden_units"] = lstm_cfg.get("hidden_units")
        row["lstm_num_layers"] = lstm_cfg.get("num_layers")
        row["lstm_dropout"] = lstm_cfg.get("dropout")
        row["lstm_residual"] = lstm_cfg.get("residual")
        row["lstm_use_layer_norm"] = lstm_cfg.get("use_layer_norm")
        row["lstm_normalize_input"] = lstm_cfg.get("normalize_input")
        row["lstm_learning_rate"] = lstm_cfg.get("learning_rate", lstm_cfg.get("lr"))
        row["lstm_batch_size"] = lstm_cfg.get("batch_size")
        row["lstm_gradient_clip"] = lstm_cfg.get("gradient_clip", lstm_cfg.get("grad_clip"))
        row["lstm_weight_decay"] = lstm_cfg.get("weight_decay")
        row["lstm_patience"] = lstm_cfg.get("patience")

        mvar_value = best_metric_value(
            extract_metric_from_summary(summary, "mvar"),
            extract_metric_from_results_csv(archive_dir / "MVAR" / "test_results.csv"),
            extract_metric_from_results_csv(archive_dir / "test" / "test_results.csv"),
        )
        lstm_value = best_metric_value(
            extract_metric_from_summary(summary, "lstm"),
            extract_metric_from_results_csv(archive_dir / "LSTM" / "test_results.csv"),
            extract_metric_from_results_csv(archive_dir / "test" / "test_results.csv"),
        )
        wsindy_value = best_metric_value(
            extract_metric_from_summary(summary, "wsindy"),
            extract_metric_from_results_csv(archive_dir / "WSINDy" / "test_results.csv"),
        )

        row["mvar_mean_r2"] = mvar_value.mean_r2
        row["mvar_mean_r2_1step"] = mvar_value.mean_r2_1step
        row["mvar_r2_source"] = mvar_value.source

        row["lstm_mean_r2"] = lstm_value.mean_r2
        row["lstm_mean_r2_1step"] = lstm_value.mean_r2_1step
        row["lstm_r2_source"] = lstm_value.source

        row["wsindy_mean_r2"] = wsindy_value.mean_r2
        row["wsindy_mean_r2_1step"] = wsindy_value.mean_r2_1step
        row["wsindy_r2_source"] = wsindy_value.source

        enriched_rows.append(row)

    return enriched_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def validate_counts(inventory_rows: list[dict], remote_counts: dict, archive_root: Path) -> dict:
    local_summary = sum((archive_root / row["name"] / "summary.json").exists() for row in inventory_rows)
    local_config = sum((archive_root / row["name"] / "config_used.yaml").exists() for row in inventory_rows)
    payload = {
        "remote_total_dirs": remote_counts["total_dirs"],
        "inventory_total_dirs": len(inventory_rows),
        "remote_with_summary": remote_counts["with_summary"],
        "inventory_with_summary": sum(row["has_summary"] for row in inventory_rows),
        "archived_with_summary": local_summary,
        "remote_with_config": remote_counts["with_config"],
        "inventory_with_config": sum(row["has_config"] for row in inventory_rows),
        "archived_with_config": local_config,
    }
    if payload["remote_total_dirs"] != payload["inventory_total_dirs"]:
        raise RuntimeError(f"Inventory count mismatch: {payload}")
    if payload["remote_with_summary"] != payload["inventory_with_summary"]:
        raise RuntimeError(f"Summary count mismatch: {payload}")
    if payload["remote_with_config"] != payload["inventory_with_config"]:
        raise RuntimeError(f"Config count mismatch: {payload}")
    if payload["archived_with_summary"] != payload["inventory_with_summary"]:
        raise RuntimeError(f"Archived summary count mismatch: {payload}")
    if payload["archived_with_config"] != payload["inventory_with_config"]:
        raise RuntimeError(f"Archived config count mismatch: {payload}")
    return payload


def select_spot_checks(inventory_rows: list[dict], limit: int = 10) -> list[dict]:
    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in inventory_rows:
        by_family[row["family"]].append(row)

    picks: list[dict] = []
    seen: set[str] = set()
    for family in TARGET_FAMILIES:
        family_rows = by_family.get(family, [])
        max_for_family = 2 if family in {"DO", "LPREP", "PSW", "VFIX"} else 1
        for row in family_rows[:max_for_family]:
            if row["name"] not in seen:
                picks.append(row)
                seen.add(row["name"])

    if len(picks) < limit:
        for row in inventory_rows:
            if row["name"] in seen:
                continue
            picks.append(row)
            seen.add(row["name"])
            if len(picks) >= limit:
                break
    return picks[:limit]


def validate_spot_checks(inventory_rows: list[dict], archive_root: Path) -> list[dict]:
    checks: list[dict] = []
    for row in select_spot_checks(inventory_rows):
        archive_dir = exp_archive_dir(archive_root, row["name"])
        files = {
            "config_used.yaml": (archive_dir / "config_used.yaml").exists(),
            "summary.json": (archive_dir / "summary.json").exists(),
            "MVAR/test_results.csv": (archive_dir / "MVAR" / "test_results.csv").exists(),
            "LSTM/test_results.csv": (archive_dir / "LSTM" / "test_results.csv").exists(),
            "WSINDy/test_results.csv": (archive_dir / "WSINDy" / "test_results.csv").exists(),
            "test/test_results.csv": (archive_dir / "test" / "test_results.csv").exists(),
        }
        checks.append(
            {
                "name": row["name"],
                "family": row["family"],
                "config_match": files["config_used.yaml"] == row["has_config"],
                "summary_match": files["summary.json"] == row["has_summary"],
                "mvar_match": files["MVAR/test_results.csv"] == row["has_mvar_results"],
                "lstm_match": files["LSTM/test_results.csv"] == row["has_lstm_results"],
                "wsindy_match": files["WSINDy/test_results.csv"] == row["has_wsindy_results"],
                "fallback_match": files["test/test_results.csv"]
                == (
                    row["has_fallback_test_results"]
                    and not row["has_mvar_results"]
                    and not row["has_lstm_results"]
                    and not row["has_wsindy_results"]
                ),
            }
        )
    for check in checks:
        if not all(value for key, value in check.items() if key.endswith("_match")):
            raise RuntimeError(f"Spot-check mismatch: {check}")
    return checks


def write_validation_report(
    dest: Path,
    *,
    count_validation: dict,
    spot_checks: list[dict],
    copied_files: int,
) -> None:
    payload = {
        "count_validation": count_validation,
        "spot_checks": spot_checks,
        "copied_files": copied_files,
    }
    with open(dest / "validation_report.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def rows_for_completed_metrics(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["has_summary"]]


def rows_for_rankings(rows: list[dict], metric_key: str) -> list[dict]:
    valid: list[dict] = []
    for row in rows:
        if row["exclude_from_rankings"]:
            continue
        value = parse_float(row.get(metric_key))
        if value is None:
            continue
        valid.append(row)
    return valid


def build_notes(rows: list[dict], dest: Path) -> None:
    completed_rows = rows_for_completed_metrics(rows)
    excluded_rows = [row for row in completed_rows if row["exclude_from_rankings"]]

    mvar_signature_keys = [
        "mvar_lag",
        "mvar_ridge_alpha",
        "density_transform",
        "mass_postprocess",
        "shift_align",
        "fixed_modes",
        "subsample",
    ]
    lstm_arch_keys = [
        "lstm_lag",
        "lstm_hidden_units",
        "lstm_num_layers",
        "lstm_dropout",
        "lstm_residual",
        "lstm_use_layer_norm",
        "lstm_normalize_input",
        "lstm_learning_rate",
        "lstm_batch_size",
        "lstm_gradient_clip",
        "lstm_weight_decay",
        "lstm_patience",
    ]
    lstm_preprocess_keys = [
        "density_transform",
        "mass_postprocess",
        "shift_align",
        "fixed_modes",
        "subsample",
    ]
    lstm_full_keys = lstm_arch_keys + lstm_preprocess_keys

    mvar_rank_rows = rows_for_rankings(completed_rows, "mvar_mean_r2")
    lstm_rank_rows = rows_for_rankings(completed_rows, "lstm_mean_r2")

    mvar_summaries = summarize_groups(mvar_rank_rows, metric_key="mvar_mean_r2", signature_keys=mvar_signature_keys)
    lstm_full_summaries = summarize_groups(lstm_rank_rows, metric_key="lstm_mean_r2", signature_keys=lstm_full_keys)
    lstm_arch_summaries = summarize_groups(lstm_rank_rows, metric_key="lstm_mean_r2", signature_keys=lstm_arch_keys)
    lstm_preprocess_summaries = summarize_groups(lstm_rank_rows, metric_key="lstm_mean_r2", signature_keys=lstm_preprocess_keys)

    mvar_top_overall, mvar_bottom_overall = top_bottom(mvar_summaries, "mean_r2")
    mvar_top_balanced, mvar_bottom_balanced = top_bottom(mvar_summaries, "family_balanced_r2")
    lstm_top_overall, lstm_bottom_overall = top_bottom(lstm_full_summaries, "mean_r2")
    lstm_top_balanced, lstm_bottom_balanced = top_bottom(lstm_full_summaries, "family_balanced_r2")
    lstm_arch_top, _ = top_bottom(lstm_arch_summaries, "family_balanced_r2")
    lstm_pre_top, _ = top_bottom(lstm_preprocess_summaries, "family_balanced_r2")

    recommended_mvar, recommended_mvar_basis = choose_recommendation(
        mvar_summaries,
        metric_key="family_balanced_r2",
        min_families=2,
        require_positive=True,
    )
    recommended_lstm_arch, recommended_lstm_arch_basis = choose_recommendation(
        lstm_arch_summaries,
        metric_key="family_balanced_r2",
        min_families=2,
        require_positive=True,
    )
    recommended_lstm_pre, recommended_lstm_pre_basis = choose_recommendation(
        lstm_preprocess_summaries,
        metric_key="family_balanced_r2",
        min_families=2,
        require_positive=True,
    )

    family_counts = Counter(row["family"] for row in rows)
    completed_family_counts = Counter(row["family"] for row in completed_rows)

    lines: list[str] = []
    lines.append("# OSCAR Audit Notes")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Remote directories inventoried: {len(rows)}")
    lines.append(f"- Completed runs with `summary.json`: {len(completed_rows)}")
    lines.append(f"- Runs excluded from rankings: {len(excluded_rows)}")
    if excluded_rows:
        excluded_counts = Counter(issue for row in excluded_rows for issue in row["known_issues"].split(";") if issue)
        lines.append(f"- Exclusion reasons: {', '.join(f'{k}={v}' for k, v in sorted(excluded_counts.items()))}")
    lines.append(f"- Dominant experiment families: {', '.join(f'{family}={count}' for family, count in family_counts.most_common(8))}")
    lines.append(f"- Completed family spread: {', '.join(f'{family}={count}' for family, count in completed_family_counts.most_common(8))}")
    lines.append("")

    def add_table_section(title: str, top_rows: list[dict], bottom_rows: list[dict], metric_key: str) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| kind | score | count | families | signature |")
        lines.append("| --- | ---: | ---: | --- | --- |")
        for entry in top_rows:
            lines.append(
                f"| top | {format_metric(parse_float(entry.get(metric_key)))} | {entry['count']} | {entry['families']} | `{entry['signature']}` |"
            )
        for entry in bottom_rows:
            lines.append(
                f"| bottom | {format_metric(parse_float(entry.get(metric_key)))} | {entry['count']} | {entry['families']} | `{entry['signature']}` |"
            )
        lines.append("")

    add_table_section("MVAR Overall Rankings", mvar_top_overall, mvar_bottom_overall, "mean_r2")
    add_table_section("MVAR Family-Balanced Rankings", mvar_top_balanced, mvar_bottom_balanced, "family_balanced_r2")
    add_table_section("LSTM Overall Rankings", lstm_top_overall, lstm_bottom_overall, "mean_r2")
    add_table_section("LSTM Family-Balanced Rankings", lstm_top_balanced, lstm_bottom_balanced, "family_balanced_r2")

    lines.append("## Recommendations")
    lines.append("")
    if recommended_mvar:
        lines.append(
            f"- Keep MVAR defaults closest to `{recommended_mvar['signature']}` "
            f"({recommended_mvar_basis}, family_count={recommended_mvar['family_count']}, score={format_metric(parse_float(recommended_mvar.get('family_balanced_r2')))})."
        )
    if recommended_lstm_arch:
        lines.append(
            f"- Use LSTM architecture closest to `{recommended_lstm_arch['signature']}` "
            f"({recommended_lstm_arch_basis}, family_count={recommended_lstm_arch['family_count']}, score={format_metric(parse_float(recommended_lstm_arch.get('family_balanced_r2')))})."
        )
    if recommended_lstm_pre:
        lines.append(
            f"- Use LSTM preprocessing closest to `{recommended_lstm_pre['signature']}` "
            f"({recommended_lstm_pre_basis}, family_count={recommended_lstm_pre['family_count']}, score={format_metric(parse_float(recommended_lstm_pre.get('family_balanced_r2')))})."
        )
    if recommended_lstm_arch_basis != "family_balanced_2plus":
        lines.append("- No positive cross-family LSTM architecture was found in the audited runs; the architecture above is a fallback rather than a robust default.")
    if recommended_lstm_pre_basis != "family_balanced_2plus":
        lines.append("- The LSTM preprocessing recommendation is driven by the best available positive fallback, not a multi-family robust winner.")
    lines.append("- Treat any `density_transform=meansub` run as audit-only; exclude it from model-selection decisions.")
    lines.append("- Use `master_inventory.csv` and `deletion_manifest.txt` together when preparing any later remote cleanup.")
    lines.append("")

    with open(dest / "notes.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def build_deletion_manifest(rows: list[dict], dest: Path) -> None:
    with open(dest / "deletion_manifest.txt", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row["remote_dir"] + "\n")


def main() -> None:
    args = parse_args()
    dest = Path(args.dest).expanduser().resolve()
    archive_root = dest / "experiments"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[1/7] Collecting remote inventory from {args.remote_host}:{args.remote_root}")
    remote_counts = collect_remote_counts(args.remote_host, args.remote_root)
    inventory_rows = collect_remote_inventory(args.remote_host, args.remote_root)

    print(f"[2/7] Mirroring compact metadata for {len(inventory_rows)} experiment directories")
    copied_files = mirror_compact_archive(
        inventory_rows,
        remote_host=args.remote_host,
        remote_root=args.remote_root,
        archive_root=archive_root,
    )

    print("[3/7] Validating archive counts")
    count_validation = validate_counts(inventory_rows, remote_counts, archive_root)
    spot_checks = validate_spot_checks(inventory_rows, archive_root)
    write_validation_report(dest, count_validation=count_validation, spot_checks=spot_checks, copied_files=copied_files)

    print("[4/7] Building inventory and metrics tables")
    enriched_rows = build_enriched_rows(inventory_rows, archive_root)

    inventory_fields = [
        "name",
        "family",
        "remote_dir",
        "completion_status",
        "has_summary",
        "has_config",
        "has_mvar_results",
        "has_lstm_results",
        "has_wsindy_results",
        "has_fallback_test_results",
        "known_issues",
        "exclude_from_rankings",
        "density_transform",
        "mass_postprocess",
        "shift_align",
        "fixed_modes",
        "subsample",
    ]
    write_csv(dest / "master_inventory.csv", enriched_rows, inventory_fields)

    completed_fields = [
        "name",
        "family",
        "timestamp",
        "n_train",
        "n_test",
        "test_T",
        "density_transform",
        "mass_postprocess",
        "shift_align",
        "fixed_modes",
        "subsample",
        "known_issues",
        "exclude_from_rankings",
        "mvar_mean_r2",
        "mvar_mean_r2_1step",
        "mvar_r2_source",
        "lstm_mean_r2",
        "lstm_mean_r2_1step",
        "lstm_r2_source",
        "wsindy_mean_r2",
        "wsindy_mean_r2_1step",
        "wsindy_r2_source",
    ]
    write_csv(dest / "completed_metrics.csv", rows_for_completed_metrics(enriched_rows), completed_fields)

    mvar_fields = [
        "name",
        "family",
        "known_issues",
        "exclude_from_rankings",
        "mvar_enabled",
        "mvar_mean_r2",
        "mvar_mean_r2_1step",
        "mvar_r2_source",
        "mvar_lag",
        "mvar_ridge_alpha",
        "density_transform",
        "mass_postprocess",
        "shift_align",
        "fixed_modes",
        "subsample",
    ]
    write_csv(dest / "mvar_design_table.csv", rows_for_completed_metrics(enriched_rows), mvar_fields)

    lstm_fields = [
        "name",
        "family",
        "known_issues",
        "exclude_from_rankings",
        "lstm_enabled",
        "lstm_mean_r2",
        "lstm_mean_r2_1step",
        "lstm_r2_source",
        "lstm_lag",
        "lstm_hidden_units",
        "lstm_num_layers",
        "lstm_dropout",
        "lstm_residual",
        "lstm_use_layer_norm",
        "lstm_normalize_input",
        "lstm_learning_rate",
        "lstm_batch_size",
        "lstm_gradient_clip",
        "lstm_weight_decay",
        "lstm_patience",
        "density_transform",
        "mass_postprocess",
        "shift_align",
        "fixed_modes",
        "subsample",
    ]
    write_csv(dest / "lstm_design_table.csv", rows_for_completed_metrics(enriched_rows), lstm_fields)

    print("[5/7] Writing recommendations and deletion manifest")
    build_notes(enriched_rows, dest)
    build_deletion_manifest(enriched_rows, dest)

    print("[6/7] Saving run metadata")
    with open(dest / "run_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "remote_host": args.remote_host,
                "remote_root": args.remote_root,
                "dest": str(dest),
                "copied_files": copied_files,
                "remote_counts": remote_counts,
            },
            handle,
            indent=2,
        )

    print("[7/7] Done")
    print(f"Archive root: {archive_root}")
    print(f"Inventory:    {dest / 'master_inventory.csv'}")
    print(f"Notes:        {dest / 'notes.md'}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            sys.stderr.write(exc.stderr)
        raise
