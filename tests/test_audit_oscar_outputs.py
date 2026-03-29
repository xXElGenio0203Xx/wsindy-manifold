from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import yaml


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "audit_oscar_outputs.py"
SPEC = importlib.util.spec_from_file_location("audit_oscar_outputs", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_extract_metric_from_summary_handles_wsindy_uppercase() -> None:
    summary = {
        "WSINDy": {
            "mean_r2_test": 0.42,
            "mean_r2_1step_test": 0.91,
        }
    }
    metric = MODULE.extract_metric_from_summary(summary, "wsindy")
    assert metric.mean_r2 == 0.42
    assert metric.mean_r2_1step == 0.91
    assert metric.source == "summary"


def test_build_enriched_rows_flags_meansub_and_uses_csv_fallback(tmp_path: Path) -> None:
    archive_root = tmp_path / "experiments"
    exp_dir = archive_root / "LPREP_TEST"
    exp_dir.mkdir(parents=True)

    with open(exp_dir / "config_used.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "rom": {
                    "density_transform": "meansub",
                    "mass_postprocess": "scale",
                    "shift_align": True,
                    "fixed_modes": 19,
                    "subsample": 3,
                    "models": {
                        "mvar": {"enabled": True, "lag": 5, "ridge_alpha": 1e-4},
                        "lstm": {"enabled": True, "lag": 5, "hidden_units": 64, "num_layers": 2},
                    },
                }
            },
            handle,
            sort_keys=False,
        )

    write_csv(
        exp_dir / "MVAR" / "test_results.csv",
        [
            {"r2_reconstructed": 0.1, "r2_1step": 0.3},
            {"r2_reconstructed": 0.5, "r2_1step": 0.7},
        ],
    )

    rows = MODULE.build_enriched_rows(
        [
            {
                "name": "LPREP_TEST",
                "family": "LPREP",
                "remote_dir": "/remote/LPREP_TEST",
                "has_summary": False,
                "has_config": True,
                "has_mvar_results": True,
                "has_lstm_results": False,
                "has_wsindy_results": False,
                "has_fallback_test_results": False,
            }
        ],
        archive_root,
    )

    row = rows[0]
    assert row["completion_status"] == "partial"
    assert "missing_summary" in row["known_issues"]
    assert "meansub_metrics_invalid" in row["known_issues"]
    assert row["exclude_from_rankings"] is True
    assert row["mvar_mean_r2"] == 0.3
    assert row["mvar_mean_r2_1step"] == 0.5
    assert row["mvar_r2_source"] == "test_results"


def test_rows_for_rankings_excludes_meansub_and_family_balances() -> None:
    rows = [
        {"family": "LPREP", "exclude_from_rankings": True, "lstm_mean_r2": 0.99, "density_transform": "meansub"},
        {"family": "DO", "exclude_from_rankings": False, "lstm_mean_r2": 0.2, "density_transform": "raw"},
        {"family": "DO", "exclude_from_rankings": False, "lstm_mean_r2": 0.4, "density_transform": "raw"},
        {"family": "NDYN", "exclude_from_rankings": False, "lstm_mean_r2": 0.8, "density_transform": "raw"},
    ]

    ranked_rows = MODULE.rows_for_rankings(rows, "lstm_mean_r2")
    assert len(ranked_rows) == 3

    summaries = MODULE.summarize_groups(
        ranked_rows,
        metric_key="lstm_mean_r2",
        signature_keys=["density_transform"],
    )
    assert len(summaries) == 1
    summary = summaries[0]
    assert round(summary["mean_r2"], 6) == round((0.2 + 0.4 + 0.8) / 3, 6)
    assert round(summary["family_balanced_r2"], 6) == round((((0.2 + 0.4) / 2) + 0.8) / 2, 6)
