#!/usr/bin/env python3
"""Targeted sync for appendix result files without pulling heavy density arrays."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from appendix_results import (
    CATALOGUE_EXPERIMENTS,
    LOCAL_OUTPUT_ROOT,
    LOCAL_SYSTEMATICS_ROOT,
    REMOTE_HOST,
    REMOTE_OUTPUT_ROOT,
    TRACKED_RESULT_FILES,
    unique_sync_targets,
)


def local_target_base(experiment: str, category: str) -> Path:
    if category == "catalogue":
        return LOCAL_SYSTEMATICS_ROOT / experiment
    return LOCAL_OUTPUT_ROOT / experiment


def mirror_local_catalogue_results() -> int:
    mirrored = 0
    for experiment in CATALOGUE_EXPERIMENTS:
        source_base = LOCAL_OUTPUT_ROOT / experiment
        target_base = LOCAL_SYSTEMATICS_ROOT / experiment
        if not source_base.exists():
            continue
        for relative_path in TRACKED_RESULT_FILES:
            source = source_base / relative_path
            target = target_base / relative_path
            if not source.exists() or target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            mirrored += 1
    return mirrored


def sync_file(host: str, remote_path: str, local_path: Path) -> bool:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["rsync", "-az", f"{host}:{remote_path}", str(local_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return True
    if completed.returncode == 23:
        return False
    raise RuntimeError(completed.stderr.strip() or f"rsync failed for {remote_path}")


def sync_experiment(host: str, remote_root: str, experiment: str, target_base: Path) -> bool:
    target_base.mkdir(parents=True, exist_ok=True)
    include_rules = [
        "--include=summary.json",
        "--include=MVAR/",
        "--include=MVAR/test_results.csv",
        "--include=LSTM/",
        "--include=LSTM/test_results.csv",
        "--include=WSINDy/",
        "--include=WSINDy/test_results.csv",
        "--include=WSINDy/multifield_model.json",
        "--include=WSINDy/identification_summary.json",
        "--exclude=*",
    ]
    completed = subprocess.run(
        [
            "rsync",
            "-az",
            "--prune-empty-dirs",
            *include_rules,
            f"{host}:{remote_root}/{experiment}/",
            str(target_base),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return True
    if completed.returncode == 23:
        return False
    raise RuntimeError(completed.stderr.strip() or f"rsync failed for {experiment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=REMOTE_HOST)
    parser.add_argument("--remote-root", default=REMOTE_OUTPUT_ROOT)
    args = parser.parse_args()

    mirrored = mirror_local_catalogue_results()
    synced = 0
    for experiment, category in unique_sync_targets().items():
        target_base = local_target_base(experiment, category)
        if sync_experiment(args.host, args.remote_root, experiment, target_base):
            synced += 1

    print(f"Mirrored {mirrored} local catalogue files into oscar_output/systematics/")
    print(f"Synced appendix results for {synced} experiments from {args.host}:{args.remote_root}")


if __name__ == "__main__":
    main()
