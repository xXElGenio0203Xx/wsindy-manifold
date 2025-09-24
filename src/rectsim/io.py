"""Input/output utilities for simulation artifacts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create a directory path if needed and return it as a :class:`Path`."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_npz(out_dir: str | Path, name: str, **arrays: np.ndarray) -> Path:
    """Save numpy arrays into an ``.npz`` archive within ``out_dir``."""

    directory = ensure_dir(out_dir)
    path = directory / f"{name}.npz"
    np.savez(path, **arrays)
    return path


def save_csv(out_dir: str | Path, name: str, df: pd.DataFrame) -> Path:
    """Persist a :class:`pandas.DataFrame` as a CSV file in ``out_dir``."""

    directory = ensure_dir(out_dir)
    path = directory / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def git_commit_hash() -> str | None:
    """Return the current git commit hash if available in the environment."""

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover - git optional
        return None
    return commit


def save_run_metadata(out_dir: str | Path, config: Dict, results: Dict) -> Path:
    """Write a JSON summary of the run configuration and outcomes."""

    directory = ensure_dir(out_dir)
    payload = {
        "seed": config.get("seed"),
        "config": config,
        "rcut": results.get("rcut"),
        "force_time": results.get("force_time"),
        "force_evals": results.get("force_evals"),
        "git_commit": git_commit_hash(),
    }
    path = directory / "run.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


__all__ = ["ensure_dir", "save_npz", "save_csv", "save_run_metadata"]
