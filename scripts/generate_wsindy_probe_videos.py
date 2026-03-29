#!/usr/bin/env python3
"""
Generate WSINDy-only prediction videos for probe experiments.

This lightweight exporter is intended for recovered WSINDy probe runs that do
not have the full ROM artifacts required by ``run_visualizations.py``.
It writes videos under:

    <workspace>/predictions/<experiment>/best_runs/WSINDY/<ic_type>/

For each IC type it selects the best recovered WSINDy run by
``r2_reconstructed`` and writes:
  - ``traj_truth.mp4``
  - ``density_forecast_only.mp4``
  - ``density_truth_vs_pred_full.mp4``
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rectsim.legacy_functions import compute_frame_metrics
from rectsim.rom_video_utils import make_truth_vs_pred_density_video


def _load_test_metadata(test_dir: Path) -> list[dict]:
    meta_path = test_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as handle:
            return json.load(handle)
    idx_path = test_dir / "index_mapping.csv"
    if idx_path.exists():
        df = pd.read_csv(idx_path)
        run_col = "run_name" if "run_name" in df.columns else df.columns[-1]
        return [{"run_name": rn, "distribution": "unknown"} for rn in df[run_col].tolist()]
    raise FileNotFoundError(f"Missing metadata in {test_dir}")


def _build_config_info(exp_dir: Path) -> str | None:
    cfg_path = exp_dir / "config_used.yaml"
    if not cfg_path.exists():
        return None
    with open(cfg_path) as handle:
        cfg = yaml.safe_load(handle)
    sim = cfg.get("sim", {})
    forces = cfg.get("forces", sim.get("forces", {}))
    fp = forces.get("params", {}) if isinstance(forces, dict) else {}
    lines = [
        f"Exp: {exp_dir.name}",
        f"N={sim.get('N','?')}  Lx={sim.get('Lx','?')}  dt={sim.get('dt','?')}  T={sim.get('T','?')}s",
    ]
    if fp:
        lines.append(
            f"forces: Cr={fp.get('Cr','?')} Ca={fp.get('Ca','?')} lr={fp.get('lr','?')} la={fp.get('la','?')}"
        )
    return "\n".join(lines)


def _align_truth_and_traj(run_dir: Path) -> dict:
    true_data = np.load(run_dir / "density_true.npz")
    pred_data = np.load(run_dir / "density_pred_wsindy.npz")
    traj_data = np.load(run_dir / "trajectory.npz")

    rho_true = true_data["rho"]
    rho_pred = pred_data["rho"]
    times_true = true_data["times"]
    times_pred = pred_data["times"]

    true_start_idx = int(np.argmin(np.abs(times_true - times_pred[0])))
    T_pred = rho_pred.shape[0]
    rho_true = rho_true[true_start_idx:true_start_idx + T_pred]
    times = times_true[true_start_idx:true_start_idx + T_pred]

    traj = traj_data["traj"]
    vel = traj_data["vel"] if "vel" in traj_data else None
    traj_times = traj_data["times"]
    traj_start_idx = int(np.argmin(np.abs(traj_times - times[0])))
    traj = traj[traj_start_idx:traj_start_idx + T_pred]
    if vel is not None:
        vel = vel[traj_start_idx:traj_start_idx + T_pred]

    if traj.shape[0] != T_pred:
        traj = traj[:T_pred]
        if vel is not None:
            vel = vel[:T_pred]
        rho_true = rho_true[:traj.shape[0]]
        rho_pred = rho_pred[:traj.shape[0]]
        times = times[:traj.shape[0]]

    fm = compute_frame_metrics(
        rho_true.reshape(rho_true.shape[0], -1),
        rho_pred.reshape(rho_pred.shape[0], -1),
    )
    return {
        "rho_true": rho_true,
        "rho_pred": rho_pred,
        "times": times,
        "traj": traj,
        "vel": vel,
        "frame_metrics": fm,
    }


def export_experiment(exp_dir: Path, fps: int = 10) -> None:
    exp_dir = Path(exp_dir)
    test_dir = exp_dir / "test"
    ws_dir = exp_dir / "WSINDy"
    if not test_dir.exists() or not ws_dir.exists():
        raise FileNotFoundError(f"Missing test/ or WSINDy/ in {exp_dir}")

    results = pd.read_csv(ws_dir / "test_results.csv")
    metadata = _load_test_metadata(test_dir)
    meta_by_run = {m["run_name"]: m for m in metadata}
    ic_key = "ic_type" if metadata and "ic_type" in metadata[0] else "distribution"
    config_info = _build_config_info(exp_dir)

    pred_root = exp_dir.parent.parent / "predictions" / exp_dir.name / "best_runs" / "WSINDY"
    pred_root.mkdir(parents=True, exist_ok=True)

    rows = []
    results = results.dropna(subset=["r2_reconstructed"]).copy()
    if results.empty:
        raise RuntimeError(f"No valid recovered WSINDy results in {ws_dir / 'test_results.csv'}")

    results["ic_type"] = [
        meta_by_run.get(rn, {}).get(ic_key, "unknown") for rn in results["run_name"]
    ]

    for ic_type, group in results.groupby("ic_type"):
        best = group.sort_values("r2_reconstructed", ascending=False).iloc[0]
        run_name = best["run_name"]
        out_dir = pred_root / str(ic_type)
        out_dir.mkdir(parents=True, exist_ok=True)

        pred = _align_truth_and_traj(test_dir / run_name)

        true_title = "Ground Truth"
        pred_title = f"WSINDy Forecast (R²={best['r2_reconstructed']:.3f})"

        make_truth_vs_pred_density_video(
            density_true=pred["rho_true"],
            density_pred=pred["rho_pred"],
            out_path=out_dir / "density_forecast_only.mp4",
            fps=fps,
            cmap="hot",
            title=f"{exp_dir.name}\n{true_title} vs {pred_title}" if config_info is None else f"{config_info}\n{pred_title}",
            times=pred["times"],
        )
        shutil.copy2(out_dir / "density_forecast_only.mp4", out_dir / "density_truth_vs_pred_full.mp4")

        rows.append(
            {
                "ic_type": ic_type,
                "run_name": run_name,
                "r2_reconstructed": float(best["r2_reconstructed"]),
            }
        )

    pd.DataFrame(rows).to_csv(pred_root.parent / "wsindy_best_runs.csv", index=False)
    print(f"[done] {exp_dir.name}: wrote WSINDY videos to {pred_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate WSINDy-only best-run videos for probe experiments")
    parser.add_argument("experiment_dirs", nargs="+", help="Experiment directories, e.g. ~/wsindy-homeviz/oscar_output/WSYR_DO_DR02_dring_C09_l09")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    for exp in args.experiment_dirs:
        export_experiment(Path(exp), fps=args.fps)


if __name__ == "__main__":
    main()
