"""Validation CLI for the EF-ROM pipeline."""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rectsim.cli import _run_single
from rectsim.config import load_config
from rectsim.density import density_movie_kde
from rectsim.metrics import (
    angular_momentum,
    mean_relative_error,
    polarization,
    r2,
    rmse,
    tolerance_horizon,
)
from wsindy_manifold.efrom import efrom_train_and_forecast
from wsindy_manifold.latent.anim import animate_heatmap_movie, animate_side_by_side


def _order_parameter_plots(out_dir: Path, traj: np.ndarray, vel: np.ndarray, Lx: float, Ly: float, bc: str) -> None:
    """Save polarization and angular momentum time-series plots."""

    times = np.arange(traj.shape[0])
    pol = [polarization(vel[t]) for t in range(traj.shape[0])]
    ang = [angular_momentum(traj[t], vel[t]) for t in range(traj.shape[0])]
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times, pol, label="polarization")
    ax.plot(times, ang, label="angular momentum")
    ax.set_xlabel("frame")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "order_params.png", dpi=150)
    plt.close(fig)


def run_validate_all(
    cfg_path: Path,
    out_dir: Path,
    efrom_rank: int,
    efrom_order: int,
    horizon: int,
) -> None:
    start_total = time.perf_counter()
    overrides = []
    config = load_config(cfg_path, overrides)
    cfg = deepcopy(config)
    sim_out = out_dir / "sim"
    cfg["out_dir"] = str(sim_out)

    run = _run_single(cfg)
    results = run["results"]
    traj = results["traj"]
    vel = results["vel"]
    times = results["times"]

    sim_cfg = cfg["sim"]
    grid_cfg = cfg["outputs"].get("grid_density", {})
    nx = int(grid_cfg.get("nx", 128))
    ny = int(grid_cfg.get("ny", 128))
    bandwidth = float(grid_cfg.get("bandwidth", 0.5))

    rho = density_movie_kde(
        traj,
        sim_cfg["Lx"],
        sim_cfg["Ly"],
        nx,
        ny,
        bandwidth,
        sim_cfg["bc"],
    )

    order = max(1, efrom_order)
    rank = min(efrom_rank, nx * ny)
    split = max(order + 1, int(0.8 * rho.shape[0]))
    split = min(split, rho.shape[0] - horizon)
    if split <= order or horizon <= 0:
        raise ValueError("Not enough frames for EF-ROM forecast")

    rho_train = rho[:split]
    rho_test = rho[split:]
    horizon = min(horizon, rho_test.shape[0])

    cell_area = (sim_cfg["Lx"] / nx) * (sim_cfg["Ly"] / ny)
    start_efrom = time.perf_counter()
    rho_pred, _ = efrom_train_and_forecast(
        rho_train,
        rho_test,
        rank=rank,
        order=order,
        horizon=horizon,
        cell_area=cell_area,
    )
    timing_efrom = time.perf_counter() - start_efrom
    rho_true = rho_test[:horizon]

    flat_true = rho_true.reshape(horizon, -1)
    flat_pred = rho_pred.reshape(horizon, -1)
    rmse_val = float(rmse(flat_true, flat_pred))
    r2_val = float(r2(flat_true, flat_pred))
    rel_err_series = mean_relative_error(flat_true, flat_pred, axis=1)
    tol_idx = int(tolerance_horizon(rel_err_series))
    mass_pred = flat_pred.sum(axis=1) * cell_area
    mass_target = traj.shape[1]
    mass_drift = float(np.max(np.abs(mass_pred - mass_target)))

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "rmse": rmse_val,
        "r2": r2_val,
        "mean_relative_error": rel_err_series.tolist(),
        "tolerance_horizon": tol_idx,
        "mass_drift": mass_drift,
        "timing": {
            "total_sec": time.perf_counter() - start_total,
            "efrom_sec": timing_efrom,
        },
        "train_frames": int(split),
        "test_frames": int(rho_test.shape[0]),
        "horizon": int(horizon),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    dx = sim_cfg["Lx"] / nx
    dy = sim_cfg["Ly"] / ny
    Xc = ((np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dy)
    grid_centres = np.stack(np.meshgrid(Xc[0], Xc[1], indexing="xy"), axis=-1).reshape(-1, 2)
    vmin = float(min(rho_true.min(), rho_pred.min()))
    vmax = float(max(rho_true.max(), rho_pred.max()))

    animate_heatmap_movie(
        rho_true.reshape(horizon, -1),
        grid_centres,
        dx,
        dy,
        sim_cfg["Lx"],
        sim_cfg["Ly"],
        out_path=str(out_dir / "anim_truth.mp4"),
        fps=20,
        vmin=vmin,
        vmax=vmax,
        title="Truth",
    )
    animate_heatmap_movie(
        rho_pred.reshape(horizon, -1),
        grid_centres,
        dx,
        dy,
        sim_cfg["Lx"],
        sim_cfg["Ly"],
        out_path=str(out_dir / "anim_pred.mp4"),
        fps=20,
        vmin=vmin,
        vmax=vmax,
        title="EF-ROM",
    )
    animate_side_by_side(
        rho_true.reshape(horizon, -1),
        rho_pred.reshape(horizon, -1),
        grid_centres,
        dx,
        dy,
        sim_cfg["Lx"],
        sim_cfg["Ly"],
        out_path=str(out_dir / "anim_side_by_side.mp4"),
        fps=20,
        vmin=vmin,
        vmax=vmax,
    )

    plots_dir = out_dir / "plots"
    _order_parameter_plots(plots_dir, traj, vel, sim_cfg["Lx"], sim_cfg["Ly"], sim_cfg["bc"])

    print(json.dumps(metrics_payload, indent=2))
    print(f"Validation artifacts saved to {out_dir}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WSINDy-Manifold validation CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    val = sub.add_parser("validate_all", help="Run EF-ROM validation pipeline")
    val.add_argument("--cfg", type=Path, required=True, help="Simulation config path")
    val.add_argument("--out", type=Path, required=True, help="Output directory")
    val.add_argument("--efrom-rank", type=int, default=20)
    val.add_argument("--efrom-order", type=int, default=2)
    val.add_argument("--horizon", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "validate_all":
        run_validate_all(
            args.cfg,
            args.out,
            args.efrom_rank,
            args.efrom_order,
            args.horizon,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
