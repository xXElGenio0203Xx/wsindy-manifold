"""Quickstart example for 2D rectangular EF-ROM pipeline.

References
~~~~~~~~~~
- D'Orsogna et al. (2006) for the Morse-based swarming equations.
- Vicsek et al. (1995) / AIM-1 Eq. 6 for the alignment rule.
- Bhaskar & Ziegelmeier (2019) for the diagnostics used here.

This script demonstrates the end-to-end workflow:
1. Load a YAML configuration.
2. Run the D'Orsogna + Vicsek simulator.
3. Convert trajectories to KDE density movies.
4. Train a POD+VAR latent model (EF-ROM).
5. Forecast future density fields, compute metrics, and write animations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rectsim.cli import _run_single
from rectsim.config import load_config
from rectsim.density import density_movie_kde
from rectsim.metrics import mean_relative_error, r2, rmse, tolerance_horizon
from wsindy_manifold.efrom import efrom_train_and_forecast
from wsindy_manifold.latent.anim import animate_heatmap_movie, animate_side_by_side

CONFIG = Path("examples/configs/single.yaml")
OUTPUT = Path("outputs/quickstart")


def main() -> None:
    cfg = load_config(CONFIG)
    cfg["out_dir"] = str(OUTPUT / "sim")
    run = _run_single(cfg)

    sim_cfg = cfg["sim"]
    grid_cfg = cfg["outputs"]["grid_density"]
    traj = run["results"]["traj"]

    rho = density_movie_kde(
        traj,
        sim_cfg["Lx"],
        sim_cfg["Ly"],
        grid_cfg["nx"],
        grid_cfg["ny"],
        grid_cfg.get("bandwidth", 0.5),
        sim_cfg["bc"],
    )

    rank = cfg["outputs"]["efrom"]["rank"]
    order = cfg["outputs"]["efrom"]["order"]
    horizon = cfg["outputs"]["efrom"]["horizon"]

    split = int(0.8 * rho.shape[0])
    rho_train = rho[:split]
    rho_test = rho[split:]
    cell_area = (sim_cfg["Lx"] / grid_cfg["nx"]) * (sim_cfg["Ly"] / grid_cfg["ny"])
    rho_pred, info = efrom_train_and_forecast(
        rho_train,
        rho_test,
        rank=rank,
        order=order,
        horizon=min(horizon, rho_test.shape[0]),
        cell_area=cell_area,
    )

    rho_true = rho_test[: rho_pred.shape[0]]
    flat_true = rho_true.reshape(rho_true.shape[0], -1)
    flat_pred = rho_pred.reshape(rho_pred.shape[0], -1)

    metrics = {
        "rmse": float(rmse(flat_true, flat_pred)),
        "r2": float(r2(flat_true, flat_pred)),
        "mean_relative_error": mean_relative_error(flat_true, flat_pred, axis=1).tolist(),
        "tolerance_horizon": tolerance_horizon(mean_relative_error(flat_true, flat_pred, axis=1)),
        "mass_drift": float(np.max(np.abs(info["mass_series"] - traj.shape[1]))),
    }
    print("Validation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    out_dir = OUTPUT / "efrom"
    out_dir.mkdir(parents=True, exist_ok=True)
    dx = sim_cfg["Lx"] / grid_cfg["nx"]
    dy = sim_cfg["Ly"] / grid_cfg["ny"]
    xv = (np.arange(grid_cfg["nx"]) + 0.5) * dx
    yv = (np.arange(grid_cfg["ny"]) + 0.5) * dy
    grid_centres = np.stack(np.meshgrid(xv, yv, indexing="xy"), axis=-1).reshape(-1, 2)
    vmin = float(min(rho_true.min(), rho_pred.min()))
    vmax = float(max(rho_true.max(), rho_pred.max()))

    animate_heatmap_movie(
        rho_true.reshape(rho_true.shape[0], -1),
        grid_centres,
        dx,
        dy,
        sim_cfg["Lx"],
        sim_cfg["Ly"],
        out_path=str(out_dir / "truth.mp4"),
        fps=20,
        vmin=vmin,
        vmax=vmax,
        title="Truth",
    )
    animate_heatmap_movie(
        rho_pred.reshape(rho_pred.shape[0], -1),
        grid_centres,
        dx,
        dy,
        sim_cfg["Lx"],
        sim_cfg["Ly"],
        out_path=str(out_dir / "forecast.mp4"),
        fps=20,
        vmin=vmin,
        vmax=vmax,
        title="EF-ROM",
    )
    animate_side_by_side(
        rho_true.reshape(rho_true.shape[0], -1),
        rho_pred.reshape(rho_pred.shape[0], -1),
        grid_centres,
        dx,
        dy,
        sim_cfg["Lx"],
        sim_cfg["Ly"],
        out_path=str(out_dir / "compare.mp4"),
        fps=20,
        vmin=vmin,
        vmax=vmax,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
