"""High-level flows for training and forecasting latent EF-ROM models."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rectsim.io import ensure_dir

from .anim import animate_heatmap_movie, animate_side_by_side
from .kde import trajectories_to_density_movie
from .mvar import fit_mvar, load_mvar, rollout, save_mvar, forecast_step
from .pod import fit_pod, lift_pod, restrict_movie
from .metrics import series_metrics, save_metrics_json_csv

Array = np.ndarray


def _reshape_density(rho: Array, nx: int, ny: int) -> Array:
    """Reshape a flattened density into (ny, nx)."""

    return rho.reshape(ny, nx)


def _save_pod_model(path: Path, model: Dict[str, Array]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        Ud=model["Ud"],
        mean=model["mean"],
        singular_values=model["singular_values"],
        energy_ratio=model["energy_ratio"],
        dx=model.get("dx", 1.0),
        dy=model.get("dy", 1.0),
    )
    return path


def _load_pod_model(path: Path) -> Dict[str, Array]:
    with np.load(path, allow_pickle=False) as data:
        model = {
            "Ud": data["Ud"],
            "mean": data["mean"],
            "singular_values": data["singular_values"],
            "energy_ratio": data["energy_ratio"],
            "dx": float(data["dx"]),
            "dy": float(data["dy"]),
        }
    return model


def _save_grid_meta(path: Path, meta: Dict[str, Array]) -> Path:
    clean_meta = {key: np.asarray(val) for key, val in meta.items()}
    np.savez(path, **clean_meta)
    return path


def _mass(res: Array, dx: float, dy: float) -> Array:
    return np.sum(res, axis=1) * dx * dy


def _train_from_density(
    Rho: Array,
    grid_meta: Dict[str, Array],
    energy_keep: float,
    w: int,
    ridge_lambda: float,
    train_frac: float,
    seed: int,
    out_dir: str,
) -> Dict[str, float]:
    """Shared training routine operating directly on density snapshots."""

    out_path = ensure_dir(out_dir)
    t_start = time.time()

    Rho = np.asarray(Rho, dtype=float)
    if Rho.ndim != 2:
        raise ValueError("Rho must be a (T, nc) array")

    grid_meta_path = Path(out_path) / "kde_grid.npz"
    _save_grid_meta(grid_meta_path, grid_meta)

    dx = float(grid_meta["dx"])
    dy = float(grid_meta["dy"])
    nx = int(grid_meta["nx"])
    ny = int(grid_meta["ny"])
    Lx = float(grid_meta["Lx"])
    Ly = float(grid_meta["Ly"])

    T = Rho.shape[0]
    if T <= w:
        raise ValueError("Not enough frames to seed the VAR model")

    masses = _mass(Rho, dx=dx, dy=dy)
    max_mass_err = float(np.max(np.abs(masses - 1.0)))
    if max_mass_err > 1e-6:
        Rho = Rho / masses[:, None]
        masses = _mass(Rho, dx=dx, dy=dy)
        max_mass_err = float(np.max(np.abs(masses - 1.0)))

    train_len = max(w + 1, int(math.floor(T * train_frac)))
    train_len = min(train_len, T)
    val_len = T - train_len
    Rho_train = Rho[:train_len]
    Rho_val = Rho[train_len:] if val_len > 0 else np.empty((0, Rho.shape[1]))

    pod_model = fit_pod(Rho_train, energy_keep=energy_keep, dx=dx, dy=dy)
    pod_rank = pod_model["Ud"].shape[1]
    energy_captured = float(pod_model["energy_ratio"][pod_rank - 1]) if pod_rank > 0 else 0.0

    Y_train = restrict_movie(Rho_train, pod_model)
    if Y_train.shape[0] <= w:
        raise ValueError("Training window must exceed the VAR order")
    mvar_model = fit_mvar(Y_train, w=w, ridge_lambda=ridge_lambda)

    pod_path = Path(out_path) / "pod_model.npz"
    mvar_path = Path(out_path) / "mvar_model.npz"
    _save_pod_model(pod_path, pod_model)
    save_mvar(mvar_path, mvar_model)

    recon_train = np.array([lift_pod(y, pod_model) for y in Y_train])
    err_weight = dx * dy
    train_recon_errors = np.sqrt(np.sum((recon_train - Rho_train) ** 2, axis=1) * err_weight)

    if val_len > 0:
        Y_val = restrict_movie(Rho_val, pod_model)
        recon_val = np.array([lift_pod(y, pod_model) for y in Y_val])
        val_recon_errors = np.sqrt(np.sum((recon_val - Rho_val) ** 2, axis=1) * err_weight)
    else:
        Y_val = np.empty((0, pod_rank))
        recon_val = np.empty_like(Rho_val)
        val_recon_errors = np.array([], dtype=float)

    latent_val_mse = float("nan")
    if Y_val.shape[0] > w:
        preds = []
        targets = []
        for idx in range(w, Y_val.shape[0]):
            history = Y_val[idx - w : idx]
            preds.append(forecast_step(history, mvar_model))
            targets.append(Y_val[idx])
        preds_arr = np.stack(preds)
        targets_arr = np.stack(targets)
        latent_val_mse = float(np.mean((preds_arr - targets_arr) ** 2))

    samples_dir = ensure_dir(Path(out_path) / "samples")
    sample_idx = np.unique(
        np.clip(
            np.array([0, train_len // 2, max(train_len - 1, 0), T - 1], dtype=int),
            0,
            T - 1,
        )
    )
    Xc = grid_meta["Xc"].reshape(-1, 2)

    fig, axes = plt.subplots(len(sample_idx), 2, figsize=(6, 3 * len(sample_idx)), constrained_layout=True)
    if len(sample_idx) == 1:
        axes = np.array([axes])
    for row, idx in enumerate(sample_idx):
        rho_true = Rho[idx]
        rho_latent = restrict_movie(rho_true[None, :], pod_model)[0]
        rho_recon = lift_pod(rho_latent, pod_model)
        axes[row, 0].imshow(
            _reshape_density(rho_true, nx, ny),
            extent=(0, Lx, 0, Ly),
            origin="lower",
            cmap="viridis",
        )
        axes[row, 0].set_title(f"t={idx} true")
        axes[row, 1].imshow(
            _reshape_density(rho_recon, nx, ny),
            extent=(0, Lx, 0, Ly),
            origin="lower",
            cmap="viridis",
        )
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 1].set_title(f"t={idx} recon")
    fig.savefig(samples_dir / "density_panel.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(np.arange(train_recon_errors.size), train_recon_errors, label="train")
    if val_recon_errors.size:
        ax.plot(np.arange(train_len, train_len + val_recon_errors.size), val_recon_errors, label="val")
    ax.set_xlabel("frame")
    ax.set_ylabel("L2 error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(samples_dir / "pod_recon_error.png", dpi=150)
    plt.close(fig)

    Y_full = restrict_movie(Rho, pod_model)
    fig, ax = plt.subplots(figsize=(6, 3))
    for j in range(pod_rank):
        ax.plot(Y_full[:, j], label=f"y{j}")
    ax.axvline(train_len - 1, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("frame")
    ax.set_ylabel("latent value")
    if pod_rank <= 6:
        ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(samples_dir / "latent_series.png", dpi=150)
    plt.close(fig)

    seed_path = Path(out_path) / "seed_lastw.npz"
    np.savez(seed_path, Rho=Rho[-w:])

    stats = {
        "T": int(T),
        "train_len": int(train_len),
        "val_len": int(val_len),
        "pod_rank": int(pod_rank),
        "pod_energy": float(energy_captured),
        "train_recon_l2_mean": float(train_recon_errors.mean()),
        "val_recon_l2_mean": float(val_recon_errors.mean()) if val_recon_errors.size else float("nan"),
        "latent_val_mse": latent_val_mse,
        "max_mass_error": max_mass_err,
        "ridge_lambda": float(ridge_lambda),
        "w": int(w),
        "energy_keep": float(energy_keep),
        "train_frac": float(train_frac),
        "seed": int(seed),
        "duration_sec": float(time.time() - t_start),
    }

    with (Path(out_path) / "train_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    print(
        f"POD rank {pod_rank} capturing {energy_captured:.4f} energy. "
        f"Train recon L2 mean {stats['train_recon_l2_mean']:.3e}."
    )
    if not np.isnan(stats["val_recon_l2_mean"]):
        print(
            f"Validation recon L2 mean {stats['val_recon_l2_mean']:.3e}. "
            f"Latent one-step MSE {latent_val_mse:.3e}."
        )
    print(f"Artifacts saved to {out_path}")

    return stats


def train_from_trajectories(
    traj_npz: str,
    Lx: float,
    Ly: float,
    bc: str,
    nx: int,
    ny: int,
    hx: float,
    hy: float,
    energy_keep: float,
    w: int,
    ridge_lambda: float,
    train_frac: float,
    seed: int,
    out_dir: str,
) -> Dict[str, float]:
    """Train the KDE → POD → MVAR latent pipeline from trajectories."""

    with np.load(traj_npz, allow_pickle=False) as data:
        if "x" not in data:
            raise KeyError("Trajectory archive must contain array 'x'")
        X_all = data["x"]

    Rho, grid_meta = trajectories_to_density_movie(
        X_all=X_all,
        Lx=Lx,
        Ly=Ly,
        nx=nx,
        ny=ny,
        hx=hx,
        hy=hy,
        bc=bc,
    )
    return _train_from_density(
        Rho=Rho,
        grid_meta=grid_meta,
        energy_keep=energy_keep,
        w=w,
        ridge_lambda=ridge_lambda,
        train_frac=train_frac,
        seed=seed,
        out_dir=out_dir,
    )


def train_from_heatmap_npz(
    heatmap_npz: str,
    energy_keep: float,
    w: int,
    ridge_lambda: float,
    train_frac: float,
    seed: int,
    out_dir: str,
) -> Dict[str, float]:
    """Train latent models from a saved heatmap trajectory archive."""

    with np.load(heatmap_npz, allow_pickle=False) as data:
        if "Rho" in data:
            Rho = data["Rho"]
        elif "rho" in data:
            Rho = data["rho"]
        else:
            raise KeyError("Heatmap archive must contain array 'Rho' or 'rho'")

        meta_keys = ["Xc", "dx", "dy", "nx", "ny", "Lx", "Ly", "hx", "hy", "bc"]
        grid_meta = {}
        for key in meta_keys:
            if key not in data:
                raise KeyError(f"Heatmap archive missing key '{key}'")
            grid_meta[key] = data[key]

    return _train_from_density(
        Rho=Rho,
        grid_meta=grid_meta,
        energy_keep=energy_keep,
        w=w,
        ridge_lambda=ridge_lambda,
        train_frac=train_frac,
        seed=seed,
        out_dir=out_dir,
    )


def forecast(
    pod_model_path: str,
    mvar_model_path: str,
    seed_frames_npz: str,
    steps: int,
    grid_meta_npz: str,
    out_dir: str,
    true_npz: str | None = None,
    fps: int = 20,
    make_movies: bool = True,
) -> Dict[str, float]:
    """Generate density forecasts using trained POD and MVAR models."""

    out_path = ensure_dir(out_dir)

    pod_model = _load_pod_model(Path(pod_model_path))
    mvar_model = load_mvar(Path(mvar_model_path))
    with np.load(seed_frames_npz, allow_pickle=False) as data:
        seed_rho = data["Rho"]
    with np.load(grid_meta_npz, allow_pickle=False) as data:
        grid_meta = {key: data[key] for key in data.files}

    dx = float(grid_meta["dx"])
    dy = float(grid_meta["dy"])
    nx = int(grid_meta["nx"])
    ny = int(grid_meta["ny"])
    Lx = float(grid_meta["Lx"])
    Ly = float(grid_meta["Ly"])
    Xc = grid_meta["Xc"]

    if seed_rho.shape[0] != mvar_model["w"]:
        raise ValueError("Seed archive must contain the last w density frames")

    Y_seed = restrict_movie(seed_rho, pod_model)
    forecasts_latent = rollout(Y_seed, steps=steps, model=mvar_model)

    Rho_hat = np.array([lift_pod(y, pod_model) for y in forecasts_latent])
    if Rho_hat.shape[0] != steps:
        raise RuntimeError("Forecast length mismatch")

    mass_err = np.max(np.abs(_mass(Rho_hat, dx=dx, dy=dy) - 1.0))
    if mass_err > 1e-6:
        raise RuntimeError("Forecast densities violate mass conservation tolerance")

    forecast_npz = Path(out_path) / "forecast.npz"
    np.savez(forecast_npz, Rho_hat=Rho_hat, Y_hat=forecasts_latent)

    stats: Dict[str, float | str] = {"steps": int(steps), "mass_error": float(mass_err), "forecast_npz": str(forecast_npz)}

    pred_anim_path = None
    if make_movies:
        try:
            pred_anim_path = animate_heatmap_movie(
                Rho=Rho_hat,
                Xc=Xc,
                dx=dx,
                dy=dy,
                Lx=Lx,
                Ly=Ly,
                out_path=str(Path(out_path) / "pred_heatmap"),
                fps=fps,
                cmap="magma",
                title="Forecast",
            )
            stats["pred_animation"] = pred_anim_path
        except Exception as exc:  # pragma: no cover - animation optional
            print(f"[WARN] Failed to write predicted animation: {exc}")

    metrics_payload = None
    if true_npz is not None:
        with np.load(true_npz, allow_pickle=False) as data:
            if "Rho" in data:
                R_true = data["Rho"]
            elif "rho" in data:
                R_true = data["rho"]
            else:
                raise KeyError("True archive must contain array 'Rho' or 'rho'")
            times = data.get("times")
        if R_true.shape != Rho_hat.shape:
            raise ValueError("True continuation must match forecast shape")

        if make_movies:
            try:
                true_anim_path = animate_heatmap_movie(
                    Rho=R_true,
                    Xc=Xc,
                    dx=dx,
                    dy=dy,
                    Lx=Lx,
                    Ly=Ly,
                    out_path=str(Path(out_path) / "true_heatmap"),
                    fps=fps,
                    cmap="magma",
                    title="True",
                )
                stats["true_animation"] = true_anim_path
            except Exception as exc:  # pragma: no cover
                print(f"[WARN] Failed to write true animation: {exc}")

            try:
                compare_path = animate_side_by_side(
                    R_true=R_true,
                    R_pred=Rho_hat,
                    Xc=Xc,
                    dx=dx,
                    dy=dy,
                    Lx=Lx,
                    Ly=Ly,
                    out_path=str(Path(out_path) / "compare_heatmap"),
                    fps=fps,
                )
                stats["compare_animation"] = compare_path
            except Exception as exc:  # pragma: no cover
                print(f"[WARN] Failed to write comparison animation: {exc}")

        metrics_payload = series_metrics(
            R_true=R_true,
            R_pred=Rho_hat,
            times=times if isinstance(times, np.ndarray) and times.shape[0] == steps else None,
            dx=dx,
            dy=dy,
            grid_shape=(ny, nx),
        )
        json_path, csv_path = save_metrics_json_csv(out_path, metrics_payload)
        stats["metrics_json"] = json_path
        stats["metrics_csv"] = csv_path
        aggregate = metrics_payload.get("aggregate", {})
        for key in ("rmse_mean", "rmse_median", "r2_mean"):
            if key in aggregate:
                stats[key] = float(aggregate[key])
        stats["t_tol"] = float(metrics_payload.get("t_tol", float("nan")))

    print(f"Forecast complete. Steps: {steps}, max mass error {mass_err:.3e}")
    if metrics_payload is not None:
        agg = metrics_payload.get("aggregate", {})
        rmse_mean = agg.get("rmse_mean")
        r2_mean = agg.get("r2_mean")
        if rmse_mean is not None and r2_mean is not None:
            print(f"Mean RMSE {rmse_mean:.3e}, mean R2 {r2_mean:.3f}")

    return stats



__all__ = ["train_from_trajectories", "train_from_heatmap_npz", "forecast"]
