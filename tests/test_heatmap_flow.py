import json
from pathlib import Path

import numpy as np

from wsindy_manifold.latent.kde import make_grid
from wsindy_manifold.latent.flow import train_from_heatmap_npz, forecast


def _make_heatmap_npz(path: Path, T: int, nx: int, ny: int, Lx: float, Ly: float) -> np.ndarray:
    Xc, dx, dy = make_grid(Lx, Ly, nx, ny)
    rng = np.random.default_rng(123)
    Rho = rng.random((T, nx * ny))
    mass = Rho.sum(axis=1, keepdims=True) * dx * dy
    Rho = Rho / mass
    payload = {
        "Rho": Rho,
        "Xc": Xc,
        "dx": np.array(dx),
        "dy": np.array(dy),
        "nx": np.array(nx),
        "ny": np.array(ny),
        "Lx": np.array(Lx),
        "Ly": np.array(Ly),
        "hx": np.array(0.1),
        "hy": np.array(0.1),
        "bc": np.array("periodic"),
        "times": np.linspace(0.0, T - 1, T),
    }
    np.savez(path, **payload)
    return Rho


def test_train_from_heatmap_and_forecast(tmp_path):
    heatmap_path = tmp_path / "heatmap.npz"
    T, nx, ny = 10, 6, 5
    Lx, Ly = 3.0, 2.0
    Rho = _make_heatmap_npz(heatmap_path, T, nx, ny, Lx, Ly)

    out_dir = tmp_path / "artifacts"
    stats = train_from_heatmap_npz(
        heatmap_npz=str(heatmap_path),
        energy_keep=0.9,
        w=2,
        ridge_lambda=1e-6,
        train_frac=0.8,
        seed=42,
        out_dir=str(out_dir),
    )
    assert stats["pod_rank"] > 0
    assert (out_dir / "pod_model.npz").exists()
    assert (out_dir / "mvar_model.npz").exists()
    assert (out_dir / "kde_grid.npz").exists()
    assert (out_dir / "seed_lastw.npz").exists()

    steps = 3
    true_npz = tmp_path / "true_future.npz"
    np.savez(true_npz, Rho=Rho[-steps:], times=np.linspace(0.0, steps - 1, steps))

    forecast_dir = tmp_path / "forecast"
    forecast_stats = forecast(
        pod_model_path=str(out_dir / "pod_model.npz"),
        mvar_model_path=str(out_dir / "mvar_model.npz"),
        seed_frames_npz=str(out_dir / "seed_lastw.npz"),
        steps=steps,
        grid_meta_npz=str(out_dir / "kde_grid.npz"),
        out_dir=str(forecast_dir),
        true_npz=str(true_npz),
        make_movies=False,
    )

    assert forecast_stats["steps"] == steps
    assert Path(forecast_stats["forecast_npz"]).exists()
    assert "metrics_json" in forecast_stats
    metrics_path = Path(forecast_stats["metrics_json"])
    assert metrics_path.exists()
    with metrics_path.open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    assert "per_frame" in metrics
    assert len(metrics["per_frame"]) == steps
