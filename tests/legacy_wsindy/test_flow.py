from pathlib import Path

import numpy as np

from wsindy_manifold.latent.flow import forecast, train_from_trajectories


def test_latent_flow_end_to_end(tmp_path):
    Lx, Ly = 4.0, 3.0
    nx, ny = 12, 10
    hx = hy = 0.2
    T, N = 16, 20
    rng = np.random.default_rng(2)
    traj = rng.uniform([0.0, 0.0], [Lx, Ly], size=(T, N, 2))
    traj_path = tmp_path / "traj.npz"
    np.savez(traj_path, x=traj)

    out_dir = tmp_path / "artifacts"
    stats = train_from_trajectories(
        traj_npz=str(traj_path),
        Lx=Lx,
        Ly=Ly,
        bc="periodic",
        nx=nx,
        ny=ny,
        hx=hx,
        hy=hy,
        energy_keep=0.95,
        w=2,
        ridge_lambda=1e-6,
        train_frac=0.75,
        seed=1,
        out_dir=str(out_dir),
    )

    assert stats["pod_rank"] > 0
    pod_path = out_dir / "pod_model.npz"
    mvar_path = out_dir / "mvar_model.npz"
    grid_meta_path = out_dir / "kde_grid.npz"
    seed_path = out_dir / "seed_lastw.npz"
    assert pod_path.exists()
    assert mvar_path.exists()
    assert grid_meta_path.exists()
    assert seed_path.exists()

    forecast_dir = tmp_path / "forecast"
    forecast_stats = forecast(
        pod_model_path=str(pod_path),
        mvar_model_path=str(mvar_path),
        seed_frames_npz=str(seed_path),
        steps=3,
        grid_meta_npz=str(grid_meta_path),
        out_dir=str(forecast_dir),
        make_movies=False,
    )
    assert forecast_stats["steps"] == 3
    forecast_npz = forecast_dir / "forecast.npz"
    assert forecast_npz.exists()
    with np.load(forecast_npz) as data:
        assert data["Rho_hat"].shape[0] == 3
