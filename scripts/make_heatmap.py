"""Build KDE heatmaps and animations from trajectory data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wsindy_manifold.latent.kde import trajectories_to_density_movie
from wsindy_manifold.latent.anim import animate_heatmap_movie


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create heatmap movies from trajectories")
    parser.add_argument("--traj_npz", required=True, help="Trajectory npz with array 'x' of shape (T, N, 2)")
    parser.add_argument("--Lx", type=float, required=True)
    parser.add_argument("--Ly", type=float, required=True)
    parser.add_argument("--bc", choices=["periodic", "reflecting"], default="periodic")
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--hx", type=float, required=True)
    parser.add_argument("--hy", type=float, required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--video_name", default="true_heatmap")
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.traj_npz, allow_pickle=False) as data:
        if "x" not in data:
            raise KeyError("Trajectory file must contain 'x'")
        X_all = data["x"]
        times = data.get("times")

    Rho, meta = trajectories_to_density_movie(
        X_all=X_all,
        Lx=args.Lx,
        Ly=args.Ly,
        nx=args.nx,
        ny=args.ny,
        hx=args.hx,
        hy=args.hy,
        bc=args.bc,
    )

    payload = {"Rho": Rho, **meta}
    if times is not None:
        payload["times"] = times

    heatmap_path = out_dir / "heatmap_true.npz"
    np.savez(heatmap_path, **payload)

    movie_path = animate_heatmap_movie(
        Rho=Rho,
        Xc=meta["Xc"],
        dx=float(meta["dx"]),
        dy=float(meta["dy"]),
        Lx=float(meta["Lx"]),
        Ly=float(meta["Ly"]),
        out_path=str(out_dir / args.video_name),
        fps=args.fps,
        cmap="magma",
        title="True Heatmap",
    )

    print(f"Heatmap npz written to {heatmap_path}")
    print(f"Animation saved to {movie_path}")


if __name__ == "__main__":
    main()
