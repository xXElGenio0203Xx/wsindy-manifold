"""Quick plotting utilities for latent EF-ROM artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wsindy_manifold.latent.kde import trajectories_to_density_movie
from wsindy_manifold.latent.pod import restrict_movie, lift_pod


def _load_pod_model(path: str) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {
            "Ud": data["Ud"],
            "mean": data["mean"],
            "singular_values": data["singular_values"],
            "energy_ratio": data["energy_ratio"],
            "dx": float(data["dx"]),
            "dy": float(data["dy"]),
        }


def _parse_frames(frames: str, T: int) -> List[int]:
    if not frames:
        return [0]
    result = []
    for item in frames.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0:
            idx = T + idx
        if not 0 <= idx < T:
            raise ValueError(f"Frame index {idx} out of bounds for T={T}")
        result.append(idx)
    return sorted(set(result))


def _reshape(rho: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return rho.reshape(ny, nx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot density frames and optional POD reconstructions")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--traj_npz", help="Trajectory npz containing array 'x'")
    group.add_argument("--density_npz", help="Density npz with array 'rho'")
    parser.add_argument("--Lx", type=float, default=None, help="Domain length in x (required for trajectories)")
    parser.add_argument("--Ly", type=float, default=None, help="Domain length in y (required for trajectories)")
    parser.add_argument("--bc", choices=["periodic", "reflecting"], default="periodic")
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--hx", type=float, default=0.5)
    parser.add_argument("--hy", type=float, default=0.5)
    parser.add_argument("--pod_model", help="Optional POD model to compute reconstructions")
    parser.add_argument("--frames", default="0", help="Comma-separated frame indices (supports negatives)")
    parser.add_argument("--out_dir", default="artifacts/latent/plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.density_npz:
        with np.load(args.density_npz, allow_pickle=False) as data:
            rho_stack = data["rho"]
            xgrid = data.get("xgrid")
            ygrid = data.get("ygrid")
            dx = float(np.mean(np.diff(xgrid))) if xgrid is not None else 1.0
            dy = float(np.mean(np.diff(ygrid))) if ygrid is not None else 1.0
            Lx = dx * rho_stack.shape[1]
            Ly = dy * rho_stack.shape[2]
        T, nx, ny = rho_stack.shape
        Rho = rho_stack.reshape(T, nx * ny)
    else:
        if args.Lx is None or args.Ly is None:
            raise ValueError("Lx and Ly must be specified when using trajectories")
        with np.load(args.traj_npz, allow_pickle=False) as data:
            X_all = data["x"]
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
        nx = args.nx
        ny = args.ny
        dx = float(meta["dx"])
        dy = float(meta["dy"])
        Lx = args.Lx
        Ly = args.Ly
        T = Rho.shape[0]

    frame_indices = _parse_frames(args.frames, T)

    if args.pod_model:
        pod_model = _load_pod_model(args.pod_model)
        Y = restrict_movie(Rho, pod_model)
    else:
        pod_model = None
        Y = None

    fig, axes = plt.subplots(len(frame_indices), 2 if pod_model else 1, figsize=(5 * (2 if pod_model else 1), 4 * len(frame_indices)), constrained_layout=True)
    if len(frame_indices) == 1:
        axes = np.array([axes])
    if axes.ndim == 1:
        axes = axes[:, None] if pod_model else axes[:, None]

    for row, idx in enumerate(frame_indices):
        rho_flat = Rho[idx]
        axes[row, 0].imshow(
            _reshape(rho_flat, nx, ny),
            extent=(0, Lx, 0, Ly),
            origin="lower",
            cmap="viridis",
        )
        axes[row, 0].set_title(f"frame {idx} true")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        if pod_model:
            rho_recon = lift_pod(Y[idx], pod_model)
            axes[row, 1].imshow(
                _reshape(rho_recon, nx, ny),
                extent=(0, Lx, 0, Ly),
                origin="lower",
                cmap="viridis",
            )
            axes[row, 1].set_title(f"frame {idx} recon")
            axes[row, 1].set_xticks([])
            axes[row, 1].set_yticks([])

    fig.savefig(out_dir / "frames.png", dpi=150)
    plt.close(fig)

    if pod_model is not None:
        fig, ax = plt.subplots(figsize=(6, 3))
        for j in range(pod_model["Ud"].shape[1]):
            ax.plot(Y[:, j], label=f"y{j}")
        ax.set_xlabel("frame")
        ax.set_ylabel("latent")
        if pod_model["Ud"].shape[1] <= 6:
            ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "latents.png", dpi=150)
        plt.close(fig)

    print(f"Plots written to {out_dir}")


if __name__ == "__main__":
    main()
