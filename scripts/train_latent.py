"""Train latent EF-ROM models from rectangular trajectories."""

from __future__ import annotations

import argparse
from datetime import datetime

from wsindy_manifold.latent.flow import train_from_trajectories


def _default_out_dir() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"artifacts/latent/run_{ts}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train latent EF-ROM pipeline from trajectories")
    parser.add_argument("--traj_npz", required=True, help="Path to trajectory npz file containing array 'x'")
    parser.add_argument("--Lx", type=float, required=True)
    parser.add_argument("--Ly", type=float, required=True)
    parser.add_argument("--bc", choices=["periodic", "reflecting"], default="periodic")
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--hx", type=float, required=True)
    parser.add_argument("--hy", type=float, required=True)
    parser.add_argument("--energy_keep", type=float, default=0.99)
    parser.add_argument("--w", type=int, default=4)
    parser.add_argument("--ridge_lambda", type=float, default=1e-6)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or _default_out_dir()

    stats = train_from_trajectories(
        traj_npz=args.traj_npz,
        Lx=args.Lx,
        Ly=args.Ly,
        bc=args.bc,
        nx=args.nx,
        ny=args.ny,
        hx=args.hx,
        hy=args.hy,
        energy_keep=args.energy_keep,
        w=args.w,
        ridge_lambda=args.ridge_lambda,
        train_frac=args.train_frac,
        seed=args.seed,
        out_dir=out_dir,
    )

    pod_rank = stats.get("pod_rank")
    print(f"Training finished. POD rank {pod_rank}, artifacts in {out_dir}")


if __name__ == "__main__":
    main()
