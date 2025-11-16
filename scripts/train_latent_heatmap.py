"""Train latent EF-ROM models directly on saved heatmap movies."""

from __future__ import annotations

import argparse

from wsindy_manifold.latent.flow import train_from_heatmap_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train latent EF-ROM on heatmap data")
    parser.add_argument("--heatmap_npz", required=True, help="NPZ produced by make_heatmap.py")
    parser.add_argument("--energy_keep", type=float, default=0.99)
    parser.add_argument("--w", type=int, default=4)
    parser.add_argument("--ridge_lambda", type=float, default=1e-6)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stats = train_from_heatmap_npz(
        heatmap_npz=args.heatmap_npz,
        energy_keep=args.energy_keep,
        w=args.w,
        ridge_lambda=args.ridge_lambda,
        train_frac=args.train_frac,
        seed=args.seed,
        out_dir=args.out_dir,
    )

    pod_rank = stats.get("pod_rank")
    energy = stats.get("pod_energy")
    print(
        f"Training complete. POD rank {pod_rank}, energy {energy:.4f}, artifacts at {args.out_dir}"
    )


if __name__ == "__main__":
    main()
