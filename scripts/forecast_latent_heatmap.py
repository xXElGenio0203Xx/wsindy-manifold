"""Forecast heatmap trajectories using trained latent models."""

from __future__ import annotations

import argparse

from wsindy_manifold.latent.flow import forecast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forecast heatmaps using latent EF-ROM")
    parser.add_argument("--pod_model", required=True)
    parser.add_argument("--mvar_model", required=True)
    parser.add_argument("--seed_npz", required=True, help="NPZ with array 'Rho' of last w frames")
    parser.add_argument("--grid_meta", required=True, help="Grid metadata npz from training")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--true_npz", help="Optional npz containing true future heatmaps for comparison")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--no_movies", action="store_true", help="Skip writing animations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stats = forecast(
        pod_model_path=args.pod_model,
        mvar_model_path=args.mvar_model,
        seed_frames_npz=args.seed_npz,
        steps=args.steps,
        grid_meta_npz=args.grid_meta,
        out_dir=args.out_dir,
        true_npz=args.true_npz,
        fps=args.fps,
        make_movies=not args.no_movies,
    )

    print(f"Forecast artifacts saved to {args.out_dir}")
    if "rmse_mean" in stats and stats["rmse_mean"] is not None:
        print(f"Mean RMSE {stats['rmse_mean']:.3e}, mean R2 {stats.get('r2_mean', float('nan')):.3f}")


if __name__ == "__main__":
    main()
