"""Forecast densities using trained latent EF-ROM models."""

from __future__ import annotations

import argparse
from datetime import datetime

from wsindy_manifold.latent.flow import forecast


def _default_out_dir() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"artifacts/latent/forecast_{ts}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forecast densities with a latent EF-ROM")
    parser.add_argument("--pod_model", required=True, help="Path to trained POD model npz")
    parser.add_argument("--mvar_model", required=True, help="Path to trained MVAR model npz")
    parser.add_argument("--seed_npz", required=True, help="NPZ with array 'Rho' of the last w density frames")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--grid_meta", required=True, help="Grid metadata npz produced during training")
    parser.add_argument("--out_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or _default_out_dir()

    stats = forecast(
        pod_model_path=args.pod_model,
        mvar_model_path=args.mvar_model,
        seed_frames_npz=args.seed_npz,
        steps=args.steps,
        grid_meta_npz=args.grid_meta,
        out_dir=out_dir,
    )
    print(f"Forecast saved to {out_dir}. Mass error {stats['mass_error']:.3e}")


if __name__ == "__main__":
    main()
