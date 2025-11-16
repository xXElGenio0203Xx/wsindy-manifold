"""Create side-by-side heatmap comparison videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wsindy_manifold.latent.anim import animate_side_by_side


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two heatmap sequences")
    parser.add_argument("--true_npz", required=True)
    parser.add_argument("--pred_npz", required=True)
    parser.add_argument("--grid_meta", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--no_error", action="store_true", help="Hide absolute error panel")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with np.load(args.true_npz, allow_pickle=False) as data:
        if "Rho" in data:
            R_true = data["Rho"]
        elif "rho" in data:
            R_true = data["rho"]
        else:
            raise KeyError("True npz missing 'Rho' or 'rho'")

    with np.load(args.pred_npz, allow_pickle=False) as data:
        if "Rho_hat" in data:
            R_pred = data["Rho_hat"]
        elif "Rho" in data:
            R_pred = data["Rho"]
        else:
            raise KeyError("Pred npz missing forecast array")

    if R_true.shape != R_pred.shape:
        raise ValueError("True and predicted heatmaps must have the same shape")

    with np.load(args.grid_meta, allow_pickle=False) as data:
        grid_meta = {key: data[key] for key in data.files}

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result_path = animate_side_by_side(
        R_true=R_true,
        R_pred=R_pred,
        Xc=grid_meta["Xc"],
        dx=float(grid_meta["dx"]),
        dy=float(grid_meta["dy"]),
        Lx=float(grid_meta["Lx"]),
        Ly=float(grid_meta["Ly"]),
        out_path=str(out_path),
        fps=args.fps,
        include_error=not args.no_error,
    )

    print(f"Comparison animation saved to {result_path}")


if __name__ == "__main__":
    main()
