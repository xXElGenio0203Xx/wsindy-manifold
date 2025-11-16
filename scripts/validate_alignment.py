"""Validate the alignment implementation against a gold Vicsek reference."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wsindy_manifold.align_check import (
    angle_diff_mean,
    neighbor_finder_ball,
    order_parameter,
    step_yours_vs_gold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Vicsek-style alignment")
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--Lx", type=float, default=20.0)
    parser.add_argument("--Ly", type=float, default=20.0)
    parser.add_argument("--bc", choices=["periodic", "reflecting"], default="periodic")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--mu_r", type=float, default=0.5)
    parser.add_argument("--lV", type=float, default=1.5)
    parser.add_argument("--Dtheta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", type=str, default="")
    return parser.parse_args()


def random_headings(rng: np.random.Generator, n: int) -> np.ndarray:
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return np.column_stack((np.cos(angles), np.sin(angles)))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    x = rng.uniform(low=[0.0, 0.0], high=[args.Lx, args.Ly], size=(args.N, 2))
    p0 = random_headings(rng, args.N)

    psi_yours = []
    psi_gold = []
    angle_gaps = []
    neighbor_counts = []

    p_curr = p0.copy()
    p_ref = p0.copy()

    for step in range(args.steps):
        psi_yours.append(order_parameter(p_curr))
        psi_gold.append(order_parameter(p_ref))
        angle_gaps.append(angle_diff_mean(p_curr, p_ref))

        p_curr, p_ref = step_yours_vs_gold(
            x,
            p_curr,
            args.Lx,
            args.Ly,
            args.bc,
            args.mu_r,
            args.lV,
            args.Dtheta,
            args.dt,
            seed=rng.integers(0, 2**32 - 1),
        )

        neigh = neighbor_finder_ball(x, args.Lx, args.Ly, args.bc, args.lV)
        neighbor_counts.append(np.mean([len(n) for n in neigh]))

    psi_yours.append(order_parameter(p_curr))
    psi_gold.append(order_parameter(p_ref))
    angle_gaps.append(angle_diff_mean(p_curr, p_ref))

    max_angle_diff = max(angle_gaps)
    psi_diff = abs(psi_yours[-1] - psi_gold[-1])

    print(f"Max mean angle difference: {max_angle_diff:.3e} rad")
    print(f"Final ψ difference: {psi_diff:.3e}")
    print(f"Average neighbour count: {np.mean(neighbor_counts):.2f}")

    passed = True
    if args.Dtheta == 0.0 and max_angle_diff > 1e-3:
        passed = False
        print("FAIL: deterministic comparison exceeds tolerance (1e-3 rad)")
    if psi_diff > 1e-2:
        passed = False
        print("FAIL: final ψ difference exceeds tolerance (1e-2)")
    if passed:
        print("PASS: alignment implementation matches gold reference within tolerances.")

    if args.plot:
        plot_path = Path(args.plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        t = np.arange(len(psi_yours)) * args.dt
        fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax[0].plot(t, psi_yours, label="yours")
        ax[0].plot(t, psi_gold, label="gold", linestyle="--")
        ax[0].set_ylabel("ψ")
        ax[0].legend()
        ax[1].plot(t, angle_gaps)
        ax[1].set_xlabel("time")
        ax[1].set_ylabel("mean |Δθ|")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
