"""Command-line interface for rectangular collective motion simulations."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config
from .density import hist2d_movie
from .dynamics import simulate
from .io import save_csv, save_npz, save_run_metadata
from .metrics import compute_timeseries


def _parse_overrides(unknown: List[str]) -> List[Tuple[str, str]]:
    """Convert unknown CLI args into key/value override tuples."""

    overrides: List[Tuple[str, str]] = []
    i = 0
    while i < len(unknown):
        key = unknown[i]
        if not key.startswith("--"):
            raise ValueError(f"Unrecognized argument '{key}'")
        if i + 1 >= len(unknown):
            raise ValueError(f"Missing value for override '{key}'")
        value = unknown[i + 1]
        overrides.append((key[2:], value))
        i += 2
    return overrides


def _convert_value(value: str):
    """Interpret an override value as JSON when possible for type fidelity."""

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _footer_text(config: Dict) -> str:
    """Create a short text summary of key simulation parameters."""

    params = config["params"]
    sim = config["sim"]
    return (
        f"N={sim['N']}  Lx={sim['Lx']}  Ly={sim['Ly']}  bc={sim['bc']}\n"
        f"alpha={params['alpha']}  beta={params['beta']}  Cr={params['Cr']}  Ca={params['Ca']}\n"
        f"lr={params['lr']}  la={params['la']}  dt={sim['dt']}"
    )


def _plot_final(out_dir: Path, traj: np.ndarray, vel: np.ndarray, config: Dict) -> None:
    """Save a scatter/velocity quiver plot for the final frame of a run."""

    final_pos = traj[-1]
    final_vel = vel[-1]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(final_pos[:, 0], final_pos[:, 1], c="C0", s=10, alpha=0.7)
    ax.quiver(
        final_pos[:, 0],
        final_pos[:, 1],
        final_vel[:, 0],
        final_vel[:, 1],
        angles="xy",
        scale_units="xy",
        scale=10.0,
        color="C1",
        alpha=0.6,
    )
    ax.set_xlim(0, config["sim"]["Lx"])
    ax.set_ylim(0, config["sim"]["Ly"])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Final positions and velocities")
    fig.text(0.01, 0.01, _footer_text(config), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "traj_final.png", dpi=200)
    plt.close(fig)


def _plot_order_params(out_dir: Path, metrics_df: pd.DataFrame, config: Dict) -> None:
    """Write plots of macroscopic order parameters across the simulation."""

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.ravel()
    axes[0].plot(metrics_df["time"], metrics_df["polarization"], label="P")
    axes[0].set_ylabel("Polarization")
    axes[0].legend()

    axes[1].plot(metrics_df["time"], metrics_df["angular_momentum"], label="M_ang")
    axes[1].set_ylabel("Angular momentum")
    axes[1].legend()

    axes[2].plot(metrics_df["time"], metrics_df["abs_angular_momentum"], label="M_abs")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("|Angular momentum|")
    axes[2].legend()

    axes[3].plot(metrics_df["time"], metrics_df["dnn"], label="DNN")
    axes[3].set_xlabel("Time")
    axes[3].set_ylabel("Mean NN distance")
    axes[3].legend()

    fig.text(0.01, 0.01, _footer_text(config), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "order_params.png", dpi=200)
    plt.close(fig)


def _plot_speeds(out_dir: Path, vel: np.ndarray, times: np.ndarray, config: Dict) -> None:
    """Record distributions of particle speeds and energy through time."""

    speeds = np.linalg.norm(vel, axis=2)
    final = speeds[-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(final, bins=30, alpha=0.7, color="C0")
    ax.set_xlabel("Speed")
    ax.set_ylabel("Count")
    ax.set_title("Speed distribution (final frame)")
    fig.text(0.01, 0.01, _footer_text(config), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "speed_hist.png", dpi=200)
    plt.close(fig)

    energy = 0.5 * np.sum(speeds**2, axis=1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times, energy)
    ax.set_xlabel("Time")
    ax.set_ylabel("Kinetic energy")
    ax.set_title("Kinetic energy vs time")
    fig.text(0.01, 0.01, _footer_text(config), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "energy_time.png", dpi=200)
    plt.close(fig)


def _run_single(config: Dict) -> Dict:
    """Execute one simulation and persist configured outputs to disk."""

    cfg = deepcopy(config)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    results = simulate(cfg)

    metrics_df = compute_timeseries(
        results["traj"],
        results["vel"],
        results["times"],
        cfg["sim"]["Lx"],
        cfg["sim"]["Ly"],
        cfg["sim"]["bc"],
    )

    if cfg["outputs"].get("save_npz", True):
        save_npz(
            out_dir,
            "traj",
            x=results["traj"],
            v=results["vel"],
            times=results["times"],
            params=np.array(json.dumps(results["params"])),
            sim=np.array(json.dumps(results["sim"])),
        )

    if cfg["outputs"].get("save_csv", True):
        save_csv(out_dir, "metrics", metrics_df)

    if cfg["outputs"].get("plots", True):
        _plot_final(out_dir, results["traj"], results["vel"], cfg)
        _plot_order_params(out_dir, metrics_df, cfg)
        _plot_speeds(out_dir, results["vel"], results["times"], cfg)

    grid_cfg = cfg["outputs"].get("grid_density", {})
    if grid_cfg.get("enabled", True):
        hist2d_movie(
            results["traj"],
            results["times"],
            cfg["sim"]["Lx"],
            cfg["sim"]["Ly"],
            grid_cfg.get("nx", 128),
            grid_cfg.get("ny", 128),
            grid_cfg.get("bandwidth", 0.5),
            cfg["sim"]["bc"],
            out_dir,
            animate=cfg["outputs"].get("animate", True),
        )

    save_run_metadata(out_dir, cfg, results)

    return {
        "config": cfg,
        "results": results,
        "metrics": metrics_df,
        "out_dir": out_dir,
    }


def cmd_single(args: argparse.Namespace, overrides: List[Tuple[str, str]]) -> None:
    """Entry point for the ``single`` CLI command."""

    override_pairs = [(k, _convert_value(v)) for k, v in overrides]
    config = load_config(args.config, override_pairs)
    _run_single(config)


def cmd_grid(args: argparse.Namespace, overrides: List[Tuple[str, str]]) -> None:
    """Entry point for the ``grid`` CLI command that sweeps parameters."""

    override_pairs = [(k, _convert_value(v)) for k, v in overrides]
    config = load_config(args.config, override_pairs)
    base_cfg = deepcopy(config)

    grid_cfg = base_cfg.get("grid", {})
    Cr_values = grid_cfg.get("Cr", base_cfg["params"].get("Cr", [base_cfg["params"]["Cr"]]))
    Ca_values = grid_cfg.get("Ca", [base_cfg["params"].get("Ca", 1.0)])
    lr_values = grid_cfg.get("lr", base_cfg["params"].get("lr", [base_cfg["params"]["lr"]]))
    la_values = grid_cfg.get("la", [base_cfg["params"].get("la", 1.0)])
    reps = int(grid_cfg.get("reps", 1))

    records = []
    root_out = Path(base_cfg.get("out_dir", "outputs/grid"))
    counter = 0
    for Cr in np.atleast_1d(Cr_values):
        for Ca in np.atleast_1d(Ca_values):
            for lr in np.atleast_1d(lr_values):
                for la in np.atleast_1d(la_values):
                    for rep in range(reps):
                        counter += 1
                        cfg = deepcopy(base_cfg)
                        cfg["params"]["Cr"] = float(Cr)
                        cfg["params"]["Ca"] = float(Ca)
                        cfg["params"]["lr"] = float(lr)
                        cfg["params"]["la"] = float(la)
                        cfg["out_dir"] = str(root_out / f"Cr_{Cr}_Ca_{Ca}_lr_{lr}_la_{la}_rep_{rep}")
                        cfg["seed"] = base_cfg["seed"] + counter

                        run = _run_single(cfg)
                        metrics = run["metrics"]
                        last = metrics.iloc[-1]
                        records.append(
                            {
                                "Cr": float(Cr),
                                "Ca": float(Ca),
                                "lr": float(lr),
                                "la": float(la),
                                "rep": rep,
                                "out_dir": str(run["out_dir"]),
                                "polarization": float(last["polarization"]),
                                "angular_momentum": float(last["angular_momentum"]),
                                "abs_angular_momentum": float(last["abs_angular_momentum"]),
                                "dnn": float(last["dnn"]),
                            }
                        )

    manifest = pd.DataFrame.from_records(records)
    save_csv(root_out, "manifest", manifest)

    if not manifest.empty:
        summary = manifest.groupby(["Cr", "lr"]).mean(numeric_only=True).reset_index()
        for field in ["polarization", "abs_angular_momentum", "dnn"]:
            pivot = summary.pivot(index="lr", columns="Cr", values=field)
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(pivot, origin="lower", aspect="auto")
            ax.set_xlabel("Cr")
            ax.set_ylabel("lr")
            ax.set_title(f"Mean {field}")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(root_out / f"summary_{field}.png", dpi=200)
            plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level CLI argument parser."""

    parser = argparse.ArgumentParser(prog="rectsim")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single", help="Run a single simulation")
    single.add_argument("--config", required=True, help="Path to configuration YAML")

    grid = subparsers.add_parser("grid", help="Run a parameter grid")
    grid.add_argument("--config", required=True, help="Path to grid configuration YAML")

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    """Run the command-line interface using the provided argv sequence."""

    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    overrides = _parse_overrides(unknown)

    if args.command == "single":
        cmd_single(args, overrides)
    elif args.command == "grid":
        cmd_grid(args, overrides)
    else:  # pragma: no cover - defensive
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
