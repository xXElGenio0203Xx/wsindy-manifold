"""Command-line interface and plotting utilities for the simulator.

This module provides the command-line entry points and the helper
functions that write plots and outputs for single runs and parameter
grids.

Key responsibilities
- parse CLI overrides and convert them into dotted-key overrides
- run single simulations and persist outputs (NPZ, CSV, plots, movies)
- run grid sweeps and produce summary heatmaps

How this module fits in
- It uses :mod:`rectsim.config` to load validated configurations.
- It calls :func:`rectsim.dynamics.simulate` to produce trajectories.
- It uses :mod:`rectsim.metrics` and :mod:`rectsim.density` to compute
    time-series and density movies, and :mod:`rectsim.io` to save files.

Why these helpers exist
- Keeping plotting and file I/O here keeps the simulation core
    (``dynamics`` and ``morse``) free of side-effects, which makes
    the numerical code easier to test and reuse.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config
from .density import density_movie_kde, hist2d_movie
from .dynamics import simulate
from .io import save_csv, save_npz, save_run_metadata
from .metrics import (
    compute_timeseries,
    mean_relative_error,
    r2,
    rmse,
    tolerance_horizon,
)
from .vicsek_discrete import simulate_vicsek


def _parse_overrides(unknown: List[str]) -> List[Tuple[str, str]]:
    """Convert unknown CLI args into key/value override tuples.

    How it works
    ------------
    The CLI accepts extra arguments like ``--sim.N 50``. The parser
    receives them as an ``unknown`` list; this helper consumes the
    list two items at a time and returns pairs without the leading
    ``--`` so callers can pass them to ``load_config``.
    """

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
    """Interpret an override value as JSON when possible for type fidelity.

    This converts strings that look like JSON (numbers, booleans, lists,
    objects) into Python objects. If parsing fails the raw string is
    returned. This preserves types for overrides (e.g., ``--sim.N 50``
    yields an int).
    """

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
    # read plotting options from config with sensible defaults
    plot_opts = config.get("outputs", {}).get("plot_options", {})
    marker_size = plot_opts.get("traj_marker_size", 20)
    quiver_scale = plot_opts.get("traj_quiver_scale", 3.0)
    quiver_width = plot_opts.get("traj_quiver_width", 0.004)
    quiver_alpha = plot_opts.get("traj_quiver_alpha", 0.8)

    ax.scatter(final_pos[:, 0], final_pos[:, 1], c="C0", s=marker_size, alpha=0.8)
    ax.quiver(
        final_pos[:, 0],
        final_pos[:, 1],
        final_vel[:, 0],
        final_vel[:, 1],
        angles="xy",
        scale_units="xy",
        scale=quiver_scale,
        width=quiver_width,
        color="C1",
        alpha=quiver_alpha,
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


def _vicsek_footer_text(cfg: Dict) -> str:
    noise = cfg.get("noise", {})
    parts = [
        f"N={cfg['N']}",
        f"Lx={cfg['Lx']}",
        f"Ly={cfg['Ly']}",
        f"bc={cfg['bc']}",
        f"v0={cfg['v0']}",
        f"R={cfg['R']}",
        f"noise={noise.get('kind', 'gaussian')}",
    ]
    if noise.get("kind", "gaussian") == "gaussian":
        parts.append(f"sigma={noise.get('sigma', 0.0)}")
    else:
        parts.append(f"eta={noise.get('eta', 0.0)}")
    return "  ".join(parts)


def _plot_vicsek_final(
    out_dir: Path,
    traj: np.ndarray,
    vel: np.ndarray,
    cfg: Dict,
    plot_opts: Dict,
) -> None:
    final_pos = traj[-1]
    final_vel = vel[-1]
    fig, ax = plt.subplots(figsize=(6, 6))

    marker_size = plot_opts.get("traj_marker_size", 20)
    draw_quiver = plot_opts.get("traj_quiver", True)
    quiver_scale = plot_opts.get("traj_quiver_scale", 3.0)
    quiver_width = plot_opts.get("traj_quiver_width", 0.004)
    quiver_alpha = plot_opts.get("traj_quiver_alpha", 0.8)

    ax.scatter(final_pos[:, 0], final_pos[:, 1], c="C0", s=marker_size, alpha=0.8)
    if draw_quiver:
        ax.quiver(
            final_pos[:, 0],
            final_pos[:, 1],
            final_vel[:, 0],
            final_vel[:, 1],
            angles="xy",
            scale_units="xy",
            scale=quiver_scale,
            width=quiver_width,
            color="C1",
            alpha=quiver_alpha,
        )
    ax.set_xlim(0, cfg["Lx"])
    ax.set_ylim(0, cfg["Ly"])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Final positions and headings")
    fig.text(0.01, 0.01, _vicsek_footer_text(cfg), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "traj_final.png", dpi=200)
    plt.close(fig)


def _plot_vicsek_order_parameter(
    out_dir: Path,
    times: np.ndarray,
    psi: np.ndarray,
    cfg: Dict,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times, psi, label="psi")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\psi$")
    ax.set_title("Vicsek order parameter")
    ax.legend()
    fig.text(0.01, 0.01, _vicsek_footer_text(cfg), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "order_parameter.png", dpi=200)
    plt.close(fig)


def _run_vicsek_single(config: Dict) -> Dict:
    """Execute a discrete Vicsek simulation and persist outputs."""

    cfg = deepcopy(config)
    
    # Check if using new backend schema (has model_config, forces, noise sections)
    # vs old vicsek schema (only has vicsek section with flat params)
    using_backend_schema = (
        "model_config" in cfg or 
        "forces" in cfg or 
        ("noise" in cfg and "sim" in cfg)  # New schema has both
    )
    
    if not using_backend_schema:
        # OLD SCHEMA: config has "vicsek" key with flat parameters
        vicsek_cfg = deepcopy(cfg.get("vicsek", {}))
        if not vicsek_cfg:
            raise ValueError("Vicsek configuration missing under key 'vicsek'.")
        out_dir = Path(vicsek_cfg.get("out_dir", cfg.get("out_dir", "outputs/vicsek")))
        out_dir.mkdir(parents=True, exist_ok=True)
        results = simulate_vicsek(vicsek_cfg)
    else:
        # NEW SCHEMA: config has "sim", "model_config", "params", etc. (backend interface)
        from .vicsek_discrete import simulate_backend
        out_dir = Path(cfg.get("out_dir", "outputs/vicsek"))
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate RNG
        seed = cfg.get("seed", 42)
        rng = np.random.default_rng(seed)
        
        # Prepare backend config: rename model_config → model
        backend_cfg = deepcopy(cfg)
        if "model_config" in backend_cfg:
            backend_cfg["model"] = backend_cfg.pop("model_config")
        elif isinstance(backend_cfg.get("model"), str):
            # If model is still a string, create empty dict
            backend_cfg["model"] = {}
        
        # Call backend interface
        results = simulate_backend(backend_cfg, rng)
        
        # Extract results (backend returns different format)
        # Backend: {times, traj, vel, head, meta}
        # Old: {traj, headings, vel, times, psi}
        # Compute order parameter if not present
        if "psi" not in results:
            # Compute psi from headings
            headings = results.get("head", results.get("headings"))
            psi = np.linalg.norm(headings.mean(axis=1), axis=1)
            results["psi"] = psi
        
        # Map backend keys to old keys for compatibility
        if "head" in results and "headings" not in results:
            results["headings"] = results["head"]
    
    # Extract results (compatible with both schemas)
    traj = results["traj"]
    headings = results.get("headings", results.get("head"))
    vel = results["vel"]
    times = results["times"]
    psi = results["psi"]

    psi_df = pd.DataFrame({"time": times, "psi": psi})

    outputs_cfg = cfg.get("outputs", {})
    plot_opts = outputs_cfg.get("plot_options", {})
    
    # Get config parameters (handle both old and new schemas)
    if "vicsek" in cfg:
        vicsek_cfg = cfg["vicsek"]
    else:
        # Build vicsek_cfg dict from new schema for compatibility
        vicsek_cfg = {
            "N": cfg["sim"]["N"],
            "Lx": cfg["sim"]["Lx"],
            "Ly": cfg["sim"]["Ly"],
            "bc": cfg["sim"]["bc"],
            "T": cfg["sim"]["T"],
            "dt": cfg["sim"]["dt"],
            "v0": cfg.get("model_config", {}).get("speed", 0.5),
            "R": cfg.get("params", {}).get("R", 1.0),
            "out_dir": str(out_dir),
        }

    if outputs_cfg.get("save_npz", True):
        payload = {
            "x": traj,
            "headings": headings,
            "v": vel,
            "times": times,
            "psi": psi,
            "vicsek": np.array(json.dumps(vicsek_cfg)),
        }
        traj_npz = save_npz(out_dir, "traj", **payload)
        alias_path = traj_npz.with_name("trajectories.npz")
        np.savez(alias_path, **payload)

    if outputs_cfg.get("save_csv", True):
        save_csv(out_dir, "order_parameter", psi_df)

    if outputs_cfg.get("plots", True):
        _plot_vicsek_final(out_dir, traj, vel, vicsek_cfg, plot_opts)
        _plot_vicsek_order_parameter(out_dir, times, psi, vicsek_cfg)

    grid_cfg = outputs_cfg.get("grid_density", {})
    if grid_cfg.get("enabled", True):
        hist2d_movie(
            traj,
            times,
            vicsek_cfg["Lx"],
            vicsek_cfg["Ly"],
            grid_cfg.get("nx", 128),
            grid_cfg.get("ny", 128),
            grid_cfg.get("bandwidth", 0.5),
            vicsek_cfg["bc"],
            out_dir,
            animate=outputs_cfg.get("animate_density", False),
        )
        from .density import traj_movie

        try:
            traj_movie(
                traj,
                vel,
                times,
                vicsek_cfg["Lx"],
                vicsek_cfg["Ly"],
                out_dir,
                fps=24,
                marker_size=plot_opts.get("traj_marker_size", 4),
                draw_vectors=plot_opts.get("traj_quiver", True),
            )
        except Exception:
            pass

    save_run_metadata(out_dir, cfg, results)

    return {
        "config": cfg,
        "results": results,
        "metrics": psi_df,
        "out_dir": out_dir,
    }


def _run_single(
    config: Dict,
    *,
    ic_id: int | None = None,
    enable_videos: bool = True,
    enable_order_plots: bool = True,
) -> Dict:
    """Execute a single simulation and persist configured outputs.

    Steps performed
    ----------------
    1. Create the output directory (with ic_XXX subfolder if ic_id is set).
    2. Call :func:`rectsim.dynamics.simulate` to compute trajectories.
    3. Compute time-series metrics via :func:`rectsim.metrics.compute_timeseries`.
    4. Optionally save NPZ/CSV files and produce plots (final positions,
       order parameters, speed histogram, energy vs time).
    5. Optionally compute and write density movies via
       :func:`rectsim.density.hist2d_movie`.
    6. Persist run metadata via :func:`rectsim.io.save_run_metadata`.

    Parameters
    ----------
    config : Dict
        The configuration dictionary for the simulation.
    ic_id : int | None, optional
        The initial condition ID (0-indexed). If provided, outputs are written
        to a subdirectory `ic_{ic_id:03d}` within the base out_dir.
    enable_videos : bool, default=True
        Whether to generate videos for this IC. Should be gated based on
        outputs.video_ics configuration.
    enable_order_plots : bool, default=True
        Whether to generate order parameter plots for this IC. Should be gated
        based on outputs.order_params_ics configuration.

    Why this is separated from the integrator
    ------------------------------------------
    The integrator (``dynamics``) returns pure data structures. Side
    effects (disk I/O and plotting) are centralized here so they can
    be modified independently from numerical code.
    """

    model = config.get("model", "social_force")
    if model == "vicsek_discrete":
        return _run_vicsek_single(config)

    cfg = deepcopy(config)
    
    # Determine output directory based on ic_id
    base_out = Path(cfg["out_dir"])
    if ic_id is None:
        out_dir = base_out
    else:
        out_dir = base_out / f"ic_{ic_id:03d}"
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
        traj_npz = save_npz(
            out_dir,
            "traj",
            x=results["traj"],
            v=results["vel"],
            times=results["times"],
            params=np.array(json.dumps(results["params"])),
            sim=np.array(json.dumps(results["sim"])),
        )
        # Duplicate the archive under ``trajectories.npz`` for downstream
        # tooling (e.g. latent-model scripts) that historically looked for
        # that filename. Keeping both avoids breaking existing consumers.
        alias_path = traj_npz.with_name("trajectories.npz")
        np.savez(
            alias_path,
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
        
        # Only plot order parameters if enabled for this IC
        if enable_order_plots and cfg["outputs"].get("plot_order_params", True):
            _plot_order_params(out_dir, metrics_df, cfg)
        
        _plot_speeds(out_dir, results["vel"], results["times"], cfg)

    grid_cfg = cfg["outputs"].get("grid_density", {})
    if grid_cfg.get("enabled", True):
        # Only generate density movie if videos are enabled for this IC
        should_animate_density = (
            enable_videos 
            and cfg["outputs"].get("animate_density", cfg["outputs"].get("animate", False))
        )
        
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
            animate=should_animate_density,
        )
        
        # Only generate trajectory video if enabled for this IC
        if enable_videos and cfg["outputs"].get("animate_traj", False):
            plot_opts = cfg.get("outputs", {}).get("plot_options", {})
            from .density import traj_movie

            try:
                traj_movie(
                    results["traj"],
                    results.get("vel"),
                    results["times"],
                    cfg["sim"]["Lx"],
                    cfg["sim"]["Ly"],
                    out_dir,
                    fps=24,
                    marker_size=plot_opts.get("traj_marker_size", 4),
                    draw_vectors=plot_opts.get("traj_quiver", False),
                )
            except Exception:
                # Do not fail the whole run if trajectory animation cannot be produced
                pass

    save_run_metadata(out_dir, cfg, results)

    # Optionally run the EF-ROM latent pipeline automatically. This is
    # Legacy EF-ROM pipeline removed - use rom_mvar_* scripts instead
    # efrom_cfg = cfg.get("outputs", {}).get("efrom", {})
    # if efrom_cfg.get("auto_run", True):
    #     try:
    #         _run_efrom_pipeline(cfg, {"results": results, "out_dir": out_dir})
    #     except Exception as exc:
    #         print(f"EF-ROM postprocessing failed: {exc}")

    return {
        "config": cfg,
        "results": results,
        "metrics": metrics_df,
        "out_dir": out_dir,
    }


def run_multi_ic(
    base_config: Dict,
    n_ic: int,
    seeds: List[int] | None = None,
) -> Dict[str, Any]:
    """Run multiple simulations with different initial conditions.
    
    This helper loops over IC indices, applies video_ics/order_params_ics
    gating, and concatenates all metrics into a single CSV.
    
    Parameters
    ----------
    base_config : Dict
        Base configuration that will be copied for each IC.
    n_ic : int
        Number of initial conditions to simulate.
    seeds : List[int] | None, optional
        List of seeds to use for each IC. If None, uses range(n_ic).
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - "base_out": Path to base output directory
        - "metrics_all": Concatenated DataFrame with all IC metrics
        - "ic_results": List of individual IC result dictionaries
    """
    import pandas as pd
    
    if seeds is None:
        seeds = list(range(n_ic))
    elif len(seeds) != n_ic:
        raise ValueError(f"Length of seeds ({len(seeds)}) must match n_ic ({n_ic})")
    
    base_out = Path(base_config["out_dir"])
    base_out.mkdir(parents=True, exist_ok=True)
    
    # Get gating parameters
    video_ics = base_config["outputs"].get("video_ics", 1)
    order_params_ics = base_config["outputs"].get("order_params_ics", 1)
    
    all_records = []
    ic_results = []
    
    for ic_id, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"Running IC {ic_id + 1}/{n_ic} (seed={seed})")
        print(f"{'='*60}\n")
        
        # Prepare config for this IC
        cfg = deepcopy(base_config)
        cfg["seed"] = seed
        
        # Determine if this IC gets videos and order plots
        enable_videos = (ic_id < video_ics)
        enable_order_plots = (ic_id < order_params_ics)
        
        # Run the simulation
        run_result = _run_single(
            cfg,
            ic_id=ic_id,
            enable_videos=enable_videos,
            enable_order_plots=enable_order_plots,
        )
        
        # Add ic_id to metrics
        metrics = run_result["metrics"].copy()
        metrics["ic_id"] = ic_id
        metrics["seed"] = seed
        all_records.append(metrics)
        
        ic_results.append(run_result)
    
    # Concatenate all metrics
    metrics_all = pd.concat(all_records, ignore_index=True)
    metrics_all_path = base_out / "metrics_all.csv"
    metrics_all.to_csv(metrics_all_path, index=False)
    print(f"\n✓ Saved concatenated metrics to {metrics_all_path}")
    
    # Write summary metadata
    summary = {
        "n_ic": n_ic,
        "seeds": seeds,
        "video_ics": video_ics,
        "order_params_ics": order_params_ics,
        "base_config": base_config,
    }
    summary_path = base_out / "run.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"✓ Saved experiment summary to {summary_path}\n")
    
    return {
        "base_out": base_out,
        "metrics_all": metrics_all,
        "ic_results": ic_results,
    }


def _grid_centers(nx: int, ny: int, Lx: float, Ly: float) -> np.ndarray:
    x_centres = (np.arange(nx) + 0.5) * (Lx / nx)
    y_centres = (np.arange(ny) + 0.5) * (Ly / ny)
    xv, yv = np.meshgrid(x_centres, y_centres, indexing="xy")
    return np.stack([xv.ravel(), yv.ravel()], axis=-1)


def cmd_single(args: argparse.Namespace, overrides: List[Tuple[str, str]]) -> None:
    """Entry point for the ``single`` CLI command."""

    override_pairs = [(k, _convert_value(v)) for k, v in overrides]
    config = load_config(args.config, override_pairs)
    if getattr(args, "model", None):
        config["model"] = args.model
    _run_single(config)


def cmd_grid(args: argparse.Namespace, overrides: List[Tuple[str, str]]) -> None:
    """Entry point for the ``grid`` CLI command that sweeps parameters."""

    override_pairs = [(k, _convert_value(v)) for k, v in overrides]
    config = load_config(args.config, override_pairs)
    if getattr(args, "model", None):
        config["model"] = args.model

    if config.get("model", "social_force") != "social_force":
        raise ValueError("Grid sweeps are currently supported only for the social_force model.")
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


def cmd_validate_all(args: argparse.Namespace, overrides: List[Tuple[str, str]]) -> None:
    """Run simulation + EF-ROM validation pipeline."""

    override_pairs = [(k, _convert_value(v)) for k, v in overrides]
    config = load_config(args.config, override_pairs)
    if getattr(args, "model", None):
        config["model"] = args.model

    if config.get("model", "social_force") != "social_force":
        raise ValueError("validate_all is currently implemented only for the social_force model.")
    run = _run_single(config)
    # Legacy EF-ROM removed - use rom_mvar_* scripts instead
    # _run_efrom_pipeline(config, run)


# Legacy EF-ROM pipeline removed - use rom_mvar_* scripts instead
def _run_efrom_pipeline_DISABLED(config: Dict, run: Dict) -> None:
    """DISABLED: Legacy EF-ROM training (use rom_mvar_* scripts instead).

    This helper mirrors the previous `cmd_validate_all` logic but is
    callable from other places (e.g., automatically after `_run_single`).
    It writes artifacts under `<out_dir>/efrom`.
    """
    return  # Function disabled - legacy code removed
    traj = results["traj"]
    times = results["times"]
    sim_cfg = config["sim"]
    out_dir = Path(run["out_dir"]) / "efrom"
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_cfg = config["outputs"]["grid_density"]
    nx = grid_cfg["nx"]
    ny = grid_cfg["ny"]
    bandwidth = grid_cfg.get("bandwidth", 0.5)
    rho = density_movie_kde(
        traj,
        sim_cfg["Lx"],
        sim_cfg["Ly"],
        nx,
        ny,
        bandwidth,
        sim_cfg["bc"],
    )

    efrom_cfg = config["outputs"].get("efrom", {})
    rank = int(efrom_cfg.get("rank", min(10, nx * ny)))
    order = int(efrom_cfg.get("order", 4))
    horizon = int(efrom_cfg.get("horizon", max(1, rho.shape[0] // 5)))

    split = max(order, int(np.floor(0.8 * rho.shape[0])))
    split = min(split, rho.shape[0] - max(horizon, order + 1))
    rho_train = rho[:split]
    rho_test = rho[split:]
    horizon = min(horizon, rho_test.shape[0])
    if horizon <= 0:
        raise ValueError("Not enough frames to compute EF-ROM forecast")

    cell_area = (sim_cfg["Lx"] / nx) * (sim_cfg["Ly"] / ny)
    rho_pred, _ = efrom_train_and_forecast(
        rho_train,
        rho_test,
        rank=rank,
        order=order,
        horizon=horizon,
        cell_area=cell_area,
    )
    rho_true = rho_test[:horizon]

    flat_true = rho_true.reshape(horizon, -1)
    flat_pred = rho_pred.reshape(horizon, -1)
    rmse_val = float(rmse(flat_true, flat_pred))
    r2_val = float(r2(flat_true, flat_pred))
    rel_err_series = mean_relative_error(flat_true, flat_pred, axis=1)
    tol_idx = tolerance_horizon(rel_err_series)
    mass_pred = flat_pred.sum(axis=1) * cell_area
    mass_target = traj.shape[1]
    mass_drift = float(np.max(np.abs(mass_pred - mass_target)))

    metrics_payload = {
        "rmse": rmse_val,
        "r2": r2_val,
        "mean_relative_error": rel_err_series.tolist(),
        "tolerance_index": int(tol_idx),
        "mass_drift": mass_drift,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    dx = sim_cfg["Lx"] / nx
    dy = sim_cfg["Ly"] / ny
    Xc = _grid_centers(nx, ny, sim_cfg["Lx"], sim_cfg["Ly"])
    vmin = float(min(rho_true.min(), rho_pred.min()))
    vmax = float(max(rho_true.max(), rho_pred.max()))
    animate_heatmap_movie(
        rho_true.reshape(horizon, -1),
        Xc,
        dx,
        dy,
        sim_cfg["Lx"],
        sim_cfg["Ly"],
        out_path=str(out_dir / "truth"),
        fps=20,
        vmin=vmin,
        vmax=vmax,
        title="Truth",
    )
    animate_heatmap_movie(
        rho_pred.reshape(horizon, -1),
        Xc,
        dx,
        dy,
        sim_cfg["Lx"],
        sim_cfg["Ly"],
        out_path=str(out_dir / "forecast"),
        fps=20,
        vmin=vmin,
        vmax=vmax,
        title="EF-ROM",
    )
    animate_side_by_side(
        rho_true.reshape(horizon, -1),
        rho_pred.reshape(horizon, -1),
        Xc,
        dx,
        dy,
        sim_cfg["Lx"],
        sim_cfg["Ly"],
        out_path=str(out_dir / "compare"),
        fps=20,
        vmin=vmin,
        vmax=vmax,
    )

    print("EF-ROM metrics:")
    print(json.dumps(metrics_payload, indent=2))
    print(f"Artifacts written to {out_dir}")


def cmd_ensemble(args: argparse.Namespace, overrides: List[Tuple[str, str]]) -> None:
    """Entry point for the ``ensemble`` CLI command.
    
    Generates multiple simulation runs with varied initial conditions
    and stores them in simulations/<model_id>/run_xxxx/ structure.
    """
    override_pairs = [(k, _convert_value(v)) for k, v in overrides]
    config = load_config(args.config, override_pairs)
    if getattr(args, "model", None):
        config["model"] = args.model
    
    from .utils import generate_model_id
    
    # Generate model ID
    model_id = generate_model_id(config)
    print(f"Model ID: {model_id}")
    
    # Get ensemble configuration
    ensemble_cfg = config.get("ensemble", {})
    n_runs = ensemble_cfg.get("n_runs", 20)
    seeds = ensemble_cfg.get("seeds")
    base_seed = ensemble_cfg.get("base_seed", 0)
    ic_types = ensemble_cfg.get("ic_types", ["gaussian", "uniform", "ring", "cluster"])
    ic_weights = ensemble_cfg.get("ic_weights")
    
    # Determine seeds
    if seeds is not None:
        seeds = list(seeds)
        n_runs = len(seeds)
    else:
        seeds = [base_seed + k for k in range(n_runs)]
    
    # Normalize IC weights
    if ic_weights is None:
        # Uniform weights
        ic_weights = [1.0 / len(ic_types)] * len(ic_types)
    else:
        # Normalize to sum to 1
        total = sum(ic_weights)
        ic_weights = [w / total for w in ic_weights]
    
    print(f"Generating {n_runs} simulation runs")
    print(f"IC types: {ic_types} with weights {ic_weights}")
    
    # Create base directory
    base_dir = Path("simulations") / model_id
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create RNG for IC sampling
    ic_rng = np.random.default_rng(base_seed)
    
    # Track ensemble metadata
    ensemble_rows = []
    
    for k in range(n_runs):
        run_name = f"run_{k:04d}"
        run_dir = base_dir / run_name
        
        # Sample IC type
        ic_type = ic_rng.choice(ic_types, p=ic_weights)
        
        # Create per-run config
        run_config = deepcopy(config)
        run_config["seed"] = seeds[k]
        run_config["ic"]["type"] = ic_type
        run_config["out_dir"] = str(run_dir)
        
        print(f"\n[{k+1}/{n_runs}] Running {run_name} (seed={seeds[k]}, ic={ic_type})")
        
        # Run simulation
        try:
            result = _run_single(run_config)
            
            # Extract final order parameters for summary
            metrics_df = result["metrics"]
            final_row = metrics_df.iloc[-1]
            
            ensemble_rows.append({
                "run_id": run_name,
                "seed": seeds[k],
                "ic_type": ic_type,
                "final_polarization": final_row["polarization"],
                "final_speed": final_row["mean_speed"],
                "final_angular_momentum": final_row["angular_momentum"],
                "final_dnn": final_row["dnn"],
            })
            
            print(f"  ✓ Completed {run_name}")
            
        except Exception as exc:
            print(f"  ✗ Failed {run_name}: {exc}")
            ensemble_rows.append({
                "run_id": run_name,
                "seed": seeds[k],
                "ic_type": ic_type,
                "final_polarization": np.nan,
                "final_speed": np.nan,
                "final_angular_momentum": np.nan,
                "final_dnn": np.nan,
            })
    
    # Save ensemble summary
    ensemble_df = pd.DataFrame(ensemble_rows)
    summary_path = base_dir / "ensemble_runs.csv"
    ensemble_df.to_csv(summary_path, index=False)
    
    print(f"\n✓ Ensemble generation complete!")
    print(f"  {len(ensemble_rows)} runs saved to: {base_dir}")
    print(f"  Summary: {summary_path}")
    print(f"\nEnsemble statistics:")
    print(f"  Mean final polarization: {ensemble_df['final_polarization'].mean():.4f}")
    print(f"  Mean final speed: {ensemble_df['final_speed'].mean():.4f}")
    print(f"  IC type distribution:")
    for ic_type in ic_types:
        count = (ensemble_df['ic_type'] == ic_type).sum()
        print(f"    {ic_type}: {count}")


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level CLI argument parser."""

    parser = argparse.ArgumentParser(prog="rectsim")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single", help="Run a single simulation")
    single.add_argument("--config", required=True, help="Path to configuration YAML")
    single.add_argument(
        "--model",
        choices=["social_force", "vicsek_discrete"],
        help="Override the model specified in the config file",
    )

    grid = subparsers.add_parser("grid", help="Run a parameter grid")
    grid.add_argument("--config", required=True, help="Path to grid configuration YAML")
    grid.add_argument(
        "--model",
        choices=["social_force", "vicsek_discrete"],
        help="Override the model specified in the config file",
    )

    validate = subparsers.add_parser(
        "validate_all",
        help="Run simulation, KDE, latent training, and forecast validation",
    )
    validate.add_argument("--config", required=True, help="Path to configuration YAML")
    validate.add_argument(
        "--model",
        choices=["social_force", "vicsek_discrete"],
        help="Override the model specified in the config file",
    )

    ensemble = subparsers.add_parser(
        "ensemble",
        help="Generate ensemble of simulations with varied initial conditions",
    )
    ensemble.add_argument("--config", required=True, help="Path to configuration YAML")
    ensemble.add_argument(
        "--model",
        choices=["social_force", "vicsek_discrete"],
        help="Override the model specified in the config file",
    )

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
    elif args.command == "validate_all":
        cmd_validate_all(args, overrides)
    elif args.command == "ensemble":
        cmd_ensemble(args, overrides)
    else:  # pragma: no cover - defensive
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
