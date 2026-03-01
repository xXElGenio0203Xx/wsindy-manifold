#!/usr/bin/env python3
"""
WSINDy Pipeline — End-to-end PDE Discovery, Forecasting & Diagnostics
=======================================================================

Analogous to ROM_pipeline.py for MVAR/LSTM, but for WSINDy PDE discovery.

Pipeline stages:
  1. Generate synthetic spatiotemporal field data (or load from file)
  2. Build WSINDy library (from YAML config or defaults)
  3. Automated model selection across test-function scales ℓ
  4. Bootstrap uncertainty quantification
  5. Stability selection
  6. Forecast with best model (ETDRK4 / RK4)
  7. Evaluate rollout metrics
  8. Generate visualisation artefacts (plots + summary)

Usage:
  python run_wsindy_pipeline.py --config configs/wsindy_experiment.yaml
  python run_wsindy_pipeline.py                          # use built-in defaults
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from wsindy.grid import GridSpec
from wsindy.library import LibraryBuilder, default_library, library_from_config, patch_feature_registries
from wsindy.test_functions import make_separable_psi
from wsindy.system import build_weak_system, default_t_margin, make_query_indices
from wsindy.fit import wsindy_fit_regression
from wsindy.select import default_ell_grid, wsindy_model_selection
from wsindy.forecast import wsindy_forecast
from wsindy.eval import rollout_metrics
from wsindy.integrators import split_linear_nonlinear
from wsindy.uncertainty import bootstrap_wsindy
from wsindy.stability import stability_selection
from wsindy.pretty import to_text, to_latex, group_terms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════
#  Synthetic data generators
# ═══════════════════════════════════════════════════════════════════

def _exact_heat(X, Y, t, D):
    """Multi-mode heat equation u_t = D Δu (periodic)."""
    return (
        1.0
        + np.exp(-D * 2 * t) * np.cos(X) * np.cos(Y)
        + 0.8 * np.exp(-D * 4 * t) * np.cos(2 * X)
        + 0.5 * np.exp(-D * 16 * t) * np.cos(4 * X)
        + 0.6 * np.exp(-D * 9 * t) * np.cos(3 * Y)
    )


def _advection_diffusion(X, Y, t, D, cx, cy):
    """Advection-diffusion: u_t = D Δu - cx u_x - cy u_y (periodic)."""
    kx1, ky1 = 1.0, 1.0
    kx2, ky2 = 2.0, 0.0
    ksq1 = kx1**2 + ky1**2
    ksq2 = kx2**2 + ky2**2
    return (
        1.0
        + np.exp(-D * ksq1 * t) * np.cos(kx1 * X - cx * t + ky1 * Y - cy * t)
        + 0.7 * np.exp(-D * ksq2 * t) * np.cos(kx2 * X - cx * t)
    )


def generate_data(cfg):
    """Generate synthetic PDE field data from config."""
    pde = cfg.get("pde", "heat")
    Lx = cfg.get("Lx", 2 * np.pi)
    Ly = cfg.get("Ly", 2 * np.pi)
    nx = cfg.get("nx", 64)
    ny = cfg.get("ny", 64)
    dt = cfg.get("dt", 0.05)
    T_steps = cfg.get("T_steps", 100)
    D = cfg.get("D", 0.1)
    cx = cfg.get("cx", 0.5)
    cy = cfg.get("cy", 0.3)

    dx = Lx / nx
    dy = Ly / ny
    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    grid = GridSpec(dt=dt, dx=dx, dy=dy)

    U = np.zeros((T_steps, nx, ny))
    for i in range(T_steps):
        t = i * dt
        if pde == "heat":
            U[i] = _exact_heat(X, Y, t, D)
        elif pde == "advection_diffusion":
            U[i] = _advection_diffusion(X, Y, t, D, cx, cy)
        else:
            raise ValueError(f"Unknown PDE type: {pde}")

    true_terms = {}
    if pde == "heat":
        true_terms = {"lap:u": D}
    elif pde == "advection_diffusion":
        true_terms = {"lap:u": D, "dx:u": -cx, "dy:u": -cy}

    return U, grid, X, Y, true_terms


# ═══════════════════════════════════════════════════════════════════
#  Visualisation
# ═══════════════════════════════════════════════════════════════════

def plot_discovery_summary(model, true_terms, out_dir):
    """Bar chart of discovered vs true coefficients."""
    fig, ax = plt.subplots(figsize=(8, 4))
    names = model.active_terms
    coeffs = model.active_coeffs

    x = np.arange(len(names))
    bars = ax.bar(x, coeffs, color="steelblue", alpha=0.8, label="Discovered")

    # Overlay true coefficients
    for i, name in enumerate(names):
        if name in true_terms:
            ax.scatter(i, true_terms[name], color="red", s=100, zorder=5,
                       marker="*", label="True" if i == 0 else "")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Coefficient", fontsize=12)
    ax.set_title("Discovered PDE Coefficients", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="k", linewidth=0.5)

    plt.tight_layout()
    path = out_dir / "coefficients.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_rollout_r2(met, out_dir, dt):
    """R² over time for the forecast rollout."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    t = np.arange(len(met["r2_t"])) * dt

    # R²
    axes[0].plot(t, met["r2_t"], linewidth=2, color="steelblue")
    axes[0].set_xlabel("Time", fontsize=12)
    axes[0].set_ylabel("R²", fontsize=12)
    axes[0].set_title("Snapshot R²", fontsize=13, fontweight="bold")
    axes[0].axhline(1.0, color="k", linestyle="--", alpha=0.3)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(min(0, met["r2_t"].min() - 0.05), 1.05)

    # Relative L2
    axes[1].semilogy(t, met["rel_l2_t"], linewidth=2, color="darkorange")
    axes[1].set_xlabel("Time", fontsize=12)
    axes[1].set_ylabel("Relative L² error", fontsize=12)
    axes[1].set_title("Relative L² Error", fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    # Mass drift
    axes[2].plot(t, met["mass_drift"], linewidth=2, color="seagreen")
    axes[2].set_xlabel("Time", fontsize=12)
    axes[2].set_ylabel("Mass drift", fontsize=12)
    axes[2].set_title("Mass Conservation", fontsize=13, fontweight="bold")
    axes[2].axhline(0, color="k", linestyle="--", alpha=0.3)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "rollout_metrics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_snapshots(U_true, U_pred, times_idx, dt, out_dir):
    """Side-by-side snapshots: true vs predicted at selected times."""
    n = len(times_idx)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 10))

    vmin = min(U_true.min(), U_pred.min())
    vmax = max(U_true.max(), U_pred.max())

    for j, ti in enumerate(times_idx):
        t_val = ti * dt

        ax_true = axes[0, j] if n > 1 else axes[0]
        ax_pred = axes[1, j] if n > 1 else axes[1]
        ax_diff = axes[2, j] if n > 1 else axes[2]

        im0 = ax_true.imshow(U_true[ti].T, origin="lower", vmin=vmin, vmax=vmax,
                              cmap="viridis", aspect="equal")
        ax_true.set_title(f"True  t={t_val:.2f}", fontsize=11)
        ax_true.axis("off")

        im1 = ax_pred.imshow(U_pred[ti].T, origin="lower", vmin=vmin, vmax=vmax,
                              cmap="viridis", aspect="equal")
        ax_pred.set_title(f"Pred  t={t_val:.2f}", fontsize=11)
        ax_pred.axis("off")

        diff = np.abs(U_true[ti] - U_pred[ti])
        im2 = ax_diff.imshow(diff.T, origin="lower", cmap="hot", aspect="equal")
        ax_diff.set_title(f"|Error|  t={t_val:.2f}", fontsize=11)
        ax_diff.axis("off")

    # Colour bars
    fig.colorbar(im0, ax=axes[0, :] if n > 1 else axes[0], shrink=0.7, label="u")
    fig.colorbar(im2, ax=axes[2, :] if n > 1 else axes[2], shrink=0.7, label="|error|")

    fig.suptitle("True vs Predicted Snapshots", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = out_dir / "snapshots.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_bootstrap_ci(boot_result, true_terms, out_dir):
    """Bootstrap coefficient distributions with confidence intervals."""
    ci = boot_result.confidence_interval(0.05)
    means = boot_result.coeff_mean
    inc_prob = boot_result.inclusion_probability

    # Only show terms that were ever active
    mask = inc_prob > 0
    if not np.any(mask):
        return None

    names = [boot_result.col_names[i] for i in np.where(mask)[0]]
    idx = np.where(mask)[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: coefficients with CI
    x = np.arange(len(names))
    axes[0].errorbar(x, means[idx], yerr=[means[idx] - ci[idx, 0], ci[idx, 1] - means[idx]],
                     fmt="o", capsize=5, color="steelblue", markersize=8, label="Bootstrap mean ± 95% CI")
    for i, nm in enumerate(names):
        if nm in true_terms:
            axes[0].scatter(i, true_terms[nm], color="red", s=120, marker="*", zorder=5,
                            label="True" if i == 0 else "")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=10)
    axes[0].set_ylabel("Coefficient", fontsize=12)
    axes[0].set_title("Bootstrap Coefficients (95% CI)", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].axhline(0, color="k", linewidth=0.5)

    # Panel 2: inclusion probability
    axes[1].bar(x, inc_prob[idx], color="darkorange", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, fontsize=10)
    axes[1].set_ylabel("Inclusion probability", fontsize=12)
    axes[1].set_title("Term Selection Frequency (Bootstrap)", fontsize=13, fontweight="bold")
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(1.0, color="k", linestyle="--", alpha=0.3)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = out_dir / "bootstrap_uq.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_stability(stab_result, out_dir):
    """Stability selection frequency bar chart."""
    freq = stab_result.freq
    names = stab_result.col_names
    threshold = stab_result.threshold

    mask = freq > 0
    if not np.any(mask):
        return None

    idx = np.where(mask)[0]
    order = np.argsort(-freq[idx])
    idx = idx[order]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(idx))
    colors = ["seagreen" if freq[i] >= threshold else "salmon" for i in idx]
    ax.bar(x, freq[idx], color=colors, alpha=0.85)
    ax.axhline(threshold, color="k", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels([names[i] for i in idx], fontsize=10)
    ax.set_ylabel("Selection frequency", fontsize=12)
    ax.set_title("Stability Selection", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = out_dir / "stability_selection.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_model_selection_sweep(sel_result, out_dir):
    """Scatter plot of model selection sweep: loss vs sparsity, coloured by score."""
    trials = sel_result.trials
    losses = np.array([t.normalised_loss for t in trials])
    n_active = np.array([t.n_active for t in trials])
    scores = np.array([t.composite_score for t in trials])

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(n_active, losses, c=scores, cmap="coolwarm_r", s=80, alpha=0.85, edgecolors="k", linewidths=0.5)
    plt.colorbar(sc, ax=ax, label="Composite score")

    # Mark best
    ax.scatter(sel_result.best.n_active, sel_result.best.normalised_loss,
               color="gold", s=200, marker="*", zorder=5, edgecolors="k", linewidths=1.5, label="Best")

    # Mark Pareto frontier
    pareto_n = [t.n_active for t in sel_result.pareto]
    pareto_l = [t.normalised_loss for t in sel_result.pareto]
    order = np.argsort(pareto_n)
    ax.plot(np.array(pareto_n)[order], np.array(pareto_l)[order], "k--", alpha=0.5, label="Pareto frontier")

    ax.set_xlabel("# Active terms", fontsize=12)
    ax.set_ylabel("Normalised loss", fontsize=12)
    ax.set_title("Model Selection Sweep", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "model_selection_sweep.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ═══════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════

def load_config_yaml(path):
    """Load YAML config, return dict."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def default_config():
    """Built-in default configuration for a quick demo."""
    return {
        "experiment_name": "wsindy_heat_demo",
        "data": {
            "pde": "heat",
            "Lx": 2 * np.pi, "Ly": 2 * np.pi,
            "nx": 64, "ny": 64,
            "dt": 0.05,
            "T_steps": 100,
            "D": 0.1,
        },
        "wsindy": {
            "library": {
                "max_poly": 3,
                "operators": ["I", "dx", "dy", "lap"],
            },
            "model_selection": {
                "n_ell": 8,
                "p": [2, 2, 2],
                "alpha": 0.1,
                "beta": 0.01,
                "lambda_min": -3,
                "lambda_max": 1,
                "n_lambda": 40,
            },
            "bootstrap": {
                "enabled": True,
                "B": 50,
            },
            "stability": {
                "enabled": True,
                "n_bootstrap": 5,
                "threshold": 0.6,
            },
            "forecast": {
                "n_steps": 60,
                "clip_negative": False,
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description="WSINDy PDE Discovery Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--experiment_name", type=str, default=None, help="Override experiment name")
    args = parser.parse_args()

    # ── Load config ────────────────────────────────────────────────
    if args.config:
        cfg = load_config_yaml(args.config)
    else:
        cfg = default_config()

    exp_name = args.experiment_name or cfg.get("experiment_name", "wsindy_experiment")
    data_cfg = cfg.get("data", {})
    ws_cfg = cfg.get("wsindy", {})

    OUT_DIR = Path(f"artifacts/wsindy/{exp_name}")
    PLOTS_DIR = OUT_DIR / "plots"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    sep = "=" * 70
    thin = "-" * 70

    print(f"\n{sep}")
    print("  WSINDy PDE DISCOVERY PIPELINE")
    print(f"{sep}")
    print(f"  Experiment : {exp_name}")
    if args.config:
        print(f"  Config     : {args.config}")
    print(f"  Output     : {OUT_DIR}")
    print(f"  Timestamp  : {time.strftime('%Y-%m-%d %H:%M:%S')}")

    t_start = time.perf_counter()

    # ================================================================
    # STEP 1: Generate / Load Data
    # ================================================================
    print(f"\n{thin}")
    print("  STEP 1: Generating data")
    print(f"{thin}")

    U, grid, X, Y, true_terms = generate_data(data_cfg)
    T_data, nx, ny = U.shape
    print(f"  PDE      : {data_cfg.get('pde', 'heat')}")
    print(f"  Grid     : T={T_data}, nx={nx}, ny={ny}")
    print(f"  Spacings : dt={grid.dt}, dx={grid.dx:.4f}, dy={grid.dy:.4f}")
    print(f"  True PDE : {true_terms}")

    # ================================================================
    # STEP 2: Build library
    # ================================================================
    print(f"\n{thin}")
    print("  STEP 2: Building library")
    print(f"{thin}")

    lib_cfg = ws_cfg.get("library", {})
    if lib_cfg:
        library_terms = library_from_config(lib_cfg)
    else:
        library_terms = default_library()
    patch_feature_registries()

    print(f"  Library size : {len(library_terms)} terms")
    for op, feat in library_terms:
        print(f"    {op}:{feat}")

    # ================================================================
    # STEP 3: Model selection sweep
    # ================================================================
    print(f"\n{thin}")
    print("  STEP 3: Model selection across test-function scales")
    print(f"{thin}")

    ms_cfg = ws_cfg.get("model_selection", {})
    n_ell = ms_cfg.get("n_ell", 8)
    p = tuple(ms_cfg.get("p", [2, 2, 2]))
    alpha = ms_cfg.get("alpha", 0.1)
    beta = ms_cfg.get("beta", 0.01)
    lam_min = ms_cfg.get("lambda_min", -3)
    lam_max = ms_cfg.get("lambda_max", 1)
    n_lam = ms_cfg.get("n_lambda", 40)
    lambdas = np.logspace(lam_min, lam_max, n_lam)

    ell_grid = default_ell_grid(T_data, nx, ny, n_points=n_ell)
    # Also add some tight scales
    ell_grid = [(2, 3, 3), (3, 4, 4), (3, 5, 5)] + ell_grid
    ell_grid = list(dict.fromkeys(ell_grid))  # deduplicate

    print(f"  ell grid   : {len(ell_grid)} configs")
    print(f"  p          : {p}")
    print(f"  lambdas    : {n_lam} in [{10**lam_min:.0e}, {10**lam_max:.0e}]")
    print(f"  alpha={alpha}, beta={beta}")

    t0 = time.perf_counter()
    sel_result = wsindy_model_selection(
        U, grid, library_terms, ell_grid,
        p=p, lambdas=lambdas, alpha=alpha, beta=beta, verbose=True,
    )
    ms_time = time.perf_counter() - t0

    print(f"\n  Completed in {ms_time:.2f}s")
    print(f"  Trials    : {len(sel_result.trials)}")
    print(f"  Pareto    : {len(sel_result.pareto)}")
    print(f"\n{sel_result.summary(top_k=5)}")

    model = sel_result.best_model

    # ── Pretty-print discovered PDE ────────────────────────────────
    print(f"\n  Discovered PDE (text):")
    print(f"    {to_text(model)}")
    print(f"\n  Discovered PDE (LaTeX):")
    print(f"    {to_latex(model)}")

    groups = group_terms(model)
    print(f"\n  Term groups:")
    for cat, entries in groups.items():
        names = [e["term"] for e in entries]
        print(f"    {cat:12s}: {', '.join(names)}")

    # Coefficient comparison
    print(f"\n  Coefficient comparison:")
    for name in model.active_terms:
        idx = model.col_names.index(name)
        true_val = true_terms.get(name, None)
        disc_val = model.w[idx]
        if true_val is not None:
            err = abs(disc_val - true_val) / max(abs(true_val), 1e-10)
            print(f"    {name:>10s}  discovered={disc_val:+.6f}  true={true_val:+.6f}  rel_err={err:.2%}")
        else:
            print(f"    {name:>10s}  discovered={disc_val:+.6f}  (spurious)")

    # ================================================================
    # STEP 4: Bootstrap UQ
    # ================================================================
    boot_cfg = ws_cfg.get("bootstrap", {})
    boot_result = None
    if boot_cfg.get("enabled", True):
        print(f"\n{thin}")
        print("  STEP 4: Bootstrap uncertainty quantification")
        print(f"{thin}")

        B = boot_cfg.get("B", 50)

        # Build system at best ℓ for bootstrap
        best_trial = sel_result.best
        psi_bundle = make_separable_psi(
            grid, best_trial.ell[0], best_trial.ell[1], best_trial.ell[2],
            best_trial.p[0], best_trial.p[1], best_trial.p[2],
        )
        t_margin = default_t_margin(psi_bundle)
        query_idx = make_query_indices(
            T_data, nx, ny,
            best_trial.stride[0], best_trial.stride[1], best_trial.stride[2],
            t_margin,
        )
        b, G, col_names = build_weak_system(U, grid, psi_bundle, library_terms, query_idx)

        t0 = time.perf_counter()
        boot_result = bootstrap_wsindy(
            b, G, col_names, B=B,
            lambdas=lambdas,
            rng=np.random.default_rng(42),
        )
        boot_time = time.perf_counter() - t0
        print(f"  B = {B} replicates in {boot_time:.2f}s")
        print(f"\n{boot_result.summary()}")

    # ================================================================
    # STEP 5: Stability selection
    # ================================================================
    stab_cfg = ws_cfg.get("stability", {})
    stab_result = None
    if stab_cfg.get("enabled", True):
        print(f"\n{thin}")
        print("  STEP 5: Stability selection")
        print(f"{thin}")

        n_boot_stab = stab_cfg.get("n_bootstrap", 5)
        threshold = stab_cfg.get("threshold", 0.6)

        # Use a subset of ell_grid for stability
        stab_ells = ell_grid[:min(5, len(ell_grid))]

        t0 = time.perf_counter()
        stab_result = stability_selection(
            U, grid, library_terms,
            ell_grid=stab_ells,
            p=p,
            n_bootstrap=n_boot_stab,
            lambdas=lambdas,
            threshold=threshold,
            rng=np.random.default_rng(123),
        )
        stab_time = time.perf_counter() - t0

        print(f"  {len(stab_result.config_labels)} trials in {stab_time:.2f}s")
        print(f"\n{stab_result.summary()}")

    # ================================================================
    # STEP 6: Forecast with best model
    # ================================================================
    print(f"\n{thin}")
    print("  STEP 6: Forecasting with discovered model")
    print(f"{thin}")

    fc_cfg = ws_cfg.get("forecast", {})
    n_forecast = fc_cfg.get("n_steps", 60)
    clip_neg = fc_cfg.get("clip_negative", False)

    lin, nonlin = split_linear_nonlinear(model)
    method = "etdrk4" if lin else "rk4"
    print(f"  Method     : {method}")
    print(f"  Steps      : {n_forecast}")
    print(f"  Linear     : {[f'{k} ({v:+.4f})' for k, v in lin.items()]}")
    print(f"  Nonlinear  : {[f'{k} ({v:+.4f})' for k, v in nonlin.items()]}")

    t0 = time.perf_counter()
    U_pred = wsindy_forecast(U[0], model, grid, n_steps=n_forecast,
                             method=method, clip_negative=clip_neg)
    fc_time = time.perf_counter() - t0
    print(f"  Forecast completed in {fc_time:.3f}s")

    # Ground truth for comparison
    U_true = np.zeros_like(U_pred)
    pde_type = data_cfg.get("pde", "heat")
    D = data_cfg.get("D", 0.1)
    cx = data_cfg.get("cx", 0.5)
    cy = data_cfg.get("cy", 0.3)
    for i in range(n_forecast + 1):
        t = i * grid.dt
        if pde_type == "heat":
            U_true[i] = _exact_heat(X, Y, t, D)
        elif pde_type == "advection_diffusion":
            U_true[i] = _advection_diffusion(X, Y, t, D, cx, cy)

    # ================================================================
    # STEP 7: Evaluate rollout
    # ================================================================
    print(f"\n{thin}")
    print("  STEP 7: Rollout evaluation")
    print(f"{thin}")

    met = rollout_metrics(U_true, U_pred, grid)

    print(f"  mean R²          = {met['r2_mean']:.6f}")
    print(f"  R²(t=0)          = {met['r2_t'][0]:.6f}")
    print(f"  R²(t=end)        = {met['r2_t'][-1]:.6f}")
    print(f"  max rel L²       = {met['rel_l2_t'].max():.4e}")
    print(f"  max |mass drift| = {np.max(np.abs(met['mass_drift'])):.2e}")

    # ================================================================
    # STEP 8: Visualisation
    # ================================================================
    print(f"\n{thin}")
    print("  STEP 8: Generating visualisations")
    print(f"{thin}")

    p1 = plot_discovery_summary(model, true_terms, PLOTS_DIR)
    print(f"  Saved: {p1}")

    p2 = plot_rollout_r2(met, PLOTS_DIR, grid.dt)
    print(f"  Saved: {p2}")

    # Snapshot times: start, 1/3, 2/3, end
    snap_idx = [0, n_forecast // 3, 2 * n_forecast // 3, n_forecast]
    p3 = plot_snapshots(U_true, U_pred, snap_idx, grid.dt, PLOTS_DIR)
    print(f"  Saved: {p3}")

    p4 = plot_model_selection_sweep(sel_result, PLOTS_DIR)
    print(f"  Saved: {p4}")

    if boot_result is not None:
        p5 = plot_bootstrap_ci(boot_result, true_terms, PLOTS_DIR)
        if p5:
            print(f"  Saved: {p5}")

    if stab_result is not None:
        p6 = plot_stability(stab_result, PLOTS_DIR)
        if p6:
            print(f"  Saved: {p6}")

    # ================================================================
    # Save artefacts
    # ================================================================
    total_time = time.perf_counter() - t_start

    # Summary JSON
    summary = {
        "experiment_name": exp_name,
        "pde": data_cfg.get("pde", "heat"),
        "grid": {"T": T_data, "nx": nx, "ny": ny, "dt": grid.dt, "dx": grid.dx, "dy": grid.dy},
        "true_terms": true_terms,
        "discovered": {
            "text": to_text(model),
            "latex": to_latex(model),
            "active_terms": model.active_terms,
            "coefficients": {n: float(model.w[model.col_names.index(n)]) for n in model.active_terms},
            "n_active": model.n_active,
            "lambda_star": float(model.best_lambda),
            "r2_weak": float(model.diagnostics.get("r2", 0)),
        },
        "model_selection": {
            "n_trials": len(sel_result.trials),
            "n_pareto": len(sel_result.pareto),
            "best_ell": list(sel_result.best.ell),
            "best_score": float(sel_result.best.composite_score),
        },
        "forecast": {
            "n_steps": n_forecast,
            "method": method,
            "r2_mean": float(met["r2_mean"]),
            "r2_final": float(met["r2_t"][-1]),
            "max_rel_l2": float(met["rel_l2_t"].max()),
            "max_mass_drift": float(np.max(np.abs(met["mass_drift"]))),
        },
        "timing": {
            "model_selection_s": round(ms_time, 2),
            "forecast_s": round(fc_time, 3),
            "total_s": round(total_time, 2),
        },
    }

    if boot_result is not None:
        summary["bootstrap"] = {
            "B": boot_result.B,
            "inclusion_probability": {
                n: float(boot_result.inclusion_probability[i])
                for i, n in enumerate(boot_result.col_names)
                if boot_result.inclusion_probability[i] > 0
            },
        }

    if stab_result is not None:
        summary["stability"] = {
            "threshold": stab_result.threshold,
            "robust_terms": stab_result.robust_terms,
            "fragile_terms": stab_result.fragile_terms,
        }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary: {OUT_DIR / 'summary.json'}")

    # NPZ artefacts
    npz_data = dict(
        U_true=U_true, U_pred=U_pred,
        r2_t=met["r2_t"], rel_l2_t=met["rel_l2_t"], mass_drift=met["mass_drift"],
        discovered_w=model.w, discovered_active=model.active,
        col_names=np.array(model.col_names),
    )
    if boot_result is not None:
        npz_data["boot_coeff_samples"] = boot_result.coeff_samples
        npz_data["boot_inclusion_prob"] = boot_result.inclusion_probability
    np.savez_compressed(OUT_DIR / "results.npz", **npz_data)
    print(f"  Results: {OUT_DIR / 'results.npz'}")

    # ── Final banner ───────────────────────────────────────────────
    print(f"\n{sep}")
    print("  PIPELINE COMPLETE")
    print(f"{sep}")
    print(f"  Total time       : {total_time:.1f}s")
    print(f"  Discovered PDE   : {to_text(model)}")
    print(f"  Forecast R²(mean): {met['r2_mean']:.6f}")
    print(f"  Output directory  : {OUT_DIR}")
    print(f"\n  Plots:")
    for p in sorted(PLOTS_DIR.glob("*.png")):
        print(f"    {p.name}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
