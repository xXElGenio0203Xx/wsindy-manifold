#!/usr/bin/env python3
"""
ROM + WSINDy Pipeline
=====================

Extends ROM_pipeline.py to include WSINDy PDE discovery on raw density
fields, all running ON THE CLUSTER where training data lives in memory.

Key design:
  - Steps 1-6: Identical to ROM_pipeline.py (sims, POD, MVAR/LSTM, eval)
  - Step 7: WSINDy PDE discovery on training densities (still in memory!)
  - Step 8: WSINDy forecast on test runs + MVAR-compatible artifacts
  - Final: Export lightweight artifacts; heavy training densities stay on Oscar

This script is designed for the Brown Oscar cluster.  Locally you then run
``run_wsindy_rom.py --experiment <name> --local`` for visualization only.

Usage (on Oscar):
  python ROM_WSINDY_pipeline.py \\
      --config configs/DYN1_gentle_wsindy.yaml \\
      --experiment_name DYN1_gentle_wsindy

Usage (locally, after downloading test artifacts):
  python run_wsindy_rom.py --experiment DYN1_gentle_wsindy --local
"""

import numpy as np
import torch
from pathlib import Path
import json
import time
import argparse
import yaml
import shutil
import os
import sys
import gc

# Add src to path for all modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

# ── ROM pipeline imports ───────────────────────────────────────────
from rectsim.config_loader import load_config
from rectsim.ic_generator import generate_training_configs, generate_test_configs
from rectsim.simulation_runner import run_simulations_parallel
from rectsim.pod_builder import build_pod_basis, save_pod_basis
from rectsim.mvar_trainer import train_mvar_model, save_mvar_model, train_mvar_kstep
from rectsim.test_evaluator import evaluate_test_runs
from rectsim.forecast_utils import mvar_forecast_fn_factory
from rectsim.rom_data_utils import build_latent_dataset, build_multistep_latent_dataset
from rom.lstm_rom import LatentLSTMROM, train_lstm_rom, lstm_forecast_fn_factory, load_lstm_model
from rectsim.runtime_analyzer import RuntimeAnalyzer, compute_mvar_params, compute_lstm_params
import pandas as pd

# ── WSINDy imports ─────────────────────────────────────────────────
from wsindy.grid import GridSpec
from wsindy.system import (
    build_weak_system,
    default_t_margin,
    make_query_indices,
    nondimensionalize_field,
    rescale_coefficients,
)
from wsindy.test_functions import make_separable_psi
from wsindy.model import WSINDyModel
from wsindy.fit import wsindy_fit_regression
from wsindy.integrators import split_linear_nonlinear
from wsindy.forecast import wsindy_forecast
from wsindy.pretty import to_text, to_latex, group_terms
from wsindy.library import (
    LibraryBuilder,
    default_library,
    library_from_config,
    patch_feature_registries,
)
from wsindy.select import (
    TrialResult,
    SelectionResult,
    _pareto_frontier,
    _composite_score,
    default_ell_grid,
)
from wsindy.diagnostics import (
    ols_comparison,
    residual_analysis,
    dominant_balance_report,
    print_dominant_balance,
    model_aic,
)

# ── Multi-field WSINDy imports ─────────────────────────────────────
from wsindy.fields import (
    FieldData,
    build_field_data,
    build_field_data_rho_only,
    compute_flux_kde,
    compute_morse_potential,
)
from wsindy.multifield import (
    build_default_library as build_mf_library,
    library_from_config_multifield,
    resolve_regime_aware_library_settings,
    discover_multifield,
    model_selection_multifield,
    forecast_multifield,
    bootstrap_multifield,
    MultiFieldResult,
    MultiFieldForecastError,
    fit_equation_multifield,
)


# ═══════════════════════════════════════════════════════════════════
#  WSINDy helper functions
# ═══════════════════════════════════════════════════════════════════


def resolve_multifield_regime_settings(raw_config, wsindy_config):
    """Resolve regime-aware multifield WSINDy settings from the config."""
    mf_cfg = (wsindy_config or {}).get("multifield_library", {}) or {}
    forces_cfg = (raw_config or {}).get("forces", {}) or {}
    forces_params = forces_cfg.get("params", {}) or {}
    return resolve_regime_aware_library_settings(
        forces_enabled=bool(forces_cfg.get("enabled", False)),
        Ca=float(forces_params.get("Ca", 0.0) or 0.0),
        Cr=float(forces_params.get("Cr", 0.0) or 0.0),
        morse_requested=bool(mf_cfg.get("morse", True)),
        regime_class=mf_cfg.get("regime_class", "auto"),
    )

def load_training_densities_from_disk(train_dir, n_train, subsample=3, seed=42):
    """Load density trajectories from disk (when run locally with limited data).

    Returns list of (T_sub, nx, ny) arrays.
    """
    rng = np.random.default_rng(seed)
    available = sorted(
        d for d in train_dir.iterdir()
        if d.is_dir() and d.name.startswith("train_")
        and (d / "density.npz").exists()
    )
    if not available:
        raise FileNotFoundError(f"No training density files in {train_dir}")

    n = min(n_train, len(available))
    if n < n_train:
        print(f"    WARNING: Only {n} runs have density.npz (requested {n_train})")
    selected = rng.choice(available, size=n, replace=False) if n < len(available) else available

    densities = []
    for run_dir in selected:
        d = np.load(run_dir / "density.npz")
        rho = d["rho"][::subsample]
        densities.append(rho)
    return densities


def collect_training_densities_from_memory(
    train_dir, metadata, n_train, subsample=3, seed=42,
):
    """Load training density fields from disk into memory.

    On Oscar, training sims have just been run so density.npz exists
    for ALL runs. We select a diverse subset.
    """
    total_available = len(metadata)
    if n_train is None or n_train <= 0 or n_train >= total_available:
        selected = list(metadata)
        print(f"    Using all {len(selected)} available training trajectories")
    else:
        rng = np.random.default_rng(seed)

        # Group by IC distribution for stratified sampling
        by_type = {}
        for m in metadata:
            ic = m.get("distribution", "unknown")
            by_type.setdefault(ic, []).append(m)

        types = sorted(by_type.keys())
        per_type = max(1, n_train // len(types))
        remainder = n_train - per_type * len(types)

        selected = []
        for i, ic_type in enumerate(types):
            runs = by_type[ic_type]
            n = per_type + (1 if i < remainder else 0)
            n = min(n, len(runs))
            idx = rng.choice(len(runs), size=n, replace=False)
            for j in idx:
                selected.append(runs[j])
        selected = selected[:n_train]

    densities = []
    for meta in selected:
        run_dir = train_dir / meta["run_name"]
        d = np.load(run_dir / "density.npz")
        rho = d["rho"][::subsample]
        densities.append(rho)
        ic = meta.get("distribution", "?")
        print(f"      {meta['run_name']}: {rho.shape} ({ic})")

    return densities, selected


def align_eval_forecast_start(eval_config, base_config_test, rom_subsample, enabled_lags):
    """Align forecast start so every enabled ROM model gets a full window."""
    eval_cfg = dict(eval_config)
    dt_rom = base_config_test["sim"]["dt"] * rom_subsample
    requested_start_s = float(eval_cfg.get("forecast_start", base_config_test["sim"]["T"]))
    requested_steps = int(round(requested_start_s / dt_rom))
    max_lag = max((int(lag) for lag in enabled_lags if lag is not None), default=0)

    effective_steps = max(requested_steps, max_lag)
    effective_start_s = effective_steps * dt_rom

    eval_cfg["forecast_start_requested"] = requested_start_s
    eval_cfg["forecast_start_effective"] = effective_start_s
    eval_cfg["forecast_start_conditioning_steps"] = effective_steps
    eval_cfg["forecast_start_required_lag"] = max_lag
    eval_cfg["forecast_start"] = effective_start_s

    if max_lag and effective_steps > requested_steps:
        print(
            f"  ℹ️  forecast_start={requested_start_s}s → {effective_start_s}s "
            f"to provide a common {effective_steps}-step conditioning window "
            f"(max enabled lag={max_lag})"
        )

    return eval_cfg


def _run_post_regression_diagnostics(b, G, model, label, wsindy_dir):
    """Run all post-regression diagnostics on a fitted WSINDy model.

    Each diagnostic is independently wrapped in try/except so that a
    failure in one never blocks the others or the rest of the pipeline.

    Returns a JSON-serializable dict of numeric results.
    """
    diag_out = {}

    # 1. OLS comparison
    try:
        ols_res = ols_comparison(b, G, model)
        diag_out["ols_comparison"] = {
            "r2_ols": ols_res["r2_ols"],
            "r2_mstls": ols_res["r2_mstls"],
            "max_rel_diff": ols_res["max_rel_diff"],
            "per_term": ols_res["per_term"],
        }
        print(f"    OLS vs MSTLS — max relative diff: {ols_res['max_rel_diff']:.4f}")
        print(f"      R²(OLS)={ols_res['r2_ols']:.6f}  R²(MSTLS)={ols_res['r2_mstls']:.6f}")
    except Exception as exc:
        print(f"    [WARN] OLS comparison failed: {exc}")

    # 2. Residual analysis (with histogram + Q-Q plot)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        resid_res = residual_analysis(b, G, model, plot=True, ax=ax)
        fig.suptitle(f"Residual diagnostics — {label}", y=1.02)
        fig.tight_layout()
        save_path = Path(wsindy_dir) / f"residual_histogram_{label}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        diag_out["residual_analysis"] = {
            "mean": resid_res["mean"],
            "std": resid_res["std"],
            "skewness": resid_res["skewness"],
            "kurtosis": resid_res["kurtosis"],
            "shapiro_p": resid_res["shapiro_p"],
            "max_abs": resid_res["max_abs"],
        }
        print(f"    Residuals: μ={resid_res['mean']:.3e}, σ={resid_res['std']:.3e}, "
              f"kurtosis={resid_res['kurtosis']:.2f}, Shapiro p={resid_res['shapiro_p']:.3e}")
        print(f"      Saved: {save_path.name}")
    except Exception as exc:
        print(f"    [WARN] Residual analysis failed: {exc}")

    # 3. Dominant balance report
    try:
        balance = dominant_balance_report(b, G, model)
        # Normalized balance ratios (sum to 1 across active terms)
        Pi_sum = balance["Pi_sum"]
        Pi_normalized = {
            g["term"]: g["Pi"] / Pi_sum if Pi_sum > 1e-30 else 0.0
            for g in balance["groups"]
        }
        diag_out["dominant_balance"] = {
            "groups": [
                {"term": g["term"], "Pi": g["Pi"], "w": g["w"]}
                for g in balance["groups"]
            ],
            "Pi_sum": balance["Pi_sum"],
        }
        diag_out["dominant_balance_ratios"] = {
            g["term"]: g["Pi"] for g in balance["groups"]
        }
        diag_out["dominant_balance_normalized"] = Pi_normalized
        print(print_dominant_balance(balance))
    except Exception as exc:
        print(f"    [WARN] Dominant balance report failed: {exc})")

    # 4. AIC
    try:
        aic_val = model_aic(b, G, model)
        diag_out["aic"] = aic_val
        print(f"    AIC = {aic_val:.2f} (k={model.n_active} active terms)")
    except Exception as exc:
        print(f"    [WARN] AIC computation failed: {exc}")

    return diag_out


def build_stacked_weak_system(
    train_densities, grid, psi_bundle, library_terms, stride=(2, 2, 2),
):
    """Build stacked weak system (b, G) from multiple trajectories.

    Nondimensionalizes all trajectories by a shared U_c before assembly
    so that polynomial features are O(1), improving numerical stability.
    Returns (b, G, col_names, U_c).
    """
    # Compute shared characteristic scale from all trajectories
    all_stds = [float(np.std(U_k)) for U_k in train_densities]
    U_c = float(np.median(all_stds)) if all_stds else 1.0
    if U_c < 1e-30:
        U_c = float(max(np.max(np.abs(U_k)) for U_k in train_densities))
    if U_c < 1e-30:
        U_c = 1.0

    all_b, all_G = [], []
    t_margin = default_t_margin(psi_bundle)

    for U_k in train_densities:
        T_k, nx, ny = U_k.shape
        if 2 * t_margin >= T_k:
            continue
        qi = make_query_indices(
            T_k, nx, ny,
            stride_t=stride[0], stride_x=stride[1], stride_y=stride[2],
            t_margin=t_margin,
        )
        if qi.shape[0] < len(library_terms) + 1:
            continue
        b_k, G_k, col_names = build_weak_system(
            U_k / U_c, grid, psi_bundle, library_terms, qi,
        )
        all_b.append(b_k)
        all_G.append(G_k)

    if not all_b:
        raise ValueError("No valid query points from any training trajectory")

    return np.concatenate(all_b), np.vstack(all_G), col_names, U_c


def fit_stacked(
    train_densities, grid, library_terms, ell, p,
    stride=(2, 2, 2), lambdas=None, max_iter=25,
):
    """Fit WSINDy from stacked trajectories at a given ℓ."""
    psi_bundle = make_separable_psi(
        grid,
        ellt=ell[0], ellx=ell[1], elly=ell[2],
        pt=p[0], px=p[1], py=p[2],
    )
    b, G, col_names, U_c = build_stacked_weak_system(
        train_densities, grid, psi_bundle, library_terms, stride,
    )
    model = wsindy_fit_regression(b, G, col_names, lambdas=lambdas, max_iter=max_iter)
    # Rescale coefficients back to physical units
    model.w = rescale_coefficients(model.w, col_names, U_c)
    return model, b, G, col_names


def model_selection_stacked(
    train_densities, grid, library_terms, ell_grid, p,
    stride=(2, 2, 2), lambdas=None, max_iter=25,
    alpha=0.1, beta=0.01, cond_threshold=1e8,
    verbose=True,
):
    """Model selection over ℓ with multi-trajectory stacking."""
    if lambdas is None:
        lambdas = np.logspace(-4, 1, 40)

    n_library = len(library_terms)
    trials = []

    for idx, ell in enumerate(ell_grid):
        t0 = time.perf_counter()
        try:
            model, b, G, col_names = fit_stacked(
                train_densities, grid, library_terms, ell, p,
                stride, lambdas, max_iter,
            )
        except Exception as exc:
            if verbose:
                print(f"    [{idx+1}/{len(ell_grid)}] ℓ={ell} FAILED: {exc}")
            continue

        elapsed = time.perf_counter() - t0
        diag = model.diagnostics
        nloss = diag.get("normalised_loss", float("inf"))
        r2w = diag.get("r2", 0.0)

        active_cols = G[:, model.active]
        cond = float(np.linalg.cond(active_cols)) if active_cols.shape[1] > 0 else float("inf")

        score = _composite_score(
            nloss, model.n_active, n_library, cond, alpha, beta, cond_threshold,
        )

        trial = TrialResult(
            ell=ell, p=tuple(p), stride=tuple(stride), model=model,
            n_query=b.shape[0],
            normalised_loss=nloss, r2_weak=r2w,
            n_active=model.n_active, best_lambda=model.best_lambda,
            condition_number=cond, composite_score=score, elapsed_s=elapsed,
        )
        trials.append(trial)

        if verbose:
            print(
                f"    [{idx+1}/{len(ell_grid)}] ℓ={ell} "
                f"active={model.n_active}  loss={nloss:.4e}  "
                f"score={score:.4e}  ({elapsed:.2f}s)"
            )

    if not trials:
        raise RuntimeError("All model selection trials failed")

    ranked = sorted(trials, key=lambda t: t.composite_score)
    best = ranked[0]
    pareto = _pareto_frontier(trials)
    result = SelectionResult(trials=trials, best=best, pareto=pareto)

    # Rebuild best system for bootstrap
    _, best_b, best_G, _ = fit_stacked(
        train_densities, grid, library_terms,
        best.ell, tuple(p), tuple(stride), lambdas, max_iter,
    )
    return result, best_b, best_G


def bootstrap_from_system(b, G, col_names, B=50, lambdas=None, ci_alpha=0.05, seed=42):
    """Bootstrap UQ by resampling rows of pre-built weak system."""
    rng = np.random.default_rng(seed)
    if lambdas is None:
        lambdas = np.logspace(-4, 1, 40)

    N = b.shape[0]
    M = len(col_names)
    coeff_samples = np.zeros((B, M))
    active_counts = np.zeros(M)

    for rep in range(B):
        idx = rng.choice(N, size=N, replace=True)
        m = wsindy_fit_regression(b[idx], G[idx], col_names, lambdas=lambdas)
        coeff_samples[rep] = m.w
        active_counts += m.active.astype(float)

    return {
        "coeff_samples": coeff_samples,
        "coeff_mean": coeff_samples.mean(axis=0),
        "coeff_std": coeff_samples.std(axis=0),
        "inclusion_probability": active_counts / B,
        "ci_lo": np.percentile(coeff_samples, 100 * ci_alpha / 2, axis=0),
        "ci_hi": np.percentile(coeff_samples, 100 * (1 - ci_alpha / 2), axis=0),
        "B": B,
        "col_names": col_names,
    }


def forecast_density(model, grid, U0, n_steps, clip_negative=True):
    """Forecast density from IC using discovered PDE."""
    try:
        U_pred = wsindy_forecast(
            U0, model, grid, n_steps=n_steps,
            method="auto", clip_negative=clip_negative,
        )
        method = "auto"
    except Exception:
        U_pred = wsindy_forecast(
            U0, model, grid, n_steps=n_steps,
            method="rk4", clip_negative=clip_negative,
        )
        method = "rk4"

    if np.any(np.isnan(U_pred)) or np.max(np.abs(U_pred)) > 1e10:
        raise RuntimeError("Forecast diverged")

    lin, _ = split_linear_nonlinear(model)
    actual_method = "etdrk4" if (lin and method == "auto") else "rk4"
    return U_pred, actual_method


def compute_r2_timeseries(rho_true, rho_pred):
    T = min(rho_true.shape[0], rho_pred.shape[0])
    r2 = np.zeros(T)
    for t in range(T):
        ss_res = np.sum((rho_true[t] - rho_pred[t]) ** 2)
        ss_tot = np.sum((rho_true[t] - rho_true[t].mean()) ** 2)
        r2[t] = 1.0 - ss_res / max(ss_tot, 1e-30)
    return r2


def _compute_stationarity_stats(y_trajs):
    """
    Run ADF and KPSS tests on training latent trajectories to assess stationarity.

    Parameters
    ----------
    y_trajs : list of np.ndarray, each (T, D)
        List of latent trajectories from POD projection.

    Returns
    -------
    dict
        Aggregate statistics across all (trajectory, mode) pairs:
          adf_pval_mean / adf_pval_median  — ADF p-values (low  → stationary)
          kpss_pval_mean                   — KPSS p-values (low → non-stationary)
          nonstationary_frac               — fraction where ADF p > 0.05
          n_pairs_tested                   — number of (trajectory, mode) pairs

        ADF null = unit root (non-stationary) → p < 0.05 rejects unit root → stationary.
        KPSS null = stationary               → p < 0.05 rejects stationarity.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        from statsmodels.tsa.stattools import kpss as kpss_test
    except ImportError:
        return {"error": "statsmodels not available"}

    import warnings

    adf_pvals = []
    kpss_pvals = []

    for traj in y_trajs:
        T, D = traj.shape
        if T < 20:
            continue
        for d in range(D):
            x = traj[:, d]
            try:
                adf_pvals.append(float(adfuller(x, autolag='AIC')[1]))
            except Exception:
                pass
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kpss_pvals.append(float(kpss_test(x, regression='c', nlags='auto')[1]))
            except Exception:
                pass

    if not adf_pvals:
        return {"error": "no valid test results"}

    adf_arr = np.array(adf_pvals)
    kpss_arr = np.array(kpss_pvals) if kpss_pvals else None

    return {
        "adf_pval_mean": float(np.mean(adf_arr)),
        "adf_pval_median": float(np.median(adf_arr)),
        "kpss_pval_mean": float(np.mean(kpss_arr)) if kpss_arr is not None else None,
        "nonstationary_frac": float(np.mean(adf_arr > 0.05)),
        "n_pairs_tested": len(adf_pvals),
        "note": (
            "ADF null=unit_root (low p → stationary). "
            "KPSS null=stationary (low p → non-stationary)."
        ),
    }


# ═══════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ROM + WSINDy Pipeline (for Oscar cluster)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--wsindy-only", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    sep = "=" * 80

    print(f"\n{sep}")
    print("   ROM + WSINDy PIPELINE")
    print(f"{sep}")
    print(f"  Experiment: {args.experiment_name}")
    print(f"  Config:     {args.config}")
    print(f"  Timestamp:  {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load full config
    (BASE_CONFIG, DENSITY_NX, DENSITY_NY, DENSITY_BANDWIDTH,
     train_ic_config, test_ic_config, test_sim_config,
     rom_config, eval_config) = load_config(args.config)

    # Load raw YAML for WSINDy section
    with open(args.config) as f:
        raw_config = yaml.safe_load(f)
    wsindy_config = raw_config.get("wsindy", {})
    wsindy_enabled = wsindy_config.get("enabled", False)
    wsindy_identification_only = wsindy_config.get("identification_only", False)

    # Setup directories
    OUTPUT_DIR = Path(args.output_dir) if args.output_dir else Path(f"oscar_output/{args.experiment_name}")
    ROM_COMMON_DIR = OUTPUT_DIR / "rom_common"
    MVAR_DIR = OUTPUT_DIR / "MVAR"
    LSTM_DIR = OUTPUT_DIR / "LSTM"
    WSINDY_DIR = OUTPUT_DIR / "WSINDy"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if wsindy_enabled:
        WSINDY_DIR.mkdir(parents=True, exist_ok=True)
        (WSINDY_DIR / "plots").mkdir(exist_ok=True)

    shutil.copy(args.config, OUTPUT_DIR / "config_used.yaml")

    runtime_analyzer = RuntimeAnalyzer()
    runtime_profiles = []

    models_cfg = rom_config.get("models", {})
    mvar_enabled = models_cfg.get("mvar", {}).get("enabled", True)
    lstm_enabled = models_cfg.get("lstm", {}).get("enabled", False)
    if args.wsindy_only:
        mvar_enabled = False
        lstm_enabled = False
    rom_enabled = mvar_enabled or lstm_enabled

    if rom_enabled:
        ROM_COMMON_DIR.mkdir(parents=True, exist_ok=True)
    if mvar_enabled:
        MVAR_DIR.mkdir(parents=True, exist_ok=True)
    if lstm_enabled:
        LSTM_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n  Models:")
    print(f"    MVAR:   {'ON' if mvar_enabled else 'OFF'}")
    print(f"    LSTM:   {'ON' if lstm_enabled else 'OFF'}")
    print(f"    WSINDy: {'ON' if wsindy_enabled else 'OFF'}")
    if args.wsindy_only:
        print(f"    CLI override: wsindy-only")

    # ================================================================
    # STEP 1: Generate Training Data
    # ================================================================
    train_configs = generate_training_configs(train_ic_config, BASE_CONFIG)
    n_train = len(train_configs)

    TRAIN_DIR = OUTPUT_DIR / "train"
    pod_data = None
    R_POD = None
    T_rom = None
    M = None
    X_latent = None
    y_trajs = None

    _latent_cache = ROM_COMMON_DIR / "latent_dataset.npz"
    _pod_cache = ROM_COMMON_DIR / "pod_basis.npz"
    _resume_from_latent = (
        raw_config.get("resume_from_latent", False)
        and _latent_cache.exists()
        and _pod_cache.exists()
    )
    stationarity_stats = None  # Populated after POD basis build (if not resuming)

    if _resume_from_latent:
        print(f"\n{sep}")
        print("  STEP 1-3 SKIPPED: resuming from existing latent_dataset.npz")
        print(f"{sep}")
        print(f"  Found: {_latent_cache}")
    elif args.wsindy_only and TRAIN_DIR.exists() and (TRAIN_DIR / "metadata.json").exists():
        print(f"\n{sep}")
        print("  STEP 1 SKIPPED: --wsindy-only and training data already on disk")
        print(f"{sep}")
        print(f"  Found: {TRAIN_DIR}")
    else:
        print(f"\n{sep}")
        print("  STEP 1: Generating Training Data")
        print(f"{sep}")
        print(f"  Total runs: {n_train}")

        train_metadata, train_time = run_simulations_parallel(
            configs=train_configs,
            base_config=BASE_CONFIG,
            output_dir=OUTPUT_DIR,
            density_nx=DENSITY_NX,
            density_ny=DENSITY_NY,
            density_bandwidth=DENSITY_BANDWIDTH,
            is_test=False,
        )
        print(f"  Done in {train_time/60:.1f}m")

    if rom_enabled and not _resume_from_latent:
        # ================================================================
        # STEP 2: Build POD Basis (shared)
        # ================================================================
        print(f"\n{sep}")
        print("  STEP 2: Building Shared POD Basis")
        print(f"{sep}")

        pod_data = build_pod_basis(TRAIN_DIR, n_train, rom_config)
        save_pod_basis(pod_data, ROM_COMMON_DIR)

        R_POD = pod_data["R_POD"]
        T_rom = pod_data["T_rom"]
        M = pod_data["M"]
        # Reshape flat latent array (M*T_rom, R_POD) into list of per-run trajectories
        y_trajs = [pod_data["X_latent"][i * T_rom:(i + 1) * T_rom] for i in range(M)]
        print(f"  Latent dim: {R_POD}, T_rom: {T_rom}")

        # Stationarity tests on training latent trajectories (ADF + KPSS)
        print("  Computing stationarity statistics on training latent trajectories...")
        stationarity_stats = _compute_stationarity_stats(y_trajs)
        nstat = stationarity_stats.get("nonstationary_frac", float("nan"))
        print(f"  Nonstationary fraction (ADF p>0.05): {nstat:.1%}")

    elif rom_enabled and _resume_from_latent:
        # Load saved latent dataset directly — skip simulation and POD rebuild
        _cached = np.load(_latent_cache)
        X_all = _cached['X_all']
        Y_all = _cached['Y_all']
        lag = int(_cached['lag'])
        R_POD = X_all.shape[2]
        T_rom = None   # not persisted; lag guard skipped for numeric lags
        M = None
        y_trajs = None
        print(f"  Loaded: X_all={X_all.shape}, Y_all={Y_all.shape}, lag={lag}")
        print(f"  Latent dim: {R_POD}")

        # Reconstruct pod_data from saved POD basis (needed for evaluation)
        _pod_npz = np.load(_pod_cache)
        _X_mean = np.load(ROM_COMMON_DIR / "X_train_mean.npy")
        _sa_path = ROM_COMMON_DIR / "shift_align.npz"
        _sa_data = None
        if _sa_path.exists():
            _sa_npz = np.load(_sa_path)
            _sa_data = {k: _sa_npz[k] for k in _sa_npz.files}
        pod_data = {
            'U_r': _pod_npz['U'],
            'X_mean': _X_mean,
            'R_POD': R_POD,
            'shift_align': rom_config.get('shift_align', False),
            'shift_align_data': _sa_data,
            'density_transform': rom_config.get('density_transform', 'raw'),
            'density_transform_eps': rom_config.get('density_transform_eps', 1e-10),
        }
        print(f"  Reconstructed pod_data from {_pod_cache}")
    else:
        print(f"\n{sep}")
        print("  STEP 2-3: Skipping Shared POD/Latent ROM Setup")
        print(f"{sep}")
        print("  No MVAR/LSTM models enabled, so WSINDy runs directly on physical fields.")

    # Determine lag for each model independently
    # Supports symbolic lag values: "bic", "aic", "hqic", "fpe"
    # resolved from alvarez_lag_selection results in rom_hyperparameters/
    def _resolve_lag(raw_lag, default, T_rom_val):
        """Resolve a lag value that may be numeric or a symbolic IC name."""
        if raw_lag is None:
            return default
        if isinstance(raw_lag, (int, float)):
            lag_val = int(raw_lag)
        elif isinstance(raw_lag, str):
            ic_key = f"lag_{raw_lag.lower()}"  # e.g. "bic" -> "lag_bic"
            # Search for alvarez results matching this experiment
            alvarez_dirs = [
                Path("rom_hyperparameters/results") / args.experiment_name,
                Path("rom_hyperparameters/results_oscar_unaligned") / args.experiment_name,
            ]
            lag_val = None
            for adir in alvarez_dirs:
                summary_path = adir / "summary.json"
                if summary_path.exists():
                    with open(summary_path) as f:
                        alvarez_summary = json.load(f)
                    if ic_key in alvarez_summary:
                        lag_val = int(alvarez_summary[ic_key])
                        print(f"    Resolved symbolic lag '{raw_lag}' -> {lag_val} from {summary_path}")
                        break
            if lag_val is None:
                print(f"    WARNING: Could not resolve symbolic lag '{raw_lag}', "
                      f"falling back to default={default}")
                lag_val = default
        else:
            lag_val = default
        # Apply lag guard: min(lag, 30, T_rom - 2)
        if T_rom_val is not None:
            lag_cap = min(lag_val, 30, T_rom_val - 2)
            if lag_cap < lag_val:
                print(f"    Lag guard: capping {lag_val} -> {lag_cap} "
                      f"(30 cap or T_rom-2={T_rom_val - 2})")
                lag_val = lag_cap
        return lag_val

    raw_mvar_lag = models_cfg.get("mvar", {}).get("lag", 5) if mvar_enabled else None
    raw_lstm_lag = models_cfg.get("lstm", {}).get("lag", 20) if lstm_enabled else None
    mvar_lag = _resolve_lag(raw_mvar_lag, 5, T_rom) if mvar_enabled else None
    lstm_lag = _resolve_lag(raw_lstm_lag, 20, T_rom) if lstm_enabled else None

    # Primary lag for shared dataset (MVAR if enabled, else LSTM)
    lag = mvar_lag if mvar_enabled else lstm_lag

    if not _resume_from_latent:
        X_all = None
        Y_all = None
    Y_multi = None
    X_lstm = None
    Y_lstm = None
    lstm_effective_lag = None

    if mvar_enabled or lstm_enabled:
        if _resume_from_latent:
            # X_all / Y_all already loaded from latent_dataset.npz — skip rebuild
            print(f"  Using pre-built latent dataset (resume): X_all={X_all.shape}, lag={lag}")
        else:
            X_all, Y_all = build_latent_dataset(y_trajs, lag=lag)
            print(f"  X_all: {X_all.shape}, Y_all: {Y_all.shape}, lag={lag}")

        # Build separate LSTM dataset if LSTM lag differs from MVAR lag
        X_lstm = X_all
        Y_lstm = Y_all
        lstm_effective_lag = lag  # track what lag was actually used for LSTM data
        if lstm_enabled and mvar_enabled and lstm_lag != mvar_lag and not _resume_from_latent:
            print(f"\n  Building SEPARATE LSTM dataset: MVAR lag={mvar_lag}, LSTM lag={lstm_lag}")
            X_lstm, Y_lstm_sep = build_latent_dataset(y_trajs, lag=lstm_lag)
            Y_lstm = Y_lstm_sep
            lstm_effective_lag = lstm_lag
            print(f"  X_lstm: {X_lstm.shape}, Y_lstm: {Y_lstm.shape}, lag={lstm_lag}")

            # Multi-step targets for LSTM
            lstm_ms_cfg = models_cfg.get("lstm", {})
            ms_enabled = lstm_ms_cfg.get("multistep_loss", False)
            ms_k = lstm_ms_cfg.get("multistep_k", 5)
            if ms_enabled and ms_k > 1:
                X_lstm, Y_multi = build_multistep_latent_dataset(y_trajs, lag=lstm_lag, k_steps=ms_k)
                Y_lstm = Y_multi[:, 0, :]
        elif lstm_enabled and not _resume_from_latent:
            lstm_effective_lag = lag
            lstm_ms_cfg = models_cfg.get("lstm", {})
            ms_enabled = lstm_ms_cfg.get("multistep_loss", False)
            ms_k = lstm_ms_cfg.get("multistep_k", 5)
            if ms_enabled and ms_k > 1:
                X_lstm, Y_multi = build_multistep_latent_dataset(y_trajs, lag=lag, k_steps=ms_k)
                Y_lstm = Y_multi[:, 0, :]

        if not _resume_from_latent:
            np.savez(ROM_COMMON_DIR / "latent_dataset.npz", X_all=X_all, Y_all=Y_all, lag=lag)
    else:
        print("  Skipping latent dataset: no MVAR/LSTM models enabled")

    # ================================================================
    # Optional: delete train/ directory after POD + latent dataset are built
    # Controlled by top-level config key cleanup_train_after_pod: true
    # Use this to reclaim disk quota on HPC when raw simulations are not
    # needed downstream (ablation runs, lag sweeps, etc.).
    # ================================================================
    if raw_config.get("cleanup_train_after_pod", False) and TRAIN_DIR.exists():
        if wsindy_enabled:
            print(f"  cleanup_train_after_pod: skipped (WSINDy enabled — TRAIN_DIR needed downstream)")
        else:
            shutil.rmtree(TRAIN_DIR)
            print(f"  cleanup_train_after_pod: deleted {TRAIN_DIR}")

    # ================================================================
    # STEP 4a: Train MVAR (if enabled)
    # ================================================================
    mvar_data = None
    mvar_training_time = None
    if mvar_enabled:
        print(f"\n{sep}")
        print("  STEP 4a: Training MVAR")
        print(f"{sep}")

        mvar_cfg = models_cfg.get("mvar", {})
        kstep_k = mvar_cfg.get("kstep_k", 0)

        with runtime_analyzer.time_operation("mvar_training") as timer:
            if kstep_k and kstep_k > 1:
                mvar_data = train_mvar_kstep(pod_data, rom_config, y_trajs=y_trajs)
            else:
                mvar_data = train_mvar_model(pod_data, rom_config)
        mvar_training_time = timer.elapsed

        save_mvar_model(mvar_data, MVAR_DIR)
        print(f"  MVAR R²_train={mvar_data['r2_train']:.4f}  ({mvar_training_time:.2f}s)")

    # ================================================================
    # STEP 4b: Train LSTM (if enabled)
    # ================================================================
    lstm_data = None
    lstm_training_time = None
    if lstm_enabled:
        print(f"\n{sep}")
        print("  STEP 4b: Training LSTM")
        print(f"{sep}")
        lstm_config = rom_config["models"]["lstm"]
        with runtime_analyzer.time_operation("lstm_training") as timer:
            lstm_model_path, lstm_val_loss = train_lstm_rom(
                X_all=X_lstm, Y_all=Y_lstm,
                config={"rom": rom_config},
                out_dir=str(LSTM_DIR),
                Y_multi=Y_multi,
            )
        lstm_training_time = timer.elapsed
        print(f"  LSTM trained with effective lag={lstm_effective_lag} "
              f"(config MVAR lag={mvar_lag}, LSTM lag={lstm_lag})")
        lstm_data = {
            "model_path": lstm_model_path,
            "val_loss": lstm_val_loss,
            "lag": lstm_effective_lag,  # actual lag used for training data
            "hidden_units": lstm_config["hidden_units"],
            "num_layers": lstm_config["num_layers"],
        }
        print(f"  LSTM val_loss={lstm_val_loss:.6f}  ({lstm_training_time:.2f}s)")

    # ================================================================
    # STEP 5: Generate Test Data
    # ================================================================
    print(f"\n{sep}")
    print("  STEP 5: Generating Test Data")
    print(f"{sep}")

    test_configs = generate_test_configs(test_ic_config, BASE_CONFIG)
    n_test = len(test_configs)
    print(f"  Test runs: {n_test}")

    mean_r2_mvar = None
    mean_r2_lstm = None

    if n_test > 0:
        test_T = test_sim_config.get("T", test_ic_config.get("test_T", BASE_CONFIG["sim"]["T"]))
        BASE_CONFIG_TEST = BASE_CONFIG.copy()
        BASE_CONFIG_TEST["sim"] = BASE_CONFIG["sim"].copy()
        BASE_CONFIG_TEST["sim"]["T"] = test_T
        print(f"  Test horizon: {test_T}s")

        TEST_DIR = OUTPUT_DIR / "test"
        _test_meta_path = TEST_DIR / "metadata.json"
        if args.wsindy_only and TEST_DIR.exists() and _test_meta_path.exists():
            print("  STEP 5 SKIPPED: --wsindy-only and test data already on disk")
            with open(_test_meta_path) as _f:
                test_metadata = json.load(_f)
        else:
            test_metadata, test_time = run_simulations_parallel(
                configs=test_configs,
                base_config=BASE_CONFIG_TEST,
                output_dir=OUTPUT_DIR,
                density_nx=DENSITY_NX,
                density_ny=DENSITY_NY,
                density_bandwidth=DENSITY_BANDWIDTH,
                is_test=True,
            )
            print(f"  Done in {test_time/60:.1f}m")

        ROM_SUBSAMPLE = rom_config.get("subsample", 1)
        eval_config = align_eval_forecast_start(
            eval_config,
            BASE_CONFIG_TEST,
            ROM_SUBSAMPLE,
            [
                mvar_data["P_LAG"] if mvar_enabled and mvar_data is not None else None,
                lstm_data["lag"] if lstm_enabled and lstm_data is not None else None,
            ],
        )

        # ── 6a: MVAR eval ──
        if mvar_enabled:
            print(f"\n{sep}")
            print("  STEP 6a: Evaluating MVAR")
            print(f"{sep}")
            mvar_lag = mvar_data["P_LAG"]
            mvar_forecast_fn = mvar_forecast_fn_factory(mvar_data["model"], mvar_lag)
            test_results_df = evaluate_test_runs(
                test_dir=TEST_DIR, n_test=n_test,
                base_config_test=BASE_CONFIG_TEST,
                pod_data=pod_data, forecast_fn=mvar_forecast_fn,
                lag=mvar_lag,
                density_nx=DENSITY_NX, density_ny=DENSITY_NY,
                rom_subsample=ROM_SUBSAMPLE,
                eval_config=eval_config,
                train_T=BASE_CONFIG["sim"]["T"],
                model_name="MVAR",
            )
            test_results_df.to_csv(MVAR_DIR / "test_results.csv", index=False)
            mean_r2_mvar = test_results_df["r2_reconstructed"].mean()
            print(f"  MVAR mean R²: {mean_r2_mvar:.4f}")

        # ── 6b: LSTM eval ──
        if lstm_enabled:
            print(f"\n{sep}")
            print("  STEP 6b: Evaluating LSTM")
            print(f"{sep}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            lstm_lag = lstm_data["lag"]
            try:
                lstm_model, lstm_input_mean, lstm_input_std = load_lstm_model(
                    str(LSTM_DIR), device=str(device))
            except Exception:
                lstm_cfg_fb = rom_config.get("models", {}).get("lstm", {})
                lstm_model = LatentLSTMROM(
                    d=R_POD, hidden_units=lstm_data["hidden_units"],
                    num_layers=lstm_data["num_layers"],
                    dropout=lstm_cfg_fb.get("dropout", 0.0),
                    residual=lstm_cfg_fb.get("residual", False),
                    use_layer_norm=lstm_cfg_fb.get("use_layer_norm", True),
                )
                state = torch.load(lstm_data["model_path"], map_location=device, weights_only=False)
                if isinstance(state, dict) and "state_dict" in state:
                    lstm_model.load_state_dict(state["state_dict"])
                else:
                    lstm_model.load_state_dict(state)
                lstm_model.to(device); lstm_model.eval()
                lstm_input_mean, lstm_input_std = None, None

            lstm_forecast_fn = lstm_forecast_fn_factory(lstm_model, lstm_input_mean, lstm_input_std)
            test_results_df = evaluate_test_runs(
                test_dir=TEST_DIR, n_test=n_test,
                base_config_test=BASE_CONFIG_TEST,
                pod_data=pod_data, forecast_fn=lstm_forecast_fn,
                lag=lstm_lag,
                density_nx=DENSITY_NX, density_ny=DENSITY_NY,
                rom_subsample=ROM_SUBSAMPLE,
                eval_config=eval_config,
                train_T=BASE_CONFIG["sim"]["T"],
                model_name="LSTM",
            )
            test_results_df.to_csv(LSTM_DIR / "test_results.csv", index=False)
            mean_r2_lstm = test_results_df["r2_reconstructed"].mean()
            print(f"  LSTM mean R²: {mean_r2_lstm:.4f}")

    if rom_enabled:
        # Free latent artifacts before WSINDy discovery to reduce peak memory.
        del pod_data, X_latent, y_trajs, X_all, Y_all, Y_multi, X_lstm, Y_lstm
        gc.collect()

    # ================================================================
    # STEP 7: WSINDy PDE DISCOVERY  (training densities still on disk!)
    # ================================================================
    wsindy_model = None          # scalar mode
    wsindy_mf_result = None      # multifield mode
    wsindy_boot = None
    wsindy_summary = {}

    if wsindy_enabled:
        print(f"\n{sep}")
        print("  STEP 7: WSINDy PDE Discovery")
        print(f"{sep}")

        w_cfg = wsindy_config
        w_n_train = w_cfg.get("n_train", 30)
        w_subsample = w_cfg.get("subsample", rom_config.get("subsample", 3))
        w_seed = w_cfg.get("seed", 42)
        lib_cfg = w_cfg.get("library", {})
        ms_cfg = w_cfg.get("model_selection", {})
        lam_cfg = w_cfg.get("lambdas", {})
        boot_cfg = w_cfg.get("bootstrap", {})
        fc_cfg = w_cfg.get("forecast", {})

        wsindy_mode = w_cfg.get("mode", "scalar")
        print(f"  Mode: {wsindy_mode}")

        # ── Load training data ──────────────────────────────────────
        print(f"\n  Loading {w_n_train} training trajectories...")
        train_meta_path = TRAIN_DIR / "metadata.json"
        with open(train_meta_path) as f:
            train_meta_list = json.load(f)

        train_densities, selected_meta = collect_training_densities_from_memory(
            TRAIN_DIR, train_meta_list, w_n_train, w_subsample, w_seed,
        )
        print(f"  Loaded {len(train_densities)} density trajectories")

        # ── Grid ────────────────────────────────────────────────────
        ref = np.load(TRAIN_DIR / selected_meta[0]["run_name"] / "density.npz")
        xgrid = ref["xgrid"]
        ygrid = ref["ygrid"]
        dx = float(xgrid[1] - xgrid[0])
        dy = float(ygrid[1] - ygrid[0])
        Lx = float(xgrid[-1] - xgrid[0]) + dx
        Ly = float(ygrid[-1] - ygrid[0]) + dy
        dt_base = float(ref["times"][1] - ref["times"][0])
        dt = dt_base * w_subsample
        grid = GridSpec(dt=dt, dx=dx, dy=dy)
        T_sub, nx, ny = train_densities[0].shape
        print(f"  Grid: dt={dt:.4f}, dx={dx:.4f}, dy={dy:.4f}")
        print(f"  Domain: Lx={Lx:.2f}, Ly={Ly:.2f}")
        print(f"  Shape: ({T_sub}, {nx}, {ny}) per trajectory")

        # ── Common model-selection parameters ───────────────────────
        n_ell = ms_cfg.get("n_ell", 12)
        p = tuple(ms_cfg.get("p", [3, 5, 5]))
        stride = tuple(ms_cfg.get("stride", [2, 2, 2]))
        alpha = ms_cfg.get("alpha", 0.1)
        beta = ms_cfg.get("beta", 0.01)
        cond_thr = ms_cfg.get("cond_threshold", 1e8)

        log_min = lam_cfg.get("log_min", -5)
        log_max = lam_cfg.get("log_max", 2)
        n_lam = lam_cfg.get("n_points", 60)
        lambdas = np.logspace(log_min, log_max, n_lam)

        ell_grid = default_ell_grid(T_sub, nx, ny, n_points=n_ell)

        # ==============================================================
        #  MULTIFIELD MODE: 3-equation system (ρ, p_x, p_y)
        # ==============================================================
        if wsindy_mode == "multifield":
            mf_cfg = w_cfg.get("multifield_library", {})
            rich = mf_cfg.get("rich", False)
            rho_strategy = mf_cfg.get("rho_strategy", "legacy")
            kde_bw = mf_cfg.get("kde_bandwidth", 5.0)
            regime_settings = resolve_multifield_regime_settings(raw_config, w_cfg)
            regime_class = regime_settings["regime_class"]
            effective_morse = regime_settings["effective_morse"]
            morse_requested = regime_settings["morse_requested"]

            # Morse parameters from forces config
            forces_enabled = bool(raw_config.get("forces", {}).get("enabled", False))
            forces_params = raw_config.get("forces", {}).get("params", {})
            Ca = forces_params.get("Ca", 0.8)
            Cr = forces_params.get("Cr", 0.3)
            la = forces_params.get("la", 1.5)
            lr = forces_params.get("lr", 0.5)

            # ── Build 3-equation library ────────────────────────────
            mf_library = build_mf_library(
                morse=effective_morse,
                rich=rich,
                rho_strategy=rho_strategy,
                regime_class=regime_class,
            )
            n_lib_total = sum(len(v) for v in mf_library.values())
            print(f"\n  Multi-field library: {n_lib_total} terms ({len(mf_library)} equations)")
            print(
                "  Regime class:"
                f" {regime_class} ({regime_settings['regime_class_source']}),"
                f" forces_enabled={forces_enabled},"
                f" Ca/Cr={regime_settings['ca_cr_ratio']:.4g},"
                f" morse={effective_morse}"
            )
            for eq_name, terms in mf_library.items():
                print(f"    {eq_name}: {len(terms)} terms")
                for t in terms:
                    danger_flag = " [!]" if t.dangerous else ""
                    print(f"      • {t.name}{danger_flag}")

            # ── Build FieldData per training trajectory ─────────────
            print(f"\n  Building FieldData for {len(train_densities)} training trajectories...")
            field_data_list = []
            for i, meta in enumerate(selected_meta):
                run_dir = TRAIN_DIR / meta["run_name"]
                rho_i = train_densities[i]

                # Try to load trajectory data for flux computation
                traj_path = run_dir / "trajectory.npz"
                if traj_path.exists():
                    td = np.load(traj_path)
                    traj_i = td["traj"][::w_subsample]
                    vel_i = td["vel"][::w_subsample]
                    fd = build_field_data(
                        rho_i, traj_i, vel_i,
                        xgrid, ygrid, Lx, Ly, dt,
                        bandwidth=kde_bw,
                        morse_params=dict(Cr=Cr, Ca=Ca, lr=lr, la=la) if effective_morse else None,
                        center_flux=True,
                    )
                else:
                    print(f"    WARNING: {meta['run_name']} has no trajectory.npz, using rho-only")
                    fd = build_field_data_rho_only(
                        rho_i, xgrid, ygrid, Lx, Ly, dt,
                        morse_params=dict(Cr=Cr, Ca=Ca, lr=lr, la=la) if effective_morse else None,
                    )

                field_data_list.append(fd)
                if i == 0 or (i + 1) % 10 == 0:
                    print(f"    [{i+1}/{len(train_densities)}] {meta['run_name']}: OK")

            # ── Multi-field model selection ──────────────────────────
            print(f"\n  Model selection: {len(ell_grid)} ℓ configs × {len(field_data_list)} trajectories")
            print(f"  (3 equations fit independently at each ℓ)")

            t_ms = time.perf_counter()
            wsindy_mf_result, best_ell = model_selection_multifield(
                field_data_list, mf_library, ell_grid,
                p=p, stride=stride, lambdas=lambdas,
                rho_strategy=rho_strategy,
                morse_params=dict(Cr=Cr, Ca=Ca, lr=lr, la=la) if effective_morse else None,
            )
            ms_time = time.perf_counter() - t_ms
            wsindy_mf_result.metadata.update(regime_settings)

            print(f"\n  Model selection done in {ms_time:.1f}s")
            print(f"  Best ℓ = {best_ell}")
            print(f"\n  Discovered 3-equation PDE system:")
            print(wsindy_mf_result.summary())

            # ── Bootstrap UQ ────────────────────────────────────────
            if boot_cfg.get("enabled", True):
                B = boot_cfg.get("B", 50)
                ci_alpha_val = boot_cfg.get("ci_alpha", 0.05)
                print(f"\n  Bootstrap: {B} replicates (trajectory resampling)...")
                t_boot = time.perf_counter()
                wsindy_boot = bootstrap_multifield(
                    field_data_list, mf_library,
                    ell=best_ell, p=p, stride=stride,
                    lambdas=lambdas, B=B, seed=w_seed,
                )
                boot_time = time.perf_counter() - t_boot
                print(f"  Done in {boot_time:.1f}s")

                for eq_name in ["rho", "px", "py"]:
                    if eq_name in wsindy_boot:
                        bdata = wsindy_boot[eq_name]
                        print(f"\n  Bootstrap — {eq_name} equation:")
                        print(f"    {'Term':>20s}  {'Mean':>12s}  {'Std':>10s}  {'P(active)':>10s}")
                        for j, nm in enumerate(bdata["col_names"]):
                            if bdata["inclusion_probability"][j] > 0.01:
                                print(f"    {nm:>20s}  {bdata['coeff_mean'][j]:+12.4e}  "
                                      f"{bdata['coeff_std'][j]:10.4e}  "
                                      f"{bdata['inclusion_probability'][j]:10.3f}")

            # ── Save multi-field result ─────────────────────────────
            mf_dict = wsindy_mf_result.to_dict()
            with open(WSINDY_DIR / "multifield_model.json", "w") as f:
                json.dump(mf_dict, f, indent=2, default=str)
            with open(WSINDY_DIR / "multifield_diagnostics.json", "w") as f:
                json.dump(wsindy_mf_result.metadata, f, indent=2, default=str)

            # Also save per-equation npz for lightweight loading
            for eq_name in ["rho", "px", "py"]:
                mdl = getattr(wsindy_mf_result, f"{eq_name}_model")
                np.savez(
                    WSINDY_DIR / f"wsindy_model_{eq_name}.npz",
                    col_names=np.array(mdl.col_names),
                    w=mdl.w,
                    active=mdl.active,
                    best_lambda=mdl.best_lambda,
                    col_scale=mdl.col_scale,
                    diagnostics_r2=mdl.diagnostics.get("r2", 0.0),
                    diagnostics_nloss=mdl.diagnostics.get("normalised_loss", float("inf")),
                )

            if wsindy_boot is not None:
                ci_alpha_val = boot_cfg.get("ci_alpha", 0.05)
                for eq_name in ["rho", "px", "py"]:
                    if eq_name in wsindy_boot:
                        bdata = wsindy_boot[eq_name]
                        samples = bdata["coeff_samples"]
                        ci_lo = np.percentile(samples, 100 * ci_alpha_val / 2, axis=0)
                        ci_hi = np.percentile(samples, 100 * (1 - ci_alpha_val / 2), axis=0)
                        np.savez(
                            WSINDY_DIR / f"bootstrap_{eq_name}.npz",
                            coeff_samples=samples,
                            coeff_mean=bdata["coeff_mean"],
                            coeff_std=bdata["coeff_std"],
                            inclusion_probability=bdata["inclusion_probability"],
                            ci_lo=ci_lo,
                            ci_hi=ci_hi,
                            col_names=np.array(bdata["col_names"]),
                        )

            # ── Build summary ───────────────────────────────────────
            wsindy_summary = {
                "mode": "multifield",
                "discovered_pde": {},
                "model_selection": {
                    "best_ell": list(best_ell),
                    "time_s": round(ms_time, 1),
                },
                "library": {
                    "n_terms_total": n_lib_total,
                    "morse": effective_morse,
                    "morse_requested": morse_requested,
                    "rich": rich,
                    "rho_strategy": rho_strategy,
                    "regime_class": regime_class,
                    "per_equation": {
                        eq: [t.name for t in terms]
                        for eq, terms in mf_library.items()
                    },
                },
                "regime_classification": dict(regime_settings),
                "n_train_trajectories": len(train_densities),
                "diagnostics": wsindy_mf_result.metadata,
            }
            for eq_name in ["rho", "px", "py"]:
                mdl = getattr(wsindy_mf_result, f"{eq_name}_model")
                wsindy_summary["discovered_pde"][eq_name] = {
                    "text": to_text(mdl),
                    "active_terms": mdl.active_terms,
                    "coefficients": {
                        n: float(mdl.w[mdl.col_names.index(n)])
                        for n in mdl.active_terms
                    },
                    "n_active": mdl.n_active,
                    "lambda_star": float(mdl.best_lambda),
                    "r2_weak": float(mdl.diagnostics.get("r2", 0)),
                }

            # ── Post-regression diagnostics (multifield) ───────────
            print(f"\n  Post-regression diagnostics (multifield)...")
            _target_fns = {
                "rho": lambda fd: fd.rho,
                "px":  lambda fd: fd.px,
                "py":  lambda fd: fd.py,
            }
            mf_diag = {}
            for eq_name in ["rho", "px", "py"]:
                mdl = getattr(wsindy_mf_result, f"{eq_name}_model")
                print(f"\n    ── {eq_name} equation ──")
                try:
                    _, eq_b, eq_G, _ = fit_equation_multifield(
                        field_data_list, mf_library[eq_name],
                        _target_fns[eq_name],
                        ell=best_ell, p=p, stride=stride, lambdas=lambdas,
                    )
                    mf_diag[eq_name] = _run_post_regression_diagnostics(
                        eq_b, eq_G, mdl, eq_name, WSINDY_DIR,
                    )
                except Exception as exc:
                    print(f"    [WARN] Diagnostics for {eq_name} failed: {exc}")
            if mf_diag:
                wsindy_summary["post_regression_diagnostics"] = mf_diag

        # ==============================================================
        #  SCALAR MODE: Single ρ equation (original fallback)
        # ==============================================================
        else:
            # Build rich scalar library
            library_terms = library_from_config(lib_cfg)
            patch_feature_registries()

            print(f"  Scalar library: {len(library_terms)} candidate terms")
            for op, feat in library_terms:
                print(f"    {op}:{feat}")

            print(f"\n  Model selection: {len(ell_grid)} ℓ configs × {len(train_densities)} trajectories")

            t_ms = time.perf_counter()
            sel_result, best_b, best_G = model_selection_stacked(
                train_densities, grid, library_terms, ell_grid, p,
                stride=stride, lambdas=lambdas,
                alpha=alpha, beta=beta, cond_threshold=cond_thr,
            )
            ms_time = time.perf_counter() - t_ms

            wsindy_model = sel_result.best.model
            col_names = wsindy_model.col_names
            print(f"\n  Model selection done in {ms_time:.1f}s")
            print(sel_result.summary())
            print(f"\n  Discovered PDE:")
            print(f"    {to_text(wsindy_model)}")

            # ── Bootstrap UQ ────────────────────────────────────────
            if boot_cfg.get("enabled", True):
                B = boot_cfg.get("B", 50)
                ci_alpha_val = boot_cfg.get("ci_alpha", 0.05)
                print(f"\n  Bootstrap: {B} replicates...")
                t_boot = time.perf_counter()
                wsindy_boot = bootstrap_from_system(
                    best_b, best_G, col_names, B=B, lambdas=lambdas,
                    ci_alpha=ci_alpha_val, seed=w_seed,
                )
                boot_time = time.perf_counter() - t_boot
                print(f"  Done in {boot_time:.1f}s")

                print(f"\n  {'Term':>14s}  {'Mean':>12s}  {'Std':>10s}  {'P(active)':>10s}")
                for i, nm in enumerate(col_names):
                    if wsindy_boot["inclusion_probability"][i] > 0.01:
                        print(f"  {nm:>14s}  {wsindy_boot['coeff_mean'][i]:+12.4e}  "
                              f"{wsindy_boot['coeff_std'][i]:10.4e}  "
                              f"{wsindy_boot['inclusion_probability'][i]:10.3f}")

            # ── Save scalar model ───────────────────────────────────
            np.savez(
                WSINDY_DIR / "wsindy_model.npz",
                col_names=np.array(col_names),
                w=wsindy_model.w,
                active=wsindy_model.active,
                best_lambda=wsindy_model.best_lambda,
                col_scale=wsindy_model.col_scale,
                diagnostics_r2=wsindy_model.diagnostics.get("r2", 0.0),
                diagnostics_nloss=wsindy_model.diagnostics.get("normalised_loss", float("inf")),
            )

            if wsindy_boot is not None:
                np.savez(
                    WSINDY_DIR / "bootstrap.npz",
                    coeff_samples=wsindy_boot["coeff_samples"],
                    coeff_mean=wsindy_boot["coeff_mean"],
                    coeff_std=wsindy_boot["coeff_std"],
                    inclusion_probability=wsindy_boot["inclusion_probability"],
                    ci_lo=wsindy_boot["ci_lo"],
                    ci_hi=wsindy_boot["ci_hi"],
                    col_names=np.array(col_names),
                )

            wsindy_summary = {
                "mode": "scalar",
                "discovered_pde": {
                    "text": to_text(wsindy_model),
                    "latex": to_latex(wsindy_model),
                    "active_terms": wsindy_model.active_terms,
                    "coefficients": {
                        n: float(wsindy_model.w[wsindy_model.col_names.index(n)])
                        for n in wsindy_model.active_terms
                    },
                    "n_active": wsindy_model.n_active,
                    "n_library": len(library_terms),
                    "lambda_star": float(wsindy_model.best_lambda),
                    "r2_weak": float(wsindy_model.diagnostics.get("r2", 0)),
                },
                "model_selection": {
                    "n_trials": len(sel_result.trials),
                    "best_ell": list(sel_result.best.ell),
                    "time_s": round(ms_time, 1),
                },
                "library": {
                    "n_terms": len(library_terms),
                    "terms": [f"{op}:{feat}" for op, feat in library_terms],
                },
                "n_train_trajectories": len(train_densities),
            }

            # ── Post-regression diagnostics (scalar) ───────────────
            print(f"\n  Post-regression diagnostics (scalar)...")
            scalar_diag = _run_post_regression_diagnostics(
                best_b, best_G, wsindy_model, "scalar", WSINDY_DIR,
            )
            if scalar_diag:
                wsindy_summary["post_regression_diagnostics"] = scalar_diag

    # ================================================================
    # STEP 8: WSINDy Test Evaluation
    # ================================================================
    mean_r2_wsindy = None

    wsindy_has_model = (wsindy_model is not None) or (wsindy_mf_result is not None)
    if wsindy_enabled and n_test > 0 and wsindy_has_model:
        print(f"\n{sep}")
        print("  STEP 8: WSINDy Test Evaluation")
        print(f"{sep}")

        TEST_DIR = OUTPUT_DIR / "test"
        test_meta_path = TEST_DIR / "metadata.json"
        with open(test_meta_path) as f:
            test_meta_list = json.load(f)

        w_subsample = wsindy_config.get("subsample", rom_config.get("subsample", 3))
        forecast_start_s = eval_config.get(
            "forecast_start_effective",
            eval_config.get("forecast_start", BASE_CONFIG["sim"]["T"]),
        )
        forecast_start = int(round(forecast_start_s / (BASE_CONFIG_TEST["sim"]["dt"] * w_subsample)))
        fc_cfg = wsindy_config.get("forecast", {})
        clip_neg = fc_cfg.get("clip_negative", True)

        test_results = []
        all_r2 = []

        def _motion_energy(field_hist):
            if field_hist.shape[0] <= 1:
                return 0.0
            diffs = np.diff(field_hist, axis=0)
            return float(np.mean(np.linalg.norm(diffs.reshape(diffs.shape[0], -1), axis=1)))

        def _coeff_for_model(model, name):
            if model is None or name not in model.col_names:
                return None
            idx = model.col_names.index(name)
            if not model.active[idx]:
                return None
            return float(model.w[idx])

        def _sign_label(value, near=1e-6):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return "inactive"
            if abs(value) < near:
                return "near_zero"
            return "positive" if value > 0 else "negative"

        selected_ell_vals = [None, None, None]
        selected_ell_json = None
        weak_r2_rho = float("nan")
        weak_r2_px = float("nan")
        weak_r2_py = float("nan")
        div_p_coeff = float("nan")
        lap_rho_coeff = float("nan")
        lap_rho_sign = "inactive"
        px_linear_coeff = float("nan")
        py_linear_coeff = float("nan")
        cond_G_rho = float("nan")
        cond_G_px = float("nan")
        cond_G_py = float("nan")
        regime_class_result = None
        regime_class_source = None
        forces_enabled_result = None
        ca_cr_ratio_result = float("nan")
        effective_morse_result = None
        if wsindy_mf_result is not None:
            best_ell_meta = wsindy_summary.get("model_selection", {}).get("best_ell")
            if best_ell_meta is not None and len(best_ell_meta) == 3:
                selected_ell_vals = [int(v) for v in best_ell_meta]
                selected_ell_json = json.dumps(selected_ell_vals)
            weak_r2_rho = float(wsindy_mf_result.rho_model.diagnostics.get("r2", float("nan")))
            weak_r2_px = float(wsindy_mf_result.px_model.diagnostics.get("r2", float("nan")))
            weak_r2_py = float(wsindy_mf_result.py_model.diagnostics.get("r2", float("nan")))
            div_p_coeff = _coeff_for_model(wsindy_mf_result.rho_model, "div_p")
            lap_rho_coeff = _coeff_for_model(wsindy_mf_result.rho_model, "lap_rho")
            lap_rho_sign = _sign_label(lap_rho_coeff)
            px_linear_coeff = _coeff_for_model(wsindy_mf_result.px_model, "px")
            py_linear_coeff = _coeff_for_model(wsindy_mf_result.py_model, "py")
            fit_diagnostics = wsindy_mf_result.metadata.get("fit_diagnostics", {})
            cond_G_rho = float(fit_diagnostics.get("rho", {}).get("condition_number", float("nan")))
            cond_G_px = float(fit_diagnostics.get("px", {}).get("condition_number", float("nan")))
            cond_G_py = float(fit_diagnostics.get("py", {}).get("condition_number", float("nan")))
            regime_class_result = wsindy_mf_result.metadata.get("regime_class")
            regime_class_source = wsindy_mf_result.metadata.get("regime_class_source")
            forces_enabled_result = wsindy_mf_result.metadata.get("forces_enabled")
            ca_cr_ratio_result = float(wsindy_mf_result.metadata.get("ca_cr_ratio", float("nan")))
            effective_morse_result = wsindy_mf_result.metadata.get("effective_morse")

        # ── Identification-only vs forecast evaluation ──────────────
        if not wsindy_identification_only:
            # ============================================================
            #  FORECAST PATH (original behavior)
            # ============================================================
            for ti in range(n_test):
                run_name = test_meta_list[ti]["run_name"]
                run_dir = TEST_DIR / run_name

                d = np.load(run_dir / "density_true.npz")
                rho_true = d["rho"][::w_subsample]
                times_sub = d["times"][::w_subsample]
                T_test_sub = rho_true.shape[0]

                n_fc = T_test_sub - forecast_start - 1
                if n_fc <= 0:
                    continue

                try:
                    # ==================================================
                    #  MULTIFIELD forecast
                    # ==================================================
                    failure_reason = None
                    failure_step = None
                    method_used = None
                    if wsindy_mf_result is not None:
                        rho0 = rho_true[forecast_start]

                        # Build initial flux fields from test trajectory
                        traj_path = run_dir / "trajectory.npz"
                        if traj_path.exists():
                            td = np.load(traj_path)
                            traj_test = td["traj"][::w_subsample]
                            vel_test = td["vel"][::w_subsample]
                            mf_cfg = wsindy_config.get("multifield_library", {})
                            kde_bw = mf_cfg.get("kde_bandwidth", 5.0)
                            # compute_flux_kde expects (T, N, 2) — wrap single frame
                            px0_arr, py0_arr = compute_flux_kde(
                                traj_test[forecast_start:forecast_start+1],
                                vel_test[forecast_start:forecast_start+1],
                                xgrid, ygrid, Lx, Ly, bandwidth=kde_bw,
                            )
                            px0, py0 = px0_arr[0], py0_arr[0]
                        else:
                            # No trajectory → zero flux ICs (degraded mode)
                            print(f"    WARNING: {run_name} has no trajectory.npz, using zero-flux ICs")
                            px0 = np.zeros_like(rho0)
                            py0 = np.zeros_like(rho0)

                        # Morse params
                        morse_params = None
                        forces_params_fc = raw_config.get("forces", {}).get("params", {})
                        if effective_morse_result:
                            morse_params = dict(
                                Cr=forces_params_fc.get("Cr", 0.3),
                                Ca=forces_params_fc.get("Ca", 0.8),
                                lr=forces_params_fc.get("lr", 0.5),
                                la=forces_params_fc.get("la", 1.5),
                            )

                        fc_method = fc_cfg.get("method", "auto")
                        mass_conserve = fc_cfg.get("mass_conserve", True)
                        try:
                            rho_pred, px_pred, py_pred = forecast_multifield(
                                rho0, px0, py0,
                                wsindy_mf_result, grid,
                                Lx=Lx, Ly=Ly,
                                n_steps=n_fc,
                                clip_negative_rho=clip_neg,
                                mass_conserve=mass_conserve,
                                method=fc_method,
                                morse_params=morse_params,
                                xgrid=xgrid, ygrid=ygrid,
                            )
                            method_used = wsindy_mf_result.metadata.get("last_forecast_method_used", fc_method)
                        except Exception as auto_exc:
                            if fc_method != "auto":
                                raise
                            print(f"    [{ti+1}/{n_test}] {run_name}: auto multifield forecast failed - {auto_exc}")
                            print("    Retrying multifield forecast with RK4...")
                            try:
                                rho_pred, px_pred, py_pred = forecast_multifield(
                                    rho0, px0, py0,
                                    wsindy_mf_result, grid,
                                    Lx=Lx, Ly=Ly,
                                    n_steps=n_fc,
                                    clip_negative_rho=clip_neg,
                                    mass_conserve=mass_conserve,
                                    method="rk4",
                                    morse_params=morse_params,
                                    xgrid=xgrid, ygrid=ygrid,
                                )
                                method_used = wsindy_mf_result.metadata.get("last_forecast_method_used", "rk4")
                            except Exception as rk4_exc:
                                raise MultiFieldForecastError(
                                    f"WSINDy multifield forecast failed with auto and RK4. "
                                    f"Auto error: {auto_exc}. RK4 error: {rk4_exc}"
                                    ,
                                    reason="fallback_failed",
                                    step=getattr(rk4_exc, "step", getattr(auto_exc, "step", None)),
                                    method=getattr(rk4_exc, "method", "rk4"),
                                ) from rk4_exc
                        method = f"{method_used}_multifield"
                        # Stack with IC → same shape as scalar path
                        U_pred = rho_pred  # already includes IC at index 0

                    # ==================================================
                    #  SCALAR forecast (fallback)
                    # ==================================================
                    else:
                        U0 = rho_true[forecast_start]
                        U_pred, method = forecast_density(
                            wsindy_model, grid, U0, n_fc, clip_negative=clip_neg,
                        )

                except Exception as e:
                    print(f"    [{ti+1}/{n_test}] {run_name}: FAILED - {e}")
                    import traceback; traceback.print_exc()
                    test_results.append({
                        "test_id": ti, "run_name": run_name,
                        "r2_reconstructed": float("nan"),
                        "forecast_status": "failed",
                        "failure_reason": getattr(e, "reason", type(e).__name__),
                        "failure_step": getattr(e, "step", None),
                        "forecast_method_attempted": fc_cfg.get("method", "auto") if wsindy_mf_result is not None else "scalar",
                        "forecast_method_used": getattr(e, "method", None),
                        "motion_energy_true": float("nan"),
                        "motion_energy_pred": float("nan"),
                        "motion_ratio": float("nan"),
                        "mass_drift_mean": float("nan"),
                        "frame_50_rho_rmse": float("nan"),
                        "mass_drift_frame_50": float("nan"),
                        "selected_ell": selected_ell_json,
                        "selected_ell_t": selected_ell_vals[0],
                        "selected_ell_x": selected_ell_vals[1],
                        "selected_ell_y": selected_ell_vals[2],
                        "weak_r2_rho": weak_r2_rho,
                        "weak_r2_px": weak_r2_px,
                        "weak_r2_py": weak_r2_py,
                        "div_p_coeff": div_p_coeff,
                        "lap_rho_sign": lap_rho_sign,
                        "lap_rho_coeff": lap_rho_coeff,
                        "px_linear_coeff": px_linear_coeff,
                        "py_linear_coeff": py_linear_coeff,
                        "cond_G_rho": cond_G_rho,
                        "cond_G_px": cond_G_px,
                        "cond_G_py": cond_G_py,
                        "regime_class": regime_class_result,
                        "regime_class_source": regime_class_source,
                        "forces_enabled": forces_enabled_result,
                        "ca_cr_ratio": ca_cr_ratio_result,
                        "effective_morse": effective_morse_result,
                    })
                    continue

                rho_true_fc = rho_true[forecast_start : forecast_start + n_fc + 1]
                times_fc = times_sub[forecast_start : forecast_start + n_fc + 1]

                r2_ts = compute_r2_timeseries(rho_true_fc, U_pred)
                mean_r2 = float(np.nanmean(r2_ts[1:]))

                # Save per-test artifacts
                save_dict = dict(
                    rho=U_pred.astype(np.float32),
                    xgrid=d["xgrid"], ygrid=d["ygrid"],
                    times=np.asarray(times_fc, dtype=np.float32),
                    forecast_start_idx=0,
                )
                if wsindy_mf_result is not None:
                    # Also save predicted flux fields (already include IC at index 0)
                    save_dict["px"] = px_pred.astype(np.float32)
                    save_dict["py"] = py_pred.astype(np.float32)
                np.savez_compressed(run_dir / "density_pred_wsindy.npz", **save_dict)

                n_save = min(len(times_fc), len(r2_ts))
                pd.DataFrame({
                    "time": times_fc[:n_save],
                    "r2_reconstructed": r2_ts[:n_save],
                    "r2_latent": r2_ts[:n_save],
                    "r2_pod": np.ones(n_save, dtype=np.float64),
                }).to_csv(run_dir / "r2_vs_time_wsindy.csv", index=False)

                # Save density_metrics_wsindy.csv (analogous to MVAR/LSTM)
                try:
                    T_pred = U_pred.shape[0]
                    density_var_pred = np.std(U_pred, axis=(1, 2))
                    mass_pred = np.sum(U_pred, axis=(1, 2))
                    density_var_true = np.std(rho_true_fc[:T_pred], axis=(1, 2))
                    mass_true_fc = np.sum(rho_true_fc[:T_pred], axis=(1, 2))
                    motion_energy_true = _motion_energy(rho_true_fc[:T_pred])
                    motion_energy_pred = _motion_energy(U_pred[:T_pred])
                    motion_ratio = (
                        float(motion_energy_pred / motion_energy_true)
                        if motion_energy_true > 1e-12 else 1.0
                    )
                    mass_drift_mean = float(np.mean(np.abs(mass_pred - mass_true_fc) / np.maximum(np.abs(mass_true_fc), 1e-12)))
                    if T_pred > 50:
                        frame_50_rho_rmse = float(np.sqrt(np.mean((rho_true_fc[50] - U_pred[50]) ** 2)))
                        mass_drift_frame_50 = float(
                            abs(np.sum(U_pred[50]) - np.sum(U_pred[0])) / max(abs(np.sum(U_pred[0])), 1e-12)
                        )
                    else:
                        frame_50_rho_rmse = float("nan")
                        mass_drift_frame_50 = float("nan")
                    dm_df = pd.DataFrame({
                        't': times_fc[:T_pred],
                        'density_variance_true': density_var_true,
                        'density_variance_pred': density_var_pred,
                        'mass_true': mass_true_fc,
                        'mass_pred': mass_pred,
                    })
                    dm_df.to_csv(run_dir / "density_metrics_wsindy.csv", index=False)
                except Exception:
                    motion_energy_true = float("nan")
                    motion_energy_pred = float("nan")
                    motion_ratio = float("nan")
                    mass_drift_mean = float("nan")
                    frame_50_rho_rmse = float("nan")
                    mass_drift_frame_50 = float("nan")

                test_results.append({
                    "test_id": ti, "run_name": run_name,
                    "r2_reconstructed": mean_r2,
                    "forecast_method": method,
                    "forecast_status": "ok",
                    "failure_reason": None,
                    "failure_step": None,
                    "forecast_method_attempted": fc_cfg.get("method", "auto") if wsindy_mf_result is not None else method,
                    "forecast_method_used": method_used if wsindy_mf_result is not None else method,
                    "motion_energy_true": motion_energy_true,
                    "motion_energy_pred": motion_energy_pred,
                    "motion_ratio": motion_ratio,
                    "mass_drift_mean": mass_drift_mean,
                    "frame_50_rho_rmse": frame_50_rho_rmse,
                    "mass_drift_frame_50": mass_drift_frame_50,
                    "selected_ell": selected_ell_json,
                    "selected_ell_t": selected_ell_vals[0],
                    "selected_ell_x": selected_ell_vals[1],
                    "selected_ell_y": selected_ell_vals[2],
                    "weak_r2_rho": weak_r2_rho,
                    "weak_r2_px": weak_r2_px,
                    "weak_r2_py": weak_r2_py,
                    "div_p_coeff": div_p_coeff,
                    "lap_rho_sign": lap_rho_sign,
                    "lap_rho_coeff": lap_rho_coeff,
                    "px_linear_coeff": px_linear_coeff,
                    "py_linear_coeff": py_linear_coeff,
                    "cond_G_rho": cond_G_rho,
                    "cond_G_px": cond_G_px,
                    "cond_G_py": cond_G_py,
                    "regime_class": regime_class_result,
                    "regime_class_source": regime_class_source,
                    "forces_enabled": forces_enabled_result,
                    "ca_cr_ratio": ca_cr_ratio_result,
                    "effective_morse": effective_morse_result,
                })
                all_r2.append(r2_ts[1:])

                ic = test_meta_list[ti].get("distribution", "?")
                print(f"    [{ti+1}/{n_test}] {run_name} ({ic}): R²={mean_r2:.4f} ({method})")

            # Save aggregate results
            results_df = pd.DataFrame(test_results)
            results_df.to_csv(WSINDY_DIR / "test_results.csv", index=False)

            mean_r2_wsindy = results_df["r2_reconstructed"].mean()
            std_r2_wsindy = results_df["r2_reconstructed"].std()
            n_success_wsindy = int((results_df.get("forecast_status") == "ok").sum()) if "forecast_status" in results_df.columns else int(results_df["r2_reconstructed"].notna().sum())
            n_failed_wsindy = int(len(results_df) - n_success_wsindy)
            print(f"\n  WSINDy mean R²: {mean_r2_wsindy:.4f} +/- {std_r2_wsindy:.4f}")

            wsindy_summary["test_evaluation"] = {
                "mean_r2": float(mean_r2_wsindy) if not np.isnan(mean_r2_wsindy) else None,
                "std_r2": float(std_r2_wsindy) if not np.isnan(std_r2_wsindy) else None,
                "n_test": n_test,
                "n_success": n_success_wsindy,
                "n_failed": n_failed_wsindy,
            }

            # Save WSINDy runtime profile (analogous to MVAR/LSTM)
            wsindy_discovery_time = wsindy_summary.get("model_selection", {}).get("time_s", 0)
            wsindy_runtime_profile = {
                "model_name": "WSINDy",
                "training_time_seconds": wsindy_discovery_time,
                "model_params": int(wsindy_summary.get("discovered_pde", {}).get("n_active", 0)
                                    if isinstance(wsindy_summary.get("discovered_pde"), dict)
                                    else sum(v.get("n_active", 0)
                                             for v in wsindy_summary.get("discovered_pde", {}).values()
                                             if isinstance(v, dict))),
                "inference": {
                    "single_step": {
                        "mean_seconds": 0.0,  # PDE integrator, not profiled per-step
                    },
                },
                "notes": "WSINDy is a PDE discovery method; training = model selection + bootstrap",
            }
            with open(WSINDY_DIR / "runtime_profile.json", "w") as f:
                json.dump(wsindy_runtime_profile, f, indent=2)

            if mean_r2_mvar is not None:
                wsindy_summary["comparison_with_mvar"] = {
                    "mvar_mean_r2": float(mean_r2_mvar),
                    "wsindy_mean_r2": float(mean_r2_wsindy) if not np.isnan(mean_r2_wsindy) else None,
                    "difference": float(mean_r2_wsindy - mean_r2_mvar) if not np.isnan(mean_r2_wsindy) else None,
                }

        else:
            # ============================================================
            #  IDENTIFICATION-ONLY PATH
            #  No forecast. Evaluate identification quality per test traj.
            # ============================================================
            print("  Mode: identification_only (no forecast)")
            identification_rows = []

            # Helper: compute px/py coefficient asymmetry
            def _px_py_asymmetry(mf_result):
                if mf_result is None:
                    return float("nan")
                px_mdl = mf_result.px_model
                py_mdl = mf_result.py_model
                # Map px term names to py equivalents
                _xy_map = {
                    "px": "py", "dx_rho": "dy_rho", "lap_px": "lap_py",
                    "rho_dx_Phi": "rho_dy_Phi", "div_px_p": "div_py_p",
                    "dx_rho2": "dy_rho2", "p_sq_px": "p_sq_py",
                    "bilap_px": "bilap_py", "p_dot_grad_px": "p_dot_grad_py",
                }
                max_diff = 0.0
                for px_name, py_name in _xy_map.items():
                    px_c = _coeff_for_model(px_mdl, px_name)
                    py_c = _coeff_for_model(py_mdl, py_name)
                    if px_c is not None and py_c is not None:
                        max_diff = max(max_diff, abs(px_c - py_c))
                    elif px_c is not None or py_c is not None:
                        max_diff = max(max_diff, abs(px_c or 0.0) + abs(py_c or 0.0))
                return float(max_diff)

            # Retrieve dominant balance from post-regression diagnostics
            post_diag = wsindy_summary.get("post_regression_diagnostics", {})

            for ti in range(n_test):
                run_name = test_meta_list[ti]["run_name"]
                ic = test_meta_list[ti].get("distribution", "?")

                # Per-equation dominant balance dicts
                rho_db = {}
                px_db = {}
                py_db = {}
                for eq_name, db_target in [("rho", rho_db), ("px", px_db), ("py", py_db)]:
                    eq_diag = post_diag.get(eq_name, {})
                    db_norm = eq_diag.get("dominant_balance_normalized", {})
                    db_target.update(db_norm)

                row = {
                    "trajectory_id": run_name,
                    "regime_class": regime_class_result,
                    "effective_morse": effective_morse_result,
                    "selected_ell": selected_ell_json,
                    # rho_t
                    "rho_weak_r2": weak_r2_rho,
                    "rho_div_p_coeff": div_p_coeff,
                    "rho_active_terms": ",".join(
                        wsindy_mf_result.rho_model.active_terms
                    ) if wsindy_mf_result else "",
                    "rho_dominant_balance": json.dumps(rho_db),
                    "cond_G_rho": cond_G_rho,
                    # px_t
                    "px_weak_r2": weak_r2_px,
                    "px_active_terms": ",".join(
                        wsindy_mf_result.px_model.active_terms
                    ) if wsindy_mf_result else "",
                    "px_dominant_balance": json.dumps(px_db),
                    "cond_G_px": cond_G_px,
                    # py_t
                    "py_weak_r2": weak_r2_py,
                    "py_active_terms": ",".join(
                        wsindy_mf_result.py_model.active_terms
                    ) if wsindy_mf_result else "",
                    "py_dominant_balance": json.dumps(py_db),
                    "cond_G_py": cond_G_py,
                    # cross-equation
                    "px_py_coeff_asymmetry": _px_py_asymmetry(wsindy_mf_result),
                }
                identification_rows.append(row)
                print(f"    [{ti+1}/{n_test}] {run_name} ({ic}): identification_only")

            # ── Write identification_results.csv ────────────────────
            id_df = pd.DataFrame(identification_rows)
            id_df.to_csv(WSINDY_DIR / "identification_results.csv", index=False)
            print(f"\n  Saved: identification_results.csv ({len(id_df)} rows)")

            # ── Write identification_summary.json ───────────────────
            n_traj = len(identification_rows)
            id_summary = {
                "regime": args.experiment_name,
                "regime_class": regime_class_result,
                "effective_morse": effective_morse_result,
                "n_test_trajectories": n_traj,
                "selected_ell": selected_ell_vals,
                "equations": {},
            }
            for eq_name in ["rho_t", "px_t", "py_t"]:
                eq_key = eq_name.replace("_t", "")  # rho, px, py
                mdl = getattr(wsindy_mf_result, f"{eq_key}_model", None) if wsindy_mf_result else None
                eq_diag = post_diag.get(eq_key, {})

                if mdl is not None:
                    # Coefficient dict for all active terms
                    coeff_dict = {
                        n: float(mdl.w[mdl.col_names.index(n)])
                        for n in mdl.active_terms
                    }
                    # Stability selection: count appearances across test trajectories
                    # (In identification-only mode, the model is fitted once on training
                    #  data, so all test trajectories see the same active set. The stable/
                    #  fragile classification becomes meaningful when bootstrap is enabled.)
                    all_active = mdl.active_terms
                    stable_terms = list(all_active)  # 100% retention = stable
                    fragile_terms = []

                    eq_r2_key = f"weak_r2_{eq_key}"
                    eq_cond_key = f"cond_G_{eq_key}"
                    r2_vals = [r.get(eq_r2_key, float("nan")) for r in identification_rows]
                    cond_vals = [r.get(eq_cond_key, float("nan")) for r in identification_rows]

                    id_summary["equations"][eq_name] = {
                        "mean_weak_r2": float(np.nanmean(r2_vals)),
                        "mean_cond_G": float(np.nanmean(cond_vals)),
                        "stable_terms": stable_terms,
                        "fragile_terms": fragile_terms,
                        "mean_coefficients": coeff_dict,
                        "std_coefficients": {n: 0.0 for n in coeff_dict},
                        "mean_dominant_balance": eq_diag.get("dominant_balance_normalized", {}),
                    }

            with open(WSINDY_DIR / "identification_summary.json", "w") as f:
                json.dump(id_summary, f, indent=2, default=str)
            print(f"  Saved: identification_summary.json")

            # ── Summary entries ─────────────────────────────────────
            wsindy_summary["test_evaluation"] = {
                "mode": "identification_only",
                "n_test": n_test,
            }

            # Save WSINDy runtime profile
            wsindy_discovery_time = wsindy_summary.get("model_selection", {}).get("time_s", 0)
            wsindy_runtime_profile = {
                "model_name": "WSINDy",
                "mode": "identification_only",
                "training_time_seconds": wsindy_discovery_time,
                "model_params": int(sum(
                    v.get("n_active", 0)
                    for v in wsindy_summary.get("discovered_pde", {}).values()
                    if isinstance(v, dict)
                )),
                "notes": "WSINDy identification-only mode; no forecast evaluation",
            }
            with open(WSINDY_DIR / "runtime_profile.json", "w") as f:
                json.dump(wsindy_runtime_profile, f, indent=2)

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    total_time = time.time() - start_time

    print(f"\n{sep}")
    print("  PIPELINE COMPLETE")
    print(f"{sep}")
    print(f"  Total time: {total_time/60:.1f}m")

    if mvar_enabled and mean_r2_mvar is not None:
        print(f"  MVAR  mean R²: {mean_r2_mvar:.4f}")
    if lstm_enabled and mean_r2_lstm is not None:
        print(f"  LSTM  mean R²: {mean_r2_lstm:.4f}")
    if wsindy_enabled and wsindy_has_model:
        wsindy_mode_used = wsindy_summary.get("mode", "scalar")
        if wsindy_mode_used == "multifield" and wsindy_mf_result is not None:
            print(f"  WSINDy mode: multifield (3-equation system)")
            print(wsindy_mf_result.summary())
        elif wsindy_model is not None:
            print(f"  WSINDy PDE:    {to_text(wsindy_model)}")
        if mean_r2_wsindy is not None:
            print(f"  WSINDy mean R²: {mean_r2_wsindy:.4f}")

    # Save master summary
    summary = {
        "experiment_name": args.experiment_name,
        "config": args.config,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_train": n_train,
        "n_test": n_test,
        "r_pod": int(R_POD) if R_POD is not None else None,
        "total_time_minutes": total_time / 60,
        "models_enabled": {
            "mvar": mvar_enabled,
            "lstm": lstm_enabled,
            "wsindy": wsindy_enabled,
            "wsindy_only_cli": bool(args.wsindy_only),
        },
        "evaluation": {
            "forecast_start_requested_s": float(
                eval_config.get("forecast_start_requested", eval_config.get("forecast_start", BASE_CONFIG["sim"]["T"]))
            ),
            "forecast_start_effective_s": float(
                eval_config.get("forecast_start_effective", eval_config.get("forecast_start", BASE_CONFIG["sim"]["T"]))
            ),
            "forecast_start_conditioning_steps": int(
                eval_config.get("forecast_start_conditioning_steps", 0)
            ),
            "forecast_start_required_lag": int(
                eval_config.get("forecast_start_required_lag", 0)
            ),
        },
    }
    if mvar_enabled and mean_r2_mvar is not None:
        summary["mvar"] = {
            "mean_r2_test": float(mean_r2_mvar) if not pd.isna(mean_r2_mvar) else None,
            "training_time_s": mvar_training_time,
            "lag_used": int(mvar_lag) if mvar_lag is not None else None,
            "lag_raw_config": models_cfg.get("mvar", {}).get("lag", 5),
        }
    if lstm_enabled and mean_r2_lstm is not None:
        summary["lstm"] = {
            "mean_r2_test": float(mean_r2_lstm) if not pd.isna(mean_r2_lstm) else None,
            "training_time_s": lstm_training_time,
            "lag_used": int(lstm_effective_lag) if lstm_effective_lag is not None else None,
            "lag_raw_config": models_cfg.get("lstm", {}).get("lag", 20),
        }
    # Record all 4 IC criteria if alvarez results are available
    _alvarez_dirs = [
        Path("rom_hyperparameters/results") / args.experiment_name,
        Path("rom_hyperparameters/results_oscar_unaligned") / args.experiment_name,
    ]
    for _adir in _alvarez_dirs:
        _alvarez_path = _adir / "summary.json"
        if _alvarez_path.exists():
            with open(_alvarez_path) as f:
                _alvarez = json.load(f)
            summary["lag_criteria"] = {
                "lag_bic": _alvarez.get("lag_bic"),
                "lag_aic": _alvarez.get("lag_aic"),
                "lag_hqic": _alvarez.get("lag_hqic"),
                "lag_fpe": _alvarez.get("lag_fpe"),
                "source": str(_alvarez_path),
            }
            break
    if wsindy_summary:
        summary["wsindy"] = wsindy_summary
    if stationarity_stats and "error" not in stationarity_stats:
        summary["stationarity"] = stationarity_stats

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Output: {OUTPUT_DIR}/")
    print(f"  Summary: {OUTPUT_DIR}/summary.json")

    # Light-weight export list (for download script)
    export_files = [
        "summary.json", "config_used.yaml",
    ]
    if rom_enabled:
        export_files += [
            "rom_common/pod_basis.npz",
            "rom_common/X_train_mean.npy",
            "rom_common/shift_align.npz",
        ]
        export_files.append("rom_common/latent_dataset.npz")
    if mvar_enabled:
        export_files += ["MVAR/mvar_model.npz", "MVAR/test_results.csv",
                         "MVAR/runtime_profile.json"]
    if lstm_enabled:
        export_files += ["LSTM/lstm_state_dict.pt", "LSTM/test_results.csv",
                         "LSTM/training_log.csv", "LSTM/runtime_profile.json"]
    if wsindy_enabled:
        wsindy_mode_used = wsindy_summary.get("mode", "scalar")
        if wsindy_mode_used == "multifield":
            export_files += [
                "WSINDy/multifield_model.json",
                "WSINDy/multifield_diagnostics.json",
                "WSINDy/wsindy_model_rho.npz",
                "WSINDy/wsindy_model_px.npz",
                "WSINDy/wsindy_model_py.npz",
                "WSINDy/bootstrap_rho.npz",
                "WSINDy/bootstrap_px.npz",
                "WSINDy/bootstrap_py.npz",
                "WSINDy/test_results.csv",
            ]
        else:
            export_files += [
                "WSINDy/wsindy_model.npz", "WSINDy/test_results.csv",
                "WSINDy/bootstrap.npz",
            ]
        export_files.append("WSINDy/runtime_profile.json")
    # Test artifacts
    export_files += ["test/metadata.json"]
    for ti in range(n_test):
        prefix = f"test/test_{ti:03d}/"
        export_files += [
            prefix + "density_true.npz",
            prefix + "density_pred.npz",
            prefix + "r2_vs_time.csv",
            prefix + "metrics_summary.json",
            prefix + "order_params.csv",
        ]
        if mvar_enabled:
            export_files += [
                prefix + "density_pred_mvar.npz",
                prefix + "r2_vs_time_mvar.csv",
                prefix + "density_metrics_mvar.csv",
            ]
        if lstm_enabled:
            export_files += [
                prefix + "density_pred_lstm.npz",
                prefix + "r2_vs_time_lstm.csv",
                prefix + "density_metrics_lstm.csv",
            ]
        if wsindy_enabled:
            export_files += [
                prefix + "density_pred_wsindy.npz",
                prefix + "r2_vs_time_wsindy.csv",
                prefix + "density_metrics_wsindy.csv",
            ]

    with open(OUTPUT_DIR / "export_manifest.json", "w") as f:
        json.dump({"files": export_files}, f, indent=2)

    print(f"  Export manifest: {OUTPUT_DIR}/export_manifest.json")
    print(f"    ({len(export_files)} files to download)")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
