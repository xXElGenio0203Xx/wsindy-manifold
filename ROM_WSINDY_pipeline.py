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
from wsindy.system import build_weak_system, default_t_margin, make_query_indices
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
    discover_multifield,
    model_selection_multifield,
    forecast_multifield,
    bootstrap_multifield,
    MultiFieldResult,
)


# ═══════════════════════════════════════════════════════════════════
#  WSINDy helper functions
# ═══════════════════════════════════════════════════════════════════

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


def build_stacked_weak_system(
    train_densities, grid, psi_bundle, library_terms, stride=(2, 2, 2),
):
    """Build stacked weak system (b, G) from multiple trajectories."""
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
            U_k, grid, psi_bundle, library_terms, qi,
        )
        all_b.append(b_k)
        all_G.append(G_k)

    if not all_b:
        raise ValueError("No valid query points from any training trajectory")

    return np.concatenate(all_b), np.vstack(all_G), col_names


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
    b, G, col_names = build_stacked_weak_system(
        train_densities, grid, psi_bundle, library_terms, stride,
    )
    model = wsindy_fit_regression(b, G, col_names, lambdas=lambdas, max_iter=max_iter)
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


# ═══════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ROM + WSINDy Pipeline (for Oscar cluster)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
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

    # Setup directories
    OUTPUT_DIR = Path(f"oscar_output/{args.experiment_name}")
    ROM_COMMON_DIR = OUTPUT_DIR / "rom_common"
    MVAR_DIR = OUTPUT_DIR / "MVAR"
    LSTM_DIR = OUTPUT_DIR / "LSTM"
    WSINDY_DIR = OUTPUT_DIR / "WSINDy"

    for d in [OUTPUT_DIR, ROM_COMMON_DIR, MVAR_DIR, LSTM_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    if wsindy_enabled:
        WSINDY_DIR.mkdir(parents=True, exist_ok=True)
        (WSINDY_DIR / "plots").mkdir(exist_ok=True)

    shutil.copy(args.config, OUTPUT_DIR / "config_used.yaml")

    runtime_analyzer = RuntimeAnalyzer()
    runtime_profiles = []

    models_cfg = rom_config.get("models", {})
    mvar_enabled = models_cfg.get("mvar", {}).get("enabled", True)
    lstm_enabled = models_cfg.get("lstm", {}).get("enabled", False)

    print(f"\n  Models:")
    print(f"    MVAR:   {'ON' if mvar_enabled else 'OFF'}")
    print(f"    LSTM:   {'ON' if lstm_enabled else 'OFF'}")
    print(f"    WSINDy: {'ON' if wsindy_enabled else 'OFF'}")

    # ================================================================
    # STEP 1: Generate Training Data
    # ================================================================
    print(f"\n{sep}")
    print("  STEP 1: Generating Training Data")
    print(f"{sep}")

    train_configs = generate_training_configs(train_ic_config, BASE_CONFIG)
    n_train = len(train_configs)
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

    # ================================================================
    # STEP 2: Build POD Basis (shared)
    # ================================================================
    print(f"\n{sep}")
    print("  STEP 2: Building Shared POD Basis")
    print(f"{sep}")

    TRAIN_DIR = OUTPUT_DIR / "train"
    pod_data = build_pod_basis(TRAIN_DIR, n_train, rom_config)
    save_pod_basis(pod_data, ROM_COMMON_DIR)

    R_POD = pod_data["R_POD"]
    T_rom = pod_data["T_rom"]
    M = pod_data["M"]
    print(f"  Latent dim: {R_POD}, T_rom: {T_rom}")

    # ================================================================
    # STEP 3: Build Shared Latent Dataset
    # ================================================================
    print(f"\n{sep}")
    print("  STEP 3: Building Latent Dataset")
    print(f"{sep}")

    X_latent = pod_data["X_latent"]

    latent_standardize = rom_config.get("latent_standardize", False)
    if latent_standardize:
        latent_mean = X_latent.mean(axis=0)
        latent_std = X_latent.std(axis=0)
        latent_std[latent_std < 1e-12] = 1.0
        X_latent = (X_latent - latent_mean) / latent_std
        pod_data["X_latent"] = X_latent
        pod_data["latent_mean"] = latent_mean
        pod_data["latent_std"] = latent_std
    else:
        pod_data["latent_mean"] = None
        pod_data["latent_std"] = None

    y_trajs = []
    for m_idx in range(M):
        y_trajs.append(X_latent[m_idx * T_rom : (m_idx + 1) * T_rom, :])

    if mvar_enabled:
        lag = models_cfg["mvar"].get("lag", 5)
    else:
        lag = models_cfg.get("lstm", {}).get("lag", 20)

    X_all, Y_all = build_latent_dataset(y_trajs, lag=lag)
    print(f"  X_all: {X_all.shape}, Y_all: {Y_all.shape}, lag={lag}")

    Y_multi = None
    X_lstm = X_all
    Y_lstm = Y_all
    if lstm_enabled:
        lstm_ms_cfg = models_cfg.get("lstm", {})
        ms_enabled = lstm_ms_cfg.get("multistep_loss", False)
        ms_k = lstm_ms_cfg.get("multistep_k", 5)
        if ms_enabled and ms_k > 1:
            X_lstm, Y_multi = build_multistep_latent_dataset(y_trajs, lag=lag, k_steps=ms_k)
            Y_lstm = Y_multi[:, 0, :]

    np.savez(ROM_COMMON_DIR / "latent_dataset.npz", X_all=X_all, Y_all=Y_all, lag=lag)

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
        lstm_data = {
            "model_path": lstm_model_path,
            "val_loss": lstm_val_loss,
            "lag": lstm_config.get("lag", lag),
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

        TEST_DIR = OUTPUT_DIR / "test"
        ROM_SUBSAMPLE = rom_config.get("subsample", 1)

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
                    residual=lstm_cfg_fb.get("residual", True),
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
        p = tuple(ms_cfg.get("p", [2, 2, 2]))
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
            morse_enabled = mf_cfg.get("morse", True)
            rich = mf_cfg.get("rich", False)
            kde_bw = mf_cfg.get("kde_bandwidth", 5.0)

            # Morse parameters from sim config
            sim_cfg = config.get("sim", {})
            Ca = sim_cfg.get("Ca", 0.8)
            Cr = sim_cfg.get("Cr", 0.3)
            la = sim_cfg.get("la", 1.5)
            lr = sim_cfg.get("lr", 0.5)

            # ── Build 3-equation library ────────────────────────────
            mf_library = build_mf_library(morse=morse_enabled, rich=rich)
            n_lib_total = sum(len(v) for v in mf_library.values())
            print(f"\n  Multi-field library: {n_lib_total} terms ({len(mf_library)} equations)")
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
                        morse_params=dict(Cr=Cr, Ca=Ca, lr=lr, la=la) if morse_enabled else None,
                    )
                else:
                    print(f"    WARNING: {meta['run_name']} has no trajectory.npz, using rho-only")
                    fd = build_field_data_rho_only(
                        rho_i, xgrid, ygrid, Lx, Ly, dt,
                        morse_params=dict(Cr=Cr, Ca=Ca, lr=lr, la=la) if morse_enabled else None,
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
            )
            ms_time = time.perf_counter() - t_ms

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
                    "morse": morse_enabled,
                    "rich": rich,
                    "per_equation": {
                        eq: [t.name for t in terms]
                        for eq, terms in mf_library.items()
                    },
                },
                "n_train_trajectories": len(train_densities),
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
        mvar_lag_val = rom_config.get("models", {}).get("mvar", {}).get("lag", 5)
        fc_cfg = wsindy_config.get("forecast", {})
        clip_neg = fc_cfg.get("clip_negative", True)

        test_results = []
        all_r2 = []

        for ti in range(n_test):
            run_name = test_meta_list[ti]["run_name"]
            run_dir = TEST_DIR / run_name

            d = np.load(run_dir / "density_true.npz")
            rho_true = d["rho"][::w_subsample]
            times_sub = d["times"][::w_subsample]
            T_test_sub = rho_true.shape[0]

            forecast_start = mvar_lag_val
            n_fc = T_test_sub - forecast_start - 1
            if n_fc <= 0:
                continue

            try:
                # ==================================================
                #  MULTIFIELD forecast
                # ==================================================
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
                        px0, py0 = compute_flux_kde(
                            traj_test[forecast_start],
                            vel_test[forecast_start],
                            xgrid, ygrid, bandwidth=kde_bw,
                        )
                    else:
                        # No trajectory → zero flux ICs (degraded mode)
                        print(f"    WARNING: {run_name} has no trajectory.npz, using zero-flux ICs")
                        px0 = np.zeros_like(rho0)
                        py0 = np.zeros_like(rho0)

                    # Morse params
                    sim_cfg = config.get("sim", {})
                    morse_params = None
                    mf_cfg_lib = wsindy_config.get("multifield_library", {})
                    if mf_cfg_lib.get("morse", True):
                        morse_params = dict(
                            Cr=sim_cfg.get("Cr", 0.3),
                            Ca=sim_cfg.get("Ca", 0.8),
                            lr=sim_cfg.get("lr", 0.5),
                            la=sim_cfg.get("la", 1.5),
                        )

                    fc_method = fc_cfg.get("method", "auto")
                    mass_conserve = fc_cfg.get("mass_conserve", True)

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
                    # Stack with IC → same shape as scalar path
                    U_pred = rho_pred  # already includes IC at index 0
                    method = f"{fc_method}_multifield"

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
                forecast_start_idx=int(forecast_start * w_subsample),
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
            }).to_csv(run_dir / "r2_vs_time_wsindy.csv", index=False)

            test_results.append({
                "test_id": ti, "run_name": run_name,
                "r2_reconstructed": mean_r2,
                "forecast_method": method,
            })
            all_r2.append(r2_ts[1:])

            ic = test_meta_list[ti].get("distribution", "?")
            print(f"    [{ti+1}/{n_test}] {run_name} ({ic}): R²={mean_r2:.4f} ({method})")

        # Save aggregate results
        results_df = pd.DataFrame(test_results)
        results_df.to_csv(WSINDY_DIR / "test_results.csv", index=False)

        mean_r2_wsindy = results_df["r2_reconstructed"].mean()
        std_r2_wsindy = results_df["r2_reconstructed"].std()
        print(f"\n  WSINDy mean R²: {mean_r2_wsindy:.4f} +/- {std_r2_wsindy:.4f}")

        wsindy_summary["test_evaluation"] = {
            "mean_r2": float(mean_r2_wsindy) if not np.isnan(mean_r2_wsindy) else None,
            "std_r2": float(std_r2_wsindy) if not np.isnan(std_r2_wsindy) else None,
            "n_test": n_test,
        }

        if mean_r2_mvar is not None:
            wsindy_summary["comparison_with_mvar"] = {
                "mvar_mean_r2": float(mean_r2_mvar),
                "wsindy_mean_r2": float(mean_r2_wsindy) if not np.isnan(mean_r2_wsindy) else None,
                "difference": float(mean_r2_wsindy - mean_r2_mvar) if not np.isnan(mean_r2_wsindy) else None,
            }

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
        "r_pod": int(R_POD),
        "total_time_minutes": total_time / 60,
        "models_enabled": {
            "mvar": mvar_enabled,
            "lstm": lstm_enabled,
            "wsindy": wsindy_enabled,
        },
    }
    if mvar_enabled and mean_r2_mvar is not None:
        summary["mvar"] = {
            "mean_r2_test": float(mean_r2_mvar) if not pd.isna(mean_r2_mvar) else None,
            "training_time_s": mvar_training_time,
        }
    if lstm_enabled and mean_r2_lstm is not None:
        summary["lstm"] = {
            "mean_r2_test": float(mean_r2_lstm) if not pd.isna(mean_r2_lstm) else None,
            "training_time_s": lstm_training_time,
        }
    if wsindy_summary:
        summary["wsindy"] = wsindy_summary

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Output: {OUTPUT_DIR}/")
    print(f"  Summary: {OUTPUT_DIR}/summary.json")

    # Light-weight export list (for download script)
    export_files = [
        "summary.json", "config_used.yaml",
        "rom_common/pod_basis.npz", "rom_common/latent_dataset.npz",
        "rom_common/X_train_mean.npy", "rom_common/shift_align.npz",
    ]
    if mvar_enabled:
        export_files += ["MVAR/mvar_model.npz", "MVAR/test_results.csv",
                         "MVAR/runtime_profile.json"]
    if lstm_enabled:
        export_files += ["LSTM/lstm_state_dict.pt", "LSTM/test_results.csv",
                         "LSTM/training_log.csv"]
    if wsindy_enabled:
        wsindy_mode_used = wsindy_summary.get("mode", "scalar")
        if wsindy_mode_used == "multifield":
            export_files += [
                "WSINDy/multifield_model.json",
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
    # Test artifacts
    export_files += ["test/metadata.json"]
    for ti in range(n_test):
        prefix = f"test/test_{ti:03d}/"
        export_files += [
            prefix + "density_true.npz",
            prefix + "density_pred.npz",
            prefix + "density_pred_mvar.npz",
            prefix + "r2_vs_time.csv",
            prefix + "metrics_summary.json",
        ]
        if wsindy_enabled:
            export_files += [
                prefix + "density_pred_wsindy.npz",
                prefix + "r2_vs_time_wsindy.csv",
            ]

    with open(OUTPUT_DIR / "export_manifest.json", "w") as f:
        json.dump({"files": export_files}, f, indent=2)

    print(f"  Export manifest: {OUTPUT_DIR}/export_manifest.json")
    print(f"    ({len(export_files)} files to download)")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
