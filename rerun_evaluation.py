#!/usr/bin/env python3
"""
Re-run ONLY the evaluation step (Step 6) for completed experiments.

Uses existing artifacts on disk:
  - config_used.yaml
  - rom_common/pod_basis.npz  +  rom_common/X_train_mean.npy
  - rom_common/shift_align.npz  (if shift_align was enabled)
  - MVAR/mvar_model.npz
  - LSTM/lstm_state_dict.pt
  - test/test_*/density_true.npz

This avoids rerunning simulation (Step 1), POD (Step 2), and training (Steps 4-5).

Usage:
    # Re-evaluate a single experiment
    python rerun_evaluation.py oscar_output/DO_CS01

    # Re-evaluate all completed experiments under a parent dir
    python rerun_evaluation.py oscar_output/ --all

    # Re-evaluate only LSTM (skip MVAR)
    python rerun_evaluation.py oscar_output/DO_CS01 --lstm-only
"""

import argparse
import sys
import yaml
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge

# ── project imports ──
from rectsim.test_evaluator import evaluate_test_runs
from rectsim.forecast_utils import mvar_forecast_fn_factory
from rom.lstm_rom import LatentLSTMROM, lstm_forecast_fn_factory, load_lstm_model


def load_pod_data(output_dir: Path, rom_config: dict) -> dict:
    """Reconstruct the pod_data dict from saved artifacts."""
    rom_common = output_dir / "rom_common"

    pod_npz = np.load(rom_common / "pod_basis.npz")
    X_mean = np.load(rom_common / "X_train_mean.npy")
    U_r = pod_npz["U"]
    R_POD = U_r.shape[1]

    pod_data = {
        "U_r": U_r,
        "X_mean": X_mean,
        "R_POD": R_POD,
        "S": pod_npz.get("all_singular_values", pod_npz["singular_values"]),
        "energy_captured": float(pod_npz.get("energy_ratio", 0.0)),
        "cumulative_energy": pod_npz.get("cumulative_ratio", None),
        "total_energy": float(pod_npz.get("total_energy", 0.0)),
        # Density transform (must match training)
        "density_transform": rom_config.get("density_transform", "raw"),
        "density_transform_eps": rom_config.get("density_transform_eps", 1e-8),
    }

    # ── Shift alignment ──
    sa_path = rom_common / "shift_align.npz"
    if sa_path.exists() and rom_config.get("shift_align", False):
        sa = np.load(sa_path, allow_pickle=True)
        pod_data["shift_align"] = True
        pod_data["shift_align_data"] = {
            "ref": sa["ref"],
            "shifts": sa["shifts"],
            "ref_method": str(sa["ref_method"]),
            "density_shape_2d": tuple(sa["density_shape_2d"]),
        }
    else:
        pod_data["shift_align"] = False
        pod_data["shift_align_data"] = None

    # ── Latent standardization ──
    # If latent_standardize was used during training, we need latent_mean/std.
    # Recompute from the saved latent dataset if available.
    latent_standardize = rom_config.get("latent_standardize", False)
    if latent_standardize:
        ds_path = rom_common / "latent_dataset.npz"
        if ds_path.exists():
            ds = np.load(ds_path)
            lag = int(ds["lag"])
            d = R_POD
            # X_all is [n_samples, lag*d].  Recover per-mode stats from Y_all [n_samples, d]
            # which are the 1-step targets (already standardized during training).
            # Safer: re-project training densities, but that's expensive.
            # Approximate: the standardized latent data has mean≈0, std≈1 — so
            # original_mean ≈ 0 and original_std ≈ 1 after POD centering.
            # This is a limitation; if latent_standardize was ON, a full re-run
            # would be more accurate.
            print("  ⚠️  latent_standardize=True but stats not saved; "
                  "setting latent_mean=0, latent_std=1 (approximate)")
            pod_data["latent_mean"] = np.zeros(d)
            pod_data["latent_std"] = np.ones(d)
        else:
            pod_data["latent_mean"] = None
            pod_data["latent_std"] = None
    else:
        pod_data["latent_mean"] = None
        pod_data["latent_std"] = None

    return pod_data


def load_mvar(mvar_dir: Path):
    """Load MVAR model and reconstruct sklearn Ridge wrapper."""
    mvar_npz = np.load(mvar_dir / "mvar_model.npz")

    model = Ridge(alpha=float(mvar_npz["alpha"]))
    model.coef_ = mvar_npz["A_companion"]
    model.intercept_ = 0.0
    model.n_features_in_ = mvar_npz["A_companion"].shape[1]

    mvar_lag = int(mvar_npz["p"])
    print(f"  MVAR: p={mvar_lag}, r={int(mvar_npz['r'])}, "
          f"alpha={float(mvar_npz['alpha']):.1e}, "
          f"train_r2={float(mvar_npz['train_r2']):.6f}")
    return model, mvar_lag


def load_lstm(lstm_dir: Path, device: str = "cpu"):
    """Load LSTM model via the standard loader."""
    import torch
    model, input_mean, input_std = load_lstm_model(str(lstm_dir), device=device)
    print(f"  LSTM loaded from {lstm_dir}")
    return model, input_mean, input_std


def evaluate_experiment(output_dir: Path, *, mvar_only=False, lstm_only=False):
    """Re-run Step 6 (evaluation) for a single experiment directory."""
    output_dir = Path(output_dir)
    print(f"\n{'=' * 72}")
    print(f"  Re-evaluating: {output_dir.name}")
    print(f"{'=' * 72}")

    # ── Load config ──
    cfg_path = output_dir / "config_used.yaml"
    if not cfg_path.exists():
        print(f"  ✗ config_used.yaml not found — skipping")
        return False
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    rom_config = config.get("rom", config.get("ROM", {}))
    eval_config = config.get("eval", {"save_time_resolved": True})
    # Force time-resolved saving so we get r2_vs_time CSVs
    eval_config["save_time_resolved"] = True

    DENSITY_NX = config["density"]["nx"]
    DENSITY_NY = config["density"]["ny"]
    ROM_SUBSAMPLE = rom_config.get("subsample", 1)

    # ── Determine which models are enabled ──
    models_cfg = rom_config.get("models", {})
    mvar_enabled = "mvar" in models_cfg and not lstm_only
    lstm_enabled = "lstm" in models_cfg and not mvar_only

    if not mvar_enabled and not lstm_enabled:
        print("  ✗ No models to evaluate — skipping")
        return False

    # ── Load POD data ──
    print("\n  Loading POD basis …")
    pod_data = load_pod_data(output_dir, rom_config)
    print(f"    R_POD = {pod_data['R_POD']}, "
          f"transform = {pod_data['density_transform']}, "
          f"shift_align = {pod_data['shift_align']}")

    # ── Load models ──
    MVAR_DIR = output_dir / "MVAR"
    LSTM_DIR = output_dir / "LSTM"

    mvar_model, mvar_lag = None, 0
    if mvar_enabled:
        if (MVAR_DIR / "mvar_model.npz").exists():
            print("\n  Loading MVAR model …")
            mvar_model, mvar_lag = load_mvar(MVAR_DIR)
        else:
            print("  ⚠️  MVAR model not found — skipping MVAR eval")
            mvar_enabled = False

    lstm_model, lstm_input_mean, lstm_input_std, lstm_lag = None, None, None, 0
    if lstm_enabled:
        if (LSTM_DIR / "lstm_state_dict.pt").exists():
            print("\n  Loading LSTM model …")
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            lstm_model, lstm_input_mean, lstm_input_std = load_lstm(LSTM_DIR, device)
            # The lag used during training — use the MVAR lag that was used to
            # build the shared latent dataset (the bug fix).
            lstm_lag = models_cfg.get("mvar", {}).get("lag", 5)
            print(f"    Using training lag = {lstm_lag} (from shared latent dataset)")
        else:
            print("  ⚠️  LSTM model not found — skipping LSTM eval")
            lstm_enabled = False

    # ── Count test runs ──
    TEST_DIR = output_dir / "test"
    if not TEST_DIR.exists():
        print("  ✗ test/ directory not found — skipping")
        return False
    test_dirs = sorted([d for d in TEST_DIR.iterdir()
                        if d.is_dir() and d.name.startswith("test_")])
    n_test = len(test_dirs)
    if n_test == 0:
        print("  ✗ No test runs found — skipping")
        return False
    print(f"\n  Found {n_test} test runs")

    # ── Test config ──
    BASE_CONFIG_TEST = config.copy()
    BASE_CONFIG_TEST["sim"] = config["sim"].copy()
    if "test_sim" in config:
        BASE_CONFIG_TEST["sim"].update(config["test_sim"])

    train_T = config["sim"]["T"]

    # ── Align forecast start across models (same logic as pipeline) ──
    _dt = BASE_CONFIG_TEST["sim"]["dt"]
    _raw_forecast_start = eval_config.get("forecast_start", train_T)
    _raw_T_train = int(_raw_forecast_start / _dt / ROM_SUBSAMPLE)

    _max_lag = 0
    if mvar_enabled:
        _max_lag = max(_max_lag, mvar_lag)
    if lstm_enabled:
        _max_lag = max(_max_lag, lstm_lag)

    if _raw_T_train < _max_lag:
        _aligned_T_train = _max_lag
        _aligned_forecast_start = _aligned_T_train * _dt * ROM_SUBSAMPLE
        print(f"\n  ℹ️  forecast_start={_raw_forecast_start}s → {_aligned_forecast_start}s "
              f"(need {_max_lag} conditioning steps)")
        eval_config = dict(eval_config)
        eval_config["forecast_start"] = _aligned_forecast_start

    # ── 6a: MVAR evaluation ──
    if mvar_enabled:
        print(f"\n  {'─' * 50}")
        print(f"  Re-evaluating MVAR ({n_test} test runs) …")
        mvar_forecast_fn = mvar_forecast_fn_factory(mvar_model, mvar_lag)
        mvar_results = evaluate_test_runs(
            test_dir=TEST_DIR, n_test=n_test,
            base_config_test=BASE_CONFIG_TEST,
            pod_data=pod_data, forecast_fn=mvar_forecast_fn,
            lag=mvar_lag,
            density_nx=DENSITY_NX, density_ny=DENSITY_NY,
            rom_subsample=ROM_SUBSAMPLE,
            eval_config=eval_config,
            train_T=train_T,
            model_name="MVAR",
        )
        out_csv = MVAR_DIR / "test_results.csv"
        mvar_results.to_csv(out_csv, index=False)
        mean_r2 = mvar_results["r2_reconstructed"].mean()
        print(f"  ✓ MVAR mean R² = {mean_r2:.4f}  →  {out_csv}")

    # ── 6b: LSTM evaluation ──
    if lstm_enabled:
        print(f"\n  {'─' * 50}")
        print(f"  Re-evaluating LSTM ({n_test} test runs) …")
        lstm_forecast_fn = lstm_forecast_fn_factory(
            lstm_model, lstm_input_mean, lstm_input_std)
        lstm_results = evaluate_test_runs(
            test_dir=TEST_DIR, n_test=n_test,
            base_config_test=BASE_CONFIG_TEST,
            pod_data=pod_data, forecast_fn=lstm_forecast_fn,
            lag=lstm_lag,
            density_nx=DENSITY_NX, density_ny=DENSITY_NY,
            rom_subsample=ROM_SUBSAMPLE,
            eval_config=eval_config,
            train_T=train_T,
            model_name="LSTM",
        )
        out_csv = LSTM_DIR / "test_results.csv"
        lstm_results.to_csv(out_csv, index=False)
        mean_r2 = lstm_results["r2_reconstructed"].mean()
        print(f"  ✓ LSTM mean R² = {mean_r2:.4f}  →  {out_csv}")

    print(f"\n  Done: {output_dir.name}")
    return True


def find_completed_experiments(parent_dir: Path) -> list:
    """Find all experiment directories that have completed (have test results)."""
    candidates = []
    for d in sorted(parent_dir.iterdir()):
        if not d.is_dir():
            continue
        # Must have config_used.yaml and test/ directory
        if (d / "config_used.yaml").exists() and (d / "test").is_dir():
            # Must have at least one model directory with a trained model
            has_mvar = (d / "MVAR" / "mvar_model.npz").exists()
            has_lstm = (d / "LSTM" / "lstm_state_dict.pt").exists()
            if has_mvar or has_lstm:
                candidates.append(d)
    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="Re-run evaluation (Step 6) using existing trained models.")
    parser.add_argument("experiment_dir", type=str,
                        help="Path to experiment output dir, or parent dir with --all")
    parser.add_argument("--all", action="store_true",
                        help="Re-evaluate all completed experiments under the given dir")
    parser.add_argument("--mvar-only", action="store_true",
                        help="Only re-evaluate MVAR")
    parser.add_argument("--lstm-only", action="store_true",
                        help="Only re-evaluate LSTM")
    args = parser.parse_args()

    exp_path = Path(args.experiment_dir)

    if args.all:
        experiments = find_completed_experiments(exp_path)
        if not experiments:
            print(f"No completed experiments found under {exp_path}")
            sys.exit(1)
        print(f"Found {len(experiments)} completed experiments:")
        for e in experiments:
            print(f"  • {e.name}")

        n_ok = 0
        for e in experiments:
            ok = evaluate_experiment(e, mvar_only=args.mvar_only,
                                     lstm_only=args.lstm_only)
            if ok:
                n_ok += 1
        print(f"\n{'=' * 72}")
        print(f"  Re-evaluated {n_ok}/{len(experiments)} experiments successfully")
        print(f"{'=' * 72}")
    else:
        if not exp_path.is_dir():
            print(f"Not a directory: {exp_path}")
            sys.exit(1)
        evaluate_experiment(exp_path, mvar_only=args.mvar_only,
                            lstm_only=args.lstm_only)


if __name__ == "__main__":
    main()
