#!/usr/bin/env python3
"""
Recover WSINDy multifield exports after a post-processing crash.

This script rebuilds Step 8 and the final lightweight export artifacts from:

* ``config_used.yaml``
* ``WSINDy/multifield_model.json``
* existing ``test/test_*/density_true.npz`` and ``trajectory.npz``

It is intended for OSCAR jobs that completed discovery but crashed before
writing ``density_pred_wsindy.npz``, ``WSINDy/test_results.csv``,
``summary.json``, and ``export_manifest.json``.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


def _repo_root() -> Path:
    here = Path(__file__) if "__file__" in globals() else Path.cwd() / "stdin_script.py"
    if here.name == "<stdin>":
        return Path.cwd()
    return here.resolve().parents[1]


ROOT = _repo_root()
sys.path.insert(0, str(ROOT / "src"))

from rectsim.config_loader import load_config
from wsindy.fields import compute_flux_kde
from wsindy.grid import GridSpec
from wsindy.model import WSINDyModel
from wsindy.multifield import MultiFieldResult, build_default_library as build_mf_library, forecast_multifield
from wsindy.eval import r2_per_snapshot


def align_eval_forecast_start(
    eval_config: Dict[str, Any],
    base_config_test: Dict[str, Any],
    rom_subsample: int,
    enabled_lags: Iterable[Optional[int]],
) -> Dict[str, Any]:
    """Mirror the pipeline's forecast-start alignment logic."""
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
    return eval_cfg


def compute_r2_timeseries(rho_true: np.ndarray, rho_pred: np.ndarray) -> np.ndarray:
    """Per-snapshot R² over the common rollout window."""
    T = min(rho_true.shape[0], rho_pred.shape[0])
    return np.array([r2_per_snapshot(rho_true[t], rho_pred[t]) for t in range(T)], dtype=np.float64)


def _safe_to_text(active_terms: List[str], coefficients: Dict[str, float], lhs: str) -> str:
    """Robust fallback text renderer for symbolic multifield terms."""
    if not active_terms:
        return f"{lhs} = 0"

    pieces: List[str] = []
    first = True
    for name in sorted(active_terms, key=lambda nm: -abs(coefficients.get(nm, 0.0))):
        coeff = float(coefficients[name])
        if first:
            coeff_str = f"{coeff:.4e}"
        elif coeff >= 0:
            coeff_str = f"+ {coeff:.4e}"
        else:
            coeff_str = f"- {abs(coeff):.4e}"
        pieces.append(f"{coeff_str} {name}")
        first = False
    return f"{lhs} = {' '.join(pieces)}"


def _load_single_model(section: Dict[str, Any]) -> WSINDyModel:
    col_names = list(section["col_names"])
    n_cols = len(col_names)
    return WSINDyModel(
        col_names=col_names,
        w=np.asarray(section["w"], dtype=np.float64),
        active=np.asarray(section["active"], dtype=bool),
        best_lambda=float(section.get("best_lambda", 0.0)),
        col_scale=np.ones(n_cols, dtype=np.float64),
        diagnostics={"r2": float(section.get("r2_weak", 0.0))},
    )


def load_multifield_result(exp_dir: Path, morse: bool, rich: bool) -> Tuple[MultiFieldResult, Dict[str, List[Any]]]:
    """Reconstruct MultiFieldResult from JSON plus deterministic library build."""
    model_path = exp_dir / "WSINDy" / "multifield_model.json"
    with open(model_path) as handle:
        data = json.load(handle)

    library = build_mf_library(morse=morse, rich=rich)
    ordered_terms: Dict[str, List[Any]] = {}
    models: Dict[str, WSINDyModel] = {}

    for eq_name in ["rho", "px", "py"]:
        section = data[eq_name]
        models[eq_name] = _load_single_model(section)
        by_name = {term.name: term for term in library[eq_name]}
        try:
            ordered_terms[eq_name] = [by_name[name] for name in section["col_names"]]
        except KeyError as exc:
            raise KeyError(f"Library term {exc} missing from reconstructed {eq_name} library") from exc

    result = MultiFieldResult(
        rho_model=models["rho"],
        px_model=models["px"],
        py_model=models["py"],
        rho_terms=ordered_terms["rho"],
        px_terms=ordered_terms["px"],
        py_terms=ordered_terms["py"],
    )
    return result, ordered_terms


def _parse_log_metadata(exp_dir: Path) -> Dict[str, Any]:
    """Best-effort recovery of model-selection metadata from slurm logs."""
    slurm_dir = exp_dir.parent.parent / "slurm_logs"
    if not slurm_dir.exists():
        return {}

    exp_name = exp_dir.name
    for log_path in sorted(slurm_dir.glob("*.out"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            text = log_path.read_text(errors="replace")
        except OSError:
            continue
        if f"Experiment: {exp_name}" not in text:
            continue

        out: Dict[str, Any] = {"slurm_log": str(log_path)}
        match = re.search(r"Model selection done in ([0-9.]+)s", text)
        if match:
            out["time_s"] = float(match.group(1))
        matches = re.findall(r"Best ℓ = \(([^)]+)\)", text)
        if matches:
            ell = tuple(int(part.strip()) for part in matches[-1].split(","))
            out["best_ell"] = list(ell)
        return out

    return {}


def _load_run_order(test_dir: Path) -> List[Dict[str, Any]]:
    meta_path = test_dir / "metadata.json"
    with open(meta_path) as handle:
        return json.load(handle)


def _build_summary(
    exp_dir: Path,
    config_path: Path,
    n_train: int,
    n_test: int,
    models_enabled: Dict[str, bool],
    eval_cfg: Dict[str, Any],
    wsindy_summary: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "experiment_name": exp_dir.name,
        "config": str(config_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_train": n_train,
        "n_test": n_test,
        "r_pod": None,
        "total_time_minutes": None,
        "models_enabled": models_enabled,
        "evaluation": {
            "forecast_start_requested_s": float(eval_cfg.get("forecast_start_requested", eval_cfg.get("forecast_start", 0.0))),
            "forecast_start_effective_s": float(eval_cfg.get("forecast_start_effective", eval_cfg.get("forecast_start", 0.0))),
            "forecast_start_conditioning_steps": int(eval_cfg.get("forecast_start_conditioning_steps", 0)),
            "forecast_start_required_lag": int(eval_cfg.get("forecast_start_required_lag", 0)),
        },
        "wsindy": wsindy_summary,
        "recovered_export": True,
    }


def _write_export_manifest(exp_dir: Path, n_test: int) -> None:
    files = [
        "summary.json",
        "config_used.yaml",
        "WSINDy/multifield_model.json",
        "WSINDy/wsindy_model_rho.npz",
        "WSINDy/wsindy_model_px.npz",
        "WSINDy/wsindy_model_py.npz",
        "WSINDy/test_results.csv",
        "WSINDy/runtime_profile.json",
        "test/metadata.json",
    ]
    for ti in range(n_test):
        prefix = f"test/test_{ti:03d}/"
        files += [
            prefix + "density_true.npz",
            prefix + "density_pred_wsindy.npz",
            prefix + "r2_vs_time_wsindy.csv",
            prefix + "density_metrics_wsindy.csv",
        ]

    with open(exp_dir / "export_manifest.json", "w") as handle:
        json.dump({"files": files}, handle, indent=2)


def recover_experiment(exp_dir: Path, overwrite: bool = False) -> None:
    exp_dir = exp_dir.resolve()
    config_path = exp_dir / "config_used.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config_used.yaml in {exp_dir}")

    with open(config_path) as handle:
        raw_config = yaml.safe_load(handle) or {}

    wsindy_config = raw_config.get("wsindy", {})
    if wsindy_config.get("mode") != "multifield":
        raise ValueError(f"{exp_dir.name} is not a multifield WSINDy experiment")

    (
        base_config,
        _density_nx,
        _density_ny,
        _density_bandwidth,
        _train_ic_config,
        test_ic_config,
        test_sim_config,
        rom_config,
        eval_config,
    ) = load_config(config_path)

    models_cfg = rom_config.get("models", {})
    models_enabled = {
        "mvar": bool(models_cfg.get("mvar", {}).get("enabled", False)),
        "lstm": bool(models_cfg.get("lstm", {}).get("enabled", False)),
        "wsindy": bool(wsindy_config.get("enabled", True)),
    }

    ws_cfg = wsindy_config
    mf_cfg = ws_cfg.get("multifield_library", {})
    morse_enabled = bool(mf_cfg.get("morse", True))
    rich = bool(mf_cfg.get("rich", False))
    kde_bw = float(mf_cfg.get("kde_bandwidth", 5.0))
    w_subsample = int(ws_cfg.get("subsample", rom_config.get("subsample", 3)))
    fc_cfg = ws_cfg.get("forecast", {})
    clip_neg = bool(fc_cfg.get("clip_negative", True))
    mass_conserve = bool(fc_cfg.get("mass_conserve", True))
    fc_method = str(fc_cfg.get("method", "auto"))

    test_horizon = test_sim_config.get("T", test_ic_config.get("test_T", base_config["sim"]["T"]))
    base_config_test = copy.deepcopy(base_config)
    base_config_test["sim"]["T"] = test_horizon
    eval_cfg = align_eval_forecast_start(eval_config, base_config_test, rom_config.get("subsample", 1), [])

    model_result, ordered_terms = load_multifield_result(exp_dir, morse=morse_enabled, rich=rich)

    test_dir = exp_dir / "test"
    test_meta_list = _load_run_order(test_dir)
    if not test_meta_list:
        raise RuntimeError(f"No test metadata found in {test_dir}")

    train_meta_path = exp_dir / "train" / "metadata.json"
    n_train = len(json.load(open(train_meta_path))) if train_meta_path.exists() else int(ws_cfg.get("n_train", 0))
    n_test = len(test_meta_list)

    first_density = np.load(test_dir / test_meta_list[0]["run_name"] / "density_true.npz")
    xgrid = first_density["xgrid"]
    ygrid = first_density["ygrid"]
    dx = float(xgrid[1] - xgrid[0])
    dy = float(ygrid[1] - ygrid[0])
    Lx = float(xgrid[-1] - xgrid[0]) + dx
    Ly = float(ygrid[-1] - ygrid[0]) + dy
    dt_base = float(first_density["times"][1] - first_density["times"][0])
    grid = GridSpec(dt=dt_base * w_subsample, dx=dx, dy=dy)

    forecast_start_s = float(eval_cfg.get("forecast_start_effective", eval_cfg.get("forecast_start", base_config["sim"]["T"])))
    forecast_start = int(round(forecast_start_s / (base_config_test["sim"]["dt"] * w_subsample)))

    forces_params = raw_config.get("forces", {}).get("params", {})
    morse_params = None
    if morse_enabled:
        morse_params = {
            "Cr": float(forces_params.get("Cr", 0.3)),
            "Ca": float(forces_params.get("Ca", 0.8)),
            "lr": float(forces_params.get("lr", 0.5)),
            "la": float(forces_params.get("la", 1.5)),
        }

    wsindy_dir = exp_dir / "WSINDy"
    wsindy_dir.mkdir(exist_ok=True)
    (wsindy_dir / "plots").mkdir(exist_ok=True)

    test_results: List[Dict[str, Any]] = []
    for ti, meta in enumerate(test_meta_list):
        run_name = meta["run_name"]
        run_dir = test_dir / run_name
        pred_path = run_dir / "density_pred_wsindy.npz"
        if pred_path.exists() and not overwrite:
            existing_summary_path = run_dir / "metrics_summary_wsindy.json"
            existing_r2_path = run_dir / "r2_vs_time_wsindy.csv"
            if existing_summary_path.exists():
                with open(existing_summary_path) as handle:
                    existing_summary = json.load(handle)
                test_results.append(
                    {
                        "test_id": ti,
                        "run_name": run_name,
                        "r2_reconstructed": float(existing_summary.get("r2_recon", float("nan"))),
                        "r2_latent": float(existing_summary.get("r2_recon", float("nan"))),
                        "r2_pod": 1.0,
                        "rmse_recon": float(existing_summary.get("rmse_recon", float("nan"))),
                        "forecast_method": existing_summary.get("method", f"{fc_method}_multifield"),
                        "n_forecast_steps": int(existing_summary.get("n_forecast_steps", 0)),
                    }
                )
            elif existing_r2_path.exists():
                existing_r2 = pd.read_csv(existing_r2_path)["r2_reconstructed"].to_numpy(dtype=np.float64)
                mean_r2_existing = float(np.nanmean(existing_r2[1:])) if len(existing_r2) > 1 else float(existing_r2[0])
                test_results.append(
                    {
                        "test_id": ti,
                        "run_name": run_name,
                        "r2_reconstructed": mean_r2_existing,
                        "r2_latent": mean_r2_existing,
                        "r2_pod": 1.0,
                        "rmse_recon": float("nan"),
                        "forecast_method": f"{fc_method}_multifield",
                        "n_forecast_steps": max(len(existing_r2) - 1, 0),
                    }
                )
            print(f"[skip] {exp_dir.name} {run_name}: density_pred_wsindy.npz already exists")
            continue

        density = np.load(run_dir / "density_true.npz")
        rho_true = density["rho"][::w_subsample]
        times_sub = density["times"][::w_subsample]
        T_test_sub = rho_true.shape[0]

        n_fc = T_test_sub - forecast_start - 1
        if n_fc <= 0:
            print(f"[skip] {exp_dir.name} {run_name}: forecast window is empty")
            continue

        traj_data = np.load(run_dir / "trajectory.npz")
        traj_test = traj_data["traj"][::w_subsample]
        vel_test = traj_data["vel"][::w_subsample]
        px0_arr, py0_arr = compute_flux_kde(
            traj_test[forecast_start:forecast_start + 1],
            vel_test[forecast_start:forecast_start + 1],
            xgrid,
            ygrid,
            Lx,
            Ly,
            bandwidth=kde_bw,
        )
        px0 = px0_arr[0]
        py0 = py0_arr[0]
        rho0 = rho_true[forecast_start]

        method_used = None
        try:
            rho_pred, px_pred, py_pred = forecast_multifield(
                rho0,
                px0,
                py0,
                model_result,
                grid,
                Lx=Lx,
                Ly=Ly,
                n_steps=n_fc,
                clip_negative_rho=clip_neg,
                mass_conserve=mass_conserve,
                method=fc_method,
                morse_params=morse_params,
                xgrid=xgrid,
                ygrid=ygrid,
            )
            method_used = f"{fc_method}_multifield"
        except Exception as exc:
            if fc_method == "auto":
                print(f"[warn] {exp_dir.name} {run_name}: auto multifield forecast failed - {exc}")
                print(f"[warn] {exp_dir.name} {run_name}: retrying multifield forecast with RK4")
                try:
                    rho_pred, px_pred, py_pred = forecast_multifield(
                        rho0,
                        px0,
                        py0,
                        model_result,
                        grid,
                        Lx=Lx,
                        Ly=Ly,
                        n_steps=n_fc,
                        clip_negative_rho=clip_neg,
                        mass_conserve=mass_conserve,
                        method="rk4",
                        morse_params=morse_params,
                        xgrid=xgrid,
                        ygrid=ygrid,
                    )
                    method_used = "rk4_multifield"
                except Exception as rk4_exc:
                    exc = RuntimeError(
                        f"WSINDy multifield forecast failed with auto and RK4. "
                        f"Auto error: {exc}. RK4 error: {rk4_exc}"
                    )
            if method_used is None:
                print(f"[fail] {exp_dir.name} {run_name}: forecast failed - {exc}")
                test_results.append(
                    {
                        "test_id": ti,
                        "run_name": run_name,
                        "r2_reconstructed": float("nan"),
                        "r2_latent": float("nan"),
                        "r2_pod": float("nan"),
                        "rmse_recon": float("nan"),
                        "forecast_method": f"{fc_method}_multifield",
                        "n_forecast_steps": int(n_fc),
                        "failure": str(exc),
                    }
                )
                continue

        rho_true_fc = rho_true[forecast_start:forecast_start + n_fc + 1]
        times_fc = times_sub[forecast_start:forecast_start + n_fc + 1]
        r2_ts = compute_r2_timeseries(rho_true_fc, rho_pred)
        rmse_ts = np.sqrt(np.mean((rho_true_fc[: len(r2_ts)] - rho_pred[: len(r2_ts)]) ** 2, axis=(1, 2)))
        mean_r2 = float(np.nanmean(r2_ts[1:])) if len(r2_ts) > 1 else float(r2_ts[0])
        mean_rmse = float(np.nanmean(rmse_ts[1:])) if len(rmse_ts) > 1 else float(rmse_ts[0])

        np.savez_compressed(
            pred_path,
            rho=rho_pred.astype(np.float32),
            px=px_pred.astype(np.float32),
            py=py_pred.astype(np.float32),
            xgrid=xgrid,
            ygrid=ygrid,
            times=np.asarray(times_fc, dtype=np.float32),
            forecast_start_idx=0,
        )

        pd.DataFrame(
            {
                "time": times_fc[: len(r2_ts)],
                "r2_reconstructed": r2_ts,
                "r2_latent": r2_ts,
                "r2_pod": np.ones(len(r2_ts), dtype=np.float64),
            }
        ).to_csv(run_dir / "r2_vs_time_wsindy.csv", index=False)

        with open(run_dir / "metrics_summary_wsindy.json", "w") as handle:
            json.dump(
                {
                    "r2_recon": mean_r2,
                    "rmse_recon": mean_rmse,
                    "method": method_used,
                    "n_forecast_steps": int(n_fc),
                },
                handle,
                indent=2,
            )

        T_pred = rho_pred.shape[0]
        pd.DataFrame(
            {
                "t": times_fc[:T_pred],
                "density_variance_true": np.std(rho_true_fc[:T_pred], axis=(1, 2)),
                "density_variance_pred": np.std(rho_pred[:T_pred], axis=(1, 2)),
                "mass_true": np.sum(rho_true_fc[:T_pred], axis=(1, 2)),
                "mass_pred": np.sum(rho_pred[:T_pred], axis=(1, 2)),
            }
        ).to_csv(run_dir / "density_metrics_wsindy.csv", index=False)

        test_results.append(
            {
                "test_id": ti,
                "run_name": run_name,
                "r2_reconstructed": mean_r2,
                "r2_latent": mean_r2,
                "r2_pod": 1.0,
                "rmse_recon": mean_rmse,
                "forecast_method": method_used,
                "n_forecast_steps": int(n_fc),
            }
        )
        print(f"[ok] {exp_dir.name} {run_name}: mean R2={mean_r2:.4f}")

    results_df = pd.DataFrame(test_results)
    results_df.to_csv(wsindy_dir / "test_results.csv", index=False)
    mean_r2_wsindy = float(results_df["r2_reconstructed"].mean()) if not results_df.empty else float("nan")
    std_r2_wsindy = float(results_df["r2_reconstructed"].std()) if len(results_df) > 1 else 0.0

    log_meta = _parse_log_metadata(exp_dir)
    wsindy_summary: Dict[str, Any] = {
        "mode": "multifield",
        "discovered_pde": {},
        "library": {
            "n_terms_total": sum(len(v) for v in ordered_terms.values()),
            "morse": morse_enabled,
            "rich": rich,
            "per_equation": {eq: [t.name for t in terms] for eq, terms in ordered_terms.items()},
        },
        "n_train_trajectories": int(ws_cfg.get("n_train", n_train)),
        "test_evaluation": {
            "mean_r2": None if np.isnan(mean_r2_wsindy) else mean_r2_wsindy,
            "std_r2": None if np.isnan(std_r2_wsindy) else std_r2_wsindy,
            "n_test": n_test,
        },
    }
    if "best_ell" in log_meta or "time_s" in log_meta:
        wsindy_summary["model_selection"] = {}
        if "best_ell" in log_meta:
            wsindy_summary["model_selection"]["best_ell"] = log_meta["best_ell"]
        if "time_s" in log_meta:
            wsindy_summary["model_selection"]["time_s"] = log_meta["time_s"]

    lhs_map = {"rho": "rho_t", "px": "p_x_t", "py": "p_y_t"}
    for eq_name in ["rho", "px", "py"]:
        model = getattr(model_result, f"{eq_name}_model")
        coeffs = {
            name: float(model.w[model.col_names.index(name)])
            for name in model.active_terms
        }
        wsindy_summary["discovered_pde"][eq_name] = {
            "text": _safe_to_text(model.active_terms, coeffs, lhs_map[eq_name]),
            "active_terms": model.active_terms,
            "coefficients": coeffs,
            "n_active": model.n_active,
            "lambda_star": float(model.best_lambda),
            "r2_weak": float(model.diagnostics.get("r2", 0.0)),
        }

    runtime_profile = {
        "model_name": "WSINDy",
        "training_time_seconds": float(log_meta.get("time_s", 0.0)),
        "model_params": int(sum(getattr(model_result, f"{eq}_model").n_active for eq in ["rho", "px", "py"])),
        "inference": {
            "single_step": {
                "mean_seconds": 0.0,
            },
        },
        "notes": "Recovered export from saved multifield WSINDy artifacts",
    }
    with open(wsindy_dir / "runtime_profile.json", "w") as handle:
        json.dump(runtime_profile, handle, indent=2)

    summary = _build_summary(
        exp_dir=exp_dir,
        config_path=config_path,
        n_train=n_train,
        n_test=n_test,
        models_enabled=models_enabled,
        eval_cfg=eval_cfg,
        wsindy_summary=wsindy_summary,
    )
    with open(exp_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    _write_export_manifest(exp_dir, n_test)
    print(f"[done] {exp_dir.name}: wrote summary.json, export_manifest.json, and WSINDy predictions")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover WSINDy multifield exports from saved artifacts")
    parser.add_argument("experiment_dirs", nargs="+", help="Experiment directories, e.g. oscar_output/WSY_DO_DR02_dring_C09_l09")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing WSINDy prediction files")
    args = parser.parse_args()

    for exp in args.experiment_dirs:
        recover_experiment(Path(exp), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
