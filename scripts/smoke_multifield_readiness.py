#!/usr/bin/env python3
"""Local smoke test for multifield WSINDy readiness.

This is intentionally lighter than the full production pipeline. It:
- uses `center_flux=True`
- fits on a truncated `test_000` field history
- uses a fixed ell and coarser query stride
- checks for moving, non-frozen, non-immediately-divergent forecasts

The goal is deployment readiness, not final scientific model selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rectsim.config_loader import load_config
from rectsim.ic_generator import generate_test_configs
from rectsim.legacy_functions import kde_density_movie
from rectsim.vicsek_discrete import simulate_backend
from wsindy.fields import build_field_data
from wsindy.multifield import (
    build_default_library,
    discover_multifield,
    forecast_multifield,
)

DEFAULT_ELL = (10, 6, 6)
DEFAULT_SMOKE_T = 40.0
DEFAULT_MAX_FRAMES = 300
DEFAULT_FORECAST_STEPS = 50


def _motion_energy(field_hist: np.ndarray) -> float:
    if field_hist.shape[0] <= 1:
        return 0.0
    diffs = np.diff(field_hist, axis=0)
    flat = diffs.reshape(diffs.shape[0], -1)
    return float(np.mean(np.linalg.norm(flat, axis=1)))


def _classify_sign(value: float | None, *, near: float = 1e-2) -> str:
    if value is None:
        return "inactive"
    if abs(value) < near:
        return "near_zero"
    return "positive" if value > 0 else "negative"


def _coeff_for(model: Any, name: str) -> float | None:
    if name not in model.col_names:
        return None
    idx = model.col_names.index(name)
    if not model.active[idx]:
        return 0.0
    return float(model.w[idx])


def _trim_series(
    rho: np.ndarray,
    traj: np.ndarray,
    vel: np.ndarray,
    max_frames: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = min(len(rho), len(traj), len(vel), max_frames)
    return rho[:t], traj[:t], vel[:t]


def _morse_params(raw_config: Dict[str, Any], mf_cfg: Dict[str, Any]) -> Dict[str, float] | None:
    if not mf_cfg.get("morse", True):
        return None
    forces_params = raw_config.get("forces", {}).get("params", {})
    return {
        "Cr": forces_params.get("Cr", 0.3),
        "Ca": forces_params.get("Ca", 0.8),
        "lr": forces_params.get("lr", 0.5),
        "la": forces_params.get("la", 1.5),
    }


def _load_local_test000(
    output_dir: Path,
    raw_config: Dict[str, Any],
    wsindy_cfg: Dict[str, Any],
    density_bandwidth: float,
    rom_config: Dict[str, Any],
    *,
    max_frames: int,
):
    mf_cfg = wsindy_cfg.get("multifield_library", {})
    run_dir = output_dir / "test" / "test_000"
    density_path = run_dir / "density_true.npz"
    traj_path = run_dir / "trajectory.npz"
    if not density_path.exists() or not traj_path.exists():
        raise FileNotFoundError(f"missing local test_000 artifacts in {run_dir}")

    d = np.load(density_path)
    td = np.load(traj_path)
    w_subsample = wsindy_cfg.get("subsample", rom_config.get("subsample", 3))
    rho = d["rho"][::w_subsample]
    traj = td["traj"][::w_subsample]
    vel = td["vel"][::w_subsample]
    rho, traj, vel = _trim_series(rho, traj, vel, max_frames)

    xgrid = d["xgrid"]
    ygrid = d["ygrid"]
    dt = float(d["times"][1] - d["times"][0]) * w_subsample
    lx = float(xgrid[-1] - xgrid[0] + (xgrid[1] - xgrid[0]))
    ly = float(ygrid[-1] - ygrid[0] + (ygrid[1] - ygrid[0]))

    fd = build_field_data(
        rho,
        traj,
        vel,
        xgrid,
        ygrid,
        lx,
        ly,
        dt,
        bandwidth=mf_cfg.get("kde_bandwidth", density_bandwidth),
        bc=raw_config.get("sim", {}).get("bc", "periodic"),
        subsample=1,
        morse_params=_morse_params(raw_config, mf_cfg),
        center_flux=True,
    )
    return fd, xgrid, ygrid, _morse_params(raw_config, mf_cfg)


def _simulate_test000_from_config(
    config_path: Path,
    *,
    smoke_t: float,
    max_frames: int,
):
    (
        base_config,
        density_nx,
        density_ny,
        density_bandwidth,
        _train_ic_config,
        test_ic_config,
        test_sim_config,
        rom_config,
        _eval_config,
    ) = load_config(config_path)
    raw = yaml.safe_load(config_path.read_text())
    wsindy_cfg = raw.get("wsindy", {})
    mf_cfg = wsindy_cfg.get("multifield_library", {})

    test_cfg = copy.deepcopy(generate_test_configs(test_ic_config, base_config)[0])
    base_test = copy.deepcopy(base_config)
    base_test["sim"] = copy.deepcopy(base_config["sim"])
    full_t = test_sim_config.get("T", test_ic_config.get("test_T", base_config["sim"]["T"]))
    base_test["sim"]["T"] = min(smoke_t, full_t)
    base_test["seed"] = test_cfg["run_id"] + 1000
    base_test["initial_distribution"] = test_cfg["distribution"]
    base_test["ic_params"] = test_cfg["ic_params"]

    rng = np.random.default_rng(base_test["seed"])
    result = simulate_backend(base_test, rng)
    times = result["times"]
    traj = result["traj"]
    vel = result["vel"]

    lx = float(base_test["sim"]["Lx"])
    ly = float(base_test["sim"]["Ly"])
    rho_full, _ = kde_density_movie(
        traj,
        Lx=lx,
        Ly=ly,
        nx=density_nx,
        ny=density_ny,
        bandwidth=density_bandwidth,
        bc=base_test["sim"].get("bc", "periodic"),
    )
    xgrid = np.linspace(0, lx, density_nx, endpoint=False) + lx / (2 * density_nx)
    ygrid = np.linspace(0, ly, density_ny, endpoint=False) + ly / (2 * density_ny)

    w_subsample = wsindy_cfg.get("subsample", rom_config.get("subsample", 3))
    dt = float(times[1] - times[0]) * w_subsample
    rho = rho_full[::w_subsample]
    traj = traj[::w_subsample]
    vel = vel[::w_subsample]
    rho, traj, vel = _trim_series(rho, traj, vel, max_frames)

    fd = build_field_data(
        rho,
        traj,
        vel,
        xgrid,
        ygrid,
        lx,
        ly,
        dt,
        bandwidth=mf_cfg.get("kde_bandwidth", density_bandwidth),
        bc=base_test["sim"].get("bc", "periodic"),
        subsample=1,
        morse_params=_morse_params(raw, mf_cfg),
        center_flux=True,
    )
    return raw, wsindy_cfg, rom_config, fd, xgrid, ygrid, _morse_params(raw, mf_cfg)


def run_regime(
    config_rel: str,
    *,
    ell: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    n_lambdas: int,
    smoke_t: float,
    max_frames: int,
    forecast_steps: int,
    prefer_local_output: bool,
) -> Dict[str, Any]:
    config_path = REPO / config_rel
    (
        _base_config,
        _density_nx,
        _density_ny,
        density_bandwidth,
        _train_ic_config,
        _test_ic_config,
        _test_sim_config,
        rom_config,
        _eval_config,
    ) = load_config(config_path)
    raw = yaml.safe_load(config_path.read_text())
    wsindy_cfg = raw.get("wsindy", {})
    rho_strategy = wsindy_cfg.get("multifield_library", {}).get("rho_strategy", "legacy")

    fd = None
    xgrid = None
    ygrid = None
    morse_params = None
    local_output = REPO / "oscar_output" / raw["experiment_name"]
    used_local_output = False
    if prefer_local_output:
        try:
            fd, xgrid, ygrid, morse_params = _load_local_test000(
                local_output,
                raw,
                wsindy_cfg,
                density_bandwidth,
                rom_config,
                max_frames=max_frames,
            )
            used_local_output = True
        except FileNotFoundError:
            pass

    if fd is None:
        raw, wsindy_cfg, rom_config, fd, xgrid, ygrid, morse_params = _simulate_test000_from_config(
            config_path,
            smoke_t=smoke_t,
            max_frames=max_frames,
        )

    lib = build_default_library(
        morse=wsindy_cfg.get("multifield_library", {}).get("morse", True),
        rich=wsindy_cfg.get("multifield_library", {}).get("rich", False),
        rho_strategy=rho_strategy,
    )
    lam_cfg = wsindy_cfg.get("lambdas", {})
    lambdas = np.logspace(
        lam_cfg.get("log_min", -5),
        lam_cfg.get("log_max", 2),
        n_lambdas,
    )
    p = tuple(wsindy_cfg.get("model_selection", {}).get("p", [3, 5, 5]))

    result = discover_multifield(
        [fd],
        lib,
        ell,
        p=p,
        stride=stride,
        lambdas=lambdas,
        rho_strategy=rho_strategy,
        verbose=False,
    )

    div_p_coeff = _coeff_for(result.rho_model, "div_p")
    lap_rho_coeff = _coeff_for(result.rho_model, "lap_rho")
    px_linear_coeff = _coeff_for(result.px_model, "px")

    n_steps = min(forecast_steps, fd.shape[0] - 1)
    try:
        rho_pred, px_pred, py_pred = forecast_multifield(
            fd.rho[0],
            fd.px[0],
            fd.py[0],
            result,
            fd.grid,
            Lx=fd.Lx,
            Ly=fd.Ly,
            n_steps=n_steps,
            clip_negative_rho=True,
            mass_conserve=True,
            method="auto",
            morse_params=morse_params,
            xgrid=xgrid,
            ygrid=ygrid,
        )
        rho_true = fd.rho[: n_steps + 1]
        true_motion = _motion_energy(rho_true)
        pred_motion = _motion_energy(rho_pred)
        motion_ratio = 1.0 if true_motion <= 1e-12 else float(pred_motion / true_motion)
        mass0 = float(np.sum(rho_pred[0]))
        massn = float(np.sum(rho_pred[-1]))
        mass_drift = 0.0 if abs(mass0) <= 1e-12 else float(abs(massn - mass0) / abs(mass0))
        forecast_status = "success"
        frozen = bool(np.max(np.abs(np.diff(rho_pred, axis=0))) == 0.0)
    except Exception as exc:  # pragma: no cover - smoke diagnostic path
        forecast_status = (
            f"failure@{getattr(exc, 'step', '?')}:{getattr(exc, 'reason', type(exc).__name__)}"
        )
        motion_ratio = None
        mass_drift = None
        frozen = False

    return {
        "experiment_name": raw["experiment_name"],
        "used_local_output": used_local_output,
        "fixed_ell": list(ell),
        "smoke_sub_frames": int(fd.shape[0]),
        "forecast_status": forecast_status,
        "motion_ratio": motion_ratio,
        "mass_conservation": mass_drift,
        "frozen": frozen,
        "div_p_coefficient": div_p_coeff,
        "lap_rho_sign": _classify_sign(lap_rho_coeff, near=1e-6),
        "lap_rho_coefficient": lap_rho_coeff,
        "px_linear_coeff_sign": _classify_sign(px_linear_coeff, near=1e-2),
        "px_linear_coefficient": px_linear_coeff,
        "px_constraints_dropped": result.px_model.diagnostics.get("sign_constraints_dropped", []),
        "py_constraints_dropped": result.py_model.diagnostics.get("sign_constraints_dropped", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", help="Config paths relative to repo root")
    parser.add_argument("--ell", nargs=3, type=int, default=DEFAULT_ELL)
    parser.add_argument("--stride", nargs=3, type=int, default=(4, 4, 4))
    parser.add_argument("--n-lambdas", type=int, default=20)
    parser.add_argument("--smoke-T", type=float, default=DEFAULT_SMOKE_T)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--forecast-steps", type=int, default=DEFAULT_FORECAST_STEPS)
    parser.add_argument("--prefer-local-output", action="store_true")
    args = parser.parse_args()

    results = []
    for config in args.configs:
        print(f"=== {config} ===", flush=True)
        out = run_regime(
            config,
            ell=tuple(args.ell),
            stride=tuple(args.stride),
            n_lambdas=args.n_lambdas,
            smoke_t=args.smoke_T,
            max_frames=args.max_frames,
            forecast_steps=args.forecast_steps,
            prefer_local_output=args.prefer_local_output,
        )
        print(json.dumps(out, indent=2), flush=True)
        results.append(out)

    print("=== FINAL_RESULTS ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
