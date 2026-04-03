#!/usr/bin/env python3
"""Standalone multifield WSINDy discovery + rollout on K6 clustering data.

Loads existing oscar_output/K6_v1_rawD19_p5_H37/train/ data directly
(Ca=0.8, Cr=0.3, gentle clustering regime) and runs the full 3-equation
multifield WSINDy pipeline with all post-regression diagnostics, then
performs a symmetric-Morse-enforced rollout on one test trajectory.

Results are written to oscar_output/K6_v1_rawD19_p5_H37/WSINDy/.

Usage
-----
  cd /path/to/wsindy-manifold
  python scripts/run_wsindy_on_k6.py [--n_train N] [--n_ell K] [--n_steps S]
                                      [--dry_run] [--skip_discovery]

  --skip_discovery  Load existing discovery_summary.json and go straight
                    to the rollout (much faster on re-runs).
  --n_steps         Rollout frames  (default: 50)
"""
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE / "src"))

K6_DIR   = WORKSPACE / "oscar_output" / "K6_v1_rawD19_p5_H37"
TRAIN_DIR = K6_DIR / "train"
OUT_DIR   = K6_DIR / "WSINDy"

# ── Physics parameters (read from config_used.yaml / verified from data) ─────
DT_BASE     = 0.04     # raw simulation time step
SUBSAMPLE   = 3        # temporal sub-sampling to apply
DT          = DT_BASE * SUBSAMPLE   # effective dt = 0.12
LX          = 15.0
LY          = 15.0
BANDWIDTH   = 4.0      # KDE bandwidth for flux fields (grid cells)
MORSE_PARAMS = dict(Ca=0.8, Cr=0.3, la=1.5, lr=0.5)
REGIME_CLASS = "attractive"   # Ca/Cr ≈ 2.67 > 1

# ── WSINDy hyper-parameters (matching DYN1_gentle_wsindy.yaml) ───────────────
N_TRAIN_DEFAULT = 30
N_ELL_DEFAULT   = 8
SEED            = 42
P               = (3, 5, 5)     # polynomial bump widths for (ρ, p_x, p_y)
STRIDE          = (2, 2, 2)     # query-point stride
LAMBDAS         = np.logspace(-5, 2, 60)  # MSTLS regularisation grid


# ── Lazy imports (after sys.path patch) ──────────────────────────────────────
def _imports():
    global build_field_data, build_default_library
    global model_selection_multifield, fit_equation_multifield
    global forecast_multifield, LibraryTerm
    global default_ell_grid
    global GridSpec
    global ols_comparison, residual_analysis
    global dominant_balance_report, print_dominant_balance, model_aic
    global to_text

    from wsindy.fields import build_field_data as _bfd
    build_field_data = _bfd

    from wsindy.multifield import (
        build_default_library,
        model_selection_multifield,
        fit_equation_multifield,
        forecast_multifield,
        LibraryTerm,
    )
    from wsindy.select import default_ell_grid
    from wsindy.grid import GridSpec
    from wsindy.diagnostics import (
        ols_comparison,
        residual_analysis,
        dominant_balance_report,
        print_dominant_balance,
        model_aic,
    )
    from wsindy.pretty import to_text


# ── Data loading ──────────────────────────────────────────────────────────────
def load_run_dirs(n_train: int, seed: int) -> list:
    """Return `n_train` randomly selected run dirs that have both data files."""
    rng = np.random.default_rng(seed)
    available = sorted(
        d for d in TRAIN_DIR.iterdir()
        if d.is_dir()
        and d.name.startswith("train_")
        and (d / "density.npz").exists()
        and (d / "trajectory.npz").exists()
    )
    if not available:
        raise FileNotFoundError(f"No paired training data in {TRAIN_DIR}")

    n = min(n_train, len(available))
    if n < n_train:
        print(f"  WARNING: only {n} runs available (requested {n_train})")
    idx = rng.choice(len(available), size=n, replace=False)
    selected = [available[i] for i in sorted(idx)]
    print(f"  Selecting {n} / {len(available)} available runs  (seed={seed})")
    return selected


def build_field_data_list(run_dirs: list) -> list:
    """Build a FieldData object for each training run."""
    # Extract grid arrays from the first run (shared across all runs)
    d0 = np.load(run_dirs[0] / "density.npz")
    xgrid = d0["xgrid"]
    ygrid = d0["ygrid"]

    field_data_list = []
    for i, run_dir in enumerate(run_dirs):
        dd = np.load(run_dir / "density.npz")
        td = np.load(run_dir / "trajectory.npz")

        rho  = dd["rho"][::SUBSAMPLE]          # (T_sub, ny, nx)
        traj = td["traj"][::SUBSAMPLE]         # (T_sub, N, 2)
        vel  = td["vel"][::SUBSAMPLE]          # (T_sub, N, 2)

        fd = build_field_data(
            rho, traj, vel,
            xgrid, ygrid, LX, LY, DT,
            bandwidth=BANDWIDTH,
            morse_params=MORSE_PARAMS,
            center_flux=True,
        )
        field_data_list.append(fd)

        if i == 0 or (i + 1) % 10 == 0:
            print(f"    [{i+1:3d}/{len(run_dirs)}] {run_dir.name}  rho∈"
                  f"[{rho.min():.2e}, {rho.max():.2e}]")

    return field_data_list


# ── Diagnostics ───────────────────────────────────────────────────────────────
def run_diagnostics(b, G, model, label: str) -> dict:
    """Run all 4 post-regression diagnostics; return JSON-safe dict."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diag_out = {}

    # 1. OLS comparison ─────────────────────────────────────────────────────
    try:
        ols_res = ols_comparison(b, G, model)
        diag_out["ols_comparison"] = {
            "r2_ols":       ols_res["r2_ols"],
            "r2_mstls":     ols_res["r2_mstls"],
            "max_rel_diff": ols_res["max_rel_diff"],
        }
        print(f"    OLS vs MSTLS — max rel diff: {ols_res['max_rel_diff']:.4f}")
        print(f"      R²(OLS)={ols_res['r2_ols']:.6f}  "
              f"R²(MSTLS)={ols_res['r2_mstls']:.6f}")
    except Exception as exc:
        print(f"    [WARN] OLS comparison failed: {exc}")

    # 2. Residual analysis ──────────────────────────────────────────────────
    try:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        resid_res = residual_analysis(b, G, model, plot=True, ax=ax)
        fig.suptitle(f"Residuals — {label} (K6 clustering)", y=1.02)
        fig.tight_layout()
        save_path = OUT_DIR / f"residual_histogram_{label}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        diag_out["residual_analysis"] = {
            "mean":      resid_res["mean"],
            "std":       resid_res["std"],
            "skewness":  resid_res["skewness"],
            "kurtosis":  resid_res["kurtosis"],
            "shapiro_p": resid_res["shapiro_p"],
            "max_abs":   resid_res["max_abs"],
        }
        print(f"    Residuals: μ={resid_res['mean']:.3e}  "
              f"σ={resid_res['std']:.3e}  "
              f"kurt={resid_res['kurtosis']:.2f}  "
              f"Shapiro p={resid_res['shapiro_p']:.3e}")
        print(f"      Saved: {save_path.name}")
    except Exception as exc:
        print(f"    [WARN] Residual analysis failed: {exc}")

    # 3. Dominant balance ───────────────────────────────────────────────────
    try:
        balance = dominant_balance_report(b, G, model)
        diag_out["dominant_balance"] = {
            "groups": [
                {"term": g["term"], "Pi": g["Pi"], "w": g["w"]}
                for g in balance["groups"]
            ],
            "Pi_sum": balance["Pi_sum"],
        }
        print(print_dominant_balance(balance))
    except Exception as exc:
        print(f"    [WARN] Dominant balance failed: {exc}")

    # 4. AIC ────────────────────────────────────────────────────────────────
    try:
        aic = model_aic(b, G, model)
        diag_out["aic"] = aic
        print(f"    AIC = {aic:.2f}")
    except Exception as exc:
        print(f"    [WARN] AIC failed: {exc}")

    return diag_out


# ── Main ──────────────────────────────────────────────────────────────────────
def _symmetrise_morse(result) -> None:
    """Force px_t to carry rho_dx_Phi with the same coefficient as
    rho_dy_Phi in py_t, enforcing isotropy of the Morse force.

    Modifies result.px_model in-place (w and active arrays).
    If py_t does not include rho_dy_Phi, this is a no-op.
    """
    py_cn = result.py_model.col_names
    px_cn = result.px_model.col_names

    if "rho_dy_Phi" not in py_cn or not result.py_model.active[py_cn.index("rho_dy_Phi")]:
        print("  [symmetrise] rho_dy_Phi not active in py_t — skipping")
        return

    morse_coeff = float(result.py_model.w[py_cn.index("rho_dy_Phi")])

    if "rho_dx_Phi" not in px_cn:
        print("  [symmetrise] rho_dx_Phi not in px library — skipping")
        return

    dx_idx = px_cn.index("rho_dx_Phi")
    result.px_model.w[dx_idx]      = morse_coeff
    result.px_model.active[dx_idx] = True
    print(f"  [symmetrise] Set px_t rho_dx_Phi = {morse_coeff:+.6e}  "
          f"(mirror of py_t rho_dy_Phi)")


def _load_result_from_summary(summary_path: Path):
    """Reconstruct a minimal MultiFieldResult from a saved discovery_summary.json
    (enough for forecast — no b/G matrices needed)."""
    from wsindy.model import WSINDyModel
    from wsindy.multifield import MultiFieldResult, build_default_library
    import dataclasses
    import numpy as np

    with open(summary_path) as fh:
        s = json.load(fh)

    lib = build_default_library(morse=True, rich=False, regime_class="attractive")

    def _make_model(eq_data, lib_terms):
        col_names = eq_data["col_names"]
        xi = np.array(eq_data["xi"])
        active = np.array([bool(a) for a in eq_data.get(
            "active", [abs(w) > 1e-30 for w in xi])], dtype=bool)
        diags = {"r2": eq_data.get("r2", 0.0)}
        return WSINDyModel(
            col_names=col_names,
            w=xi,
            active=active,
            best_lambda=eq_data.get("best_lambda", 0.0),
            col_scale=np.ones(len(col_names)),
            diagnostics=diags,
        )

    rho_model = _make_model(s["equations"]["rho"], lib["rho"])
    px_model  = _make_model(s["equations"]["px"],  lib["px"])
    py_model  = _make_model(s["equations"]["py"],  lib["py"])

    best_ell = tuple(s["wsindy"]["best_ell"])
    result = MultiFieldResult(
        rho_model=rho_model, px_model=px_model, py_model=py_model,
        rho_terms=lib["rho"], px_terms=lib["px"], py_terms=lib["py"],
    )
    return result, best_ell


def _drop_cubic_terms(result) -> None:
    """Zero out |p|²p_x and |p|²p_y before rollout.

    These terms have |coeff| ~ 1e-3 and Π < 0.05, so they are negligible
    for the forecast quality.  However, raw KDE flux values can reach
    |p| ~ 10+ at isolated pixels, making |p|²·p ~ 10³ which overruns the
    diffusion limiter inside the ETDRK4 step and causes overflow by step ~6.
    Dropping them makes the momentum equations linear and unconditionally
    stable under ETDRK4.
    """
    for model, term_name in [
        (result.px_model, "p_sq_px"),
        (result.py_model, "p_sq_py"),
    ]:
        if term_name in model.col_names:
            idx = model.col_names.index(term_name)
            if model.active[idx]:
                coeff = model.w[idx]
                model.w[idx]      = 0.0
                model.active[idx] = False
                print(f"  [drop_cubic] Zeroed {term_name}  "
                      f"(coeff was {coeff:+.3e}, Π < 0.05 — rollout-unsafe)")


def _drop_rho_morse_term(result) -> None:
    """Zero out div_rho_gradPhi and lap_rho2 from the rho equation.

    div_rho_gradPhi: coeff +5.2e-4 (positive, anti-diffusive Morse source).
    lap_rho2: coeff -4.2e-4.  Δ(ρ²) at density peaks is negative because
      the Laplacian at a peak is negative and dominates; with the negative
      coefficient the contribution is +|c|·|Δ(ρ²)| > 0 — anti-diffusive.
      On the highly-concentrated IC (Gini=0.85) both terms grow density at
      peaks and cause overflow around step 10.  Keeping only div_p + lap_rho
      gives the exact continuity equation with weak regularisation.
    """
    for term in ("div_rho_gradPhi", "lap_rho2"):
        model = result.rho_model
        if term in model.col_names:
            idx = model.col_names.index(term)
            if model.active[idx]:
                coeff = model.w[idx]
                model.w[idx]      = 0.0
                model.active[idx] = False
                print(f"  [drop_rho_nonlinear] Zeroed {term}  "
                      f"(coeff={coeff:+.3e} — anti-diffusive at IC, unsafe)")


def _drop_momentum_morse_terms(result) -> None:
    """Zero out rho_dx_Phi and rho_dy_Phi from the momentum equations.

    These Morse forcing terms (Π ≈ 0.22 in py_t) drive flux toward density
    maxima — physically correct for clustering, but the feedback loop
    ρ → Φ → flux → ρ → Φ → ... causes finite-time blowup in the continuum
    PDE without a congestion/saturation term (which the particle model has
    implicitly through finite-size exclusion).  The pressure-diffusion
    sub-model (dx_rho + lap_px/py) is stable and captures how the
    mean-field redistributes density.

    The identification result (Morse terms ARE present with correct sign)
    is the scientific finding; the forecast uses the stable sub-model.
    """
    for model, term_name in [
        (result.px_model, "rho_dx_Phi"),
        (result.py_model, "rho_dy_Phi"),
    ]:
        if term_name in model.col_names:
            idx = model.col_names.index(term_name)
            if model.active[idx]:
                coeff = model.w[idx]
                model.w[idx]      = 0.0
                model.active[idx] = False
                print(f"  [drop_morse_momentum] Zeroed {term_name}  "
                      f"(coeff={coeff:+.3e} — Morse blowup without congestion term)")


def run_rollout(result, n_steps: int) -> None:
    """Run an n_steps forecast on test_000, enforce symmetric Morse,
    and print the forecast status table."""
    TEST_DIR = K6_DIR / "test"
    test_dirs = sorted(d for d in TEST_DIR.iterdir()
                       if d.is_dir() and d.name.startswith("test_"))
    if not test_dirs:
        print("  [rollout] No test directories found — skipping")
        return

    test_dir = test_dirs[0]
    print(f"  Test trajectory: {test_dir.name}")

    # Load ground truth
    dd = np.load(test_dir / "density_true.npz")
    rho_gt = dd["rho"][::SUBSAMPLE]       # (T_sub, ny, nx)
    xgrid  = dd["xgrid"]
    ygrid  = dd["ygrid"]

    # Build flux IC from trajectory
    td = np.load(test_dir / "trajectory.npz")
    from wsindy.fields import compute_flux_kde
    px_full, py_full = compute_flux_kde(
        td["traj"][::SUBSAMPLE], td["vel"][::SUBSAMPLE],
        xgrid, ygrid, LX, LY,
        bandwidth=BANDWIDTH, bc="periodic", subsample=1,
    )
    T_test = min(rho_gt.shape[0], px_full.shape[0])
    rho_gt  = rho_gt[:T_test]
    px_full = px_full[:T_test]
    py_full = py_full[:T_test]

    # IC = frame 0; center flux to match training (center_flux=True was used)
    rho0 = rho_gt[0].copy()
    px0  = px_full[0].copy()
    py0  = py_full[0].copy()
    px0 -= np.mean(px0)   # remove spatial mean (matches center_flux=True in training)
    py0 -= np.mean(py0)
    M0   = float(np.sum(rho0))

    # Enforce symmetric Morse before rollout
    print("\n  Enforcing Morse symmetry (px_t ← rho_dx_Phi = py_t coeff):")
    _symmetrise_morse(result)

    # Drop cubic |p|²p terms (rollout-unsafe at raw KDE flux magnitudes)
    print("  Dropping cubic terms before rollout (|p|²p, Π < 0.05):")
    _drop_cubic_terms(result)

    # Drop rho Morse term (positive coeff → anti-diffusive aggregation → blowup)
    _drop_rho_morse_term(result)

    # Drop momentum Morse terms (aggregation feedback → blowup without congestion)
    _drop_momentum_morse_terms(result)
    print(f"  rho_t active: {result.rho_model.active_terms}")
    print(f"  px_t active:  {result.px_model.active_terms}")
    print(f"  py_t active:  {result.py_model.active_terms}")

    grid = GridSpec(dt=DT, dx=LX / len(xgrid), dy=LY / len(ygrid))

    clamp_steps = min(n_steps, T_test - 1)
    print(f"\n  Running {clamp_steps}-step ETDRK4 rollout...", flush=True)

    import time as _time
    t0 = _time.perf_counter()
    try:
        rho_hist, px_hist, py_hist = forecast_multifield(
            rho0, px0, py0,
            result, grid, LX, LY,
            n_steps=clamp_steps,
            morse_params=None,   # Morse terms dropped from model; no potential needed
            xgrid=xgrid, ygrid=ygrid,
            clip_negative_rho=True,
            mass_conserve=True,
            method="auto",
        )
        elapsed = _time.perf_counter() - t0
        integrator = result.metadata.get("last_forecast_method_used", "unknown")
        forecast_ok = True
    except Exception as exc:
        elapsed = _time.perf_counter() - t0
        print(f"  [WARN] Rollout failed: {exc}")
        forecast_ok = False
        integrator = "FAILED"

    M_final = float(np.sum(rho_hist[-1])) if forecast_ok else float("nan")
    motion  = float(np.mean(np.std(rho_hist, axis=0))) if forecast_ok else float("nan")
    motion0 = float(np.mean(np.std(rho_gt[:clamp_steps+1], axis=0)))
    motion_ratio = motion / motion0 if motion0 > 1e-12 else float("nan")

    def rmse_at(frame):
        if not forecast_ok or frame >= rho_hist.shape[0] or frame >= rho_gt.shape[0]:
            return float("nan")
        return float(np.sqrt(np.mean((rho_hist[frame] - rho_gt[frame]) ** 2)))

    rmse10 = rmse_at(10)
    rmse25 = rmse_at(25)
    rmse50 = rmse_at(min(50, clamp_steps))

    # Does the density visually cluster? Gini coefficient of final frame
    def gini(arr):
        arr = arr.ravel()
        arr = arr[arr > 0]
        if len(arr) == 0:
            return 0.0
        arr = np.sort(arr)
        n = len(arr)
        return float((2 * np.sum(np.arange(1, n+1) * arr) / (n * arr.sum())) - (n+1)/n)

    gini_ic    = gini(rho0)
    gini_final = gini(rho_hist[-1]) if forecast_ok else float("nan")

    print()
    print("=" * 65)
    print("  FORECAST STATUS TABLE — K6 Gentle Clustering")
    print("=" * 65)
    print(f"  Forecast status    : {'SUCCESS' if forecast_ok else 'FAILED'}")
    print(f"  Integrator used    : {integrator}")
    print(f"  Elapsed time       : {elapsed:.1f}s")
    print(f"  Motion ratio       : {motion_ratio:.4f}  (1.0 = same variance as GT)")
    print(f"  Mass at frame {clamp_steps:3d}  : {M_final:.4f}  (IC mass = {M0:.4f})")
    print(f"  Frame 10  rho RMSE : {rmse10:.4f}")
    print(f"  Frame 25  rho RMSE : {rmse25:.4f}")
    print(f"  Frame 50  rho RMSE : {rmse50:.4f}")
    print(f"  Gini(rho, IC)      : {gini_ic:.4f}")
    print(f"  Gini(rho, frame {clamp_steps:2d}) : {gini_final:.4f}  "
          f"(clustering if > IC)")
    print("=" * 65)

    if forecast_ok:
        np.savez_compressed(
            OUT_DIR / "rollout_test000.npz",
            rho_hist=rho_hist,
            px_hist=px_hist,
            py_hist=py_hist,
            rho_gt=rho_gt[:clamp_steps+1],
        )
        print(f"  Saved rollout: {OUT_DIR / 'rollout_test000.npz'}")


def main(n_train: int, n_ell: int, dry_run: bool, n_steps: int = 50,
         skip_discovery: bool = False) -> None:
    _imports()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  WSINDy Multifield Discovery — K6 Gentle Clustering Regime")
    print(f"  Ca={MORSE_PARAMS['Ca']}, Cr={MORSE_PARAMS['Cr']}, "
          f"η=0.2, N=100, Lx={LX}")
    print(f"  n_train={n_train}, n_ell={n_ell}, p={P}, stride={STRIDE}")
    print(f"  subsample={SUBSAMPLE} → dt_eff={DT}")
    print("=" * 65)

    # ── Skip discovery and load from disk ────────────────────────────────────
    if skip_discovery:
        summary_path = OUT_DIR / "discovery_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"--skip_discovery requires {summary_path} — run discovery first"
            )
        _imports()
        print(f"\n  Loading existing discovery from {summary_path}")
        result, best_ell = _load_result_from_summary(summary_path)
        print(f"  Best ell: {best_ell}")
        for eq, model in [("rho", result.rho_model), ("px", result.px_model),
                          ("py", result.py_model)]:
            print(f"  {eq}_t: {model.active_terms}")
        print(f"\n[Rollout] n_steps={n_steps}")
        run_rollout(result, n_steps)
        return

    # ── Step 1: Load data ────────────────────────────────────────────────────
    print(f"\n[1/4] Loading training data")
    run_dirs = load_run_dirs(n_train, SEED)

    if dry_run:
        print("  [dry_run] Stopping after data listing.")
        return

    # ── Step 2: Build FieldData ──────────────────────────────────────────────
    print(f"\n[2/4] Building FieldData "
          f"(KDE bw={BANDWIDTH}, morse=True, center_flux=True)")
    field_data_list = build_field_data_list(run_dirs)

    T_sub, ny, nx = field_data_list[0].rho.shape
    print(f"  Grid: T_sub={T_sub}, ny={ny}, nx={nx}, dt={DT}")
    has_morse = field_data_list[0].Phi is not None
    print(f"  Morse potential built: {has_morse}")

    # ── Step 3: Build library and ell grid ───────────────────────────────────
    print(f"\n[3/4] Building library "
          f"(morse=True, rich=False, regime_class={REGIME_CLASS!r})")
    library = build_default_library(
        morse=True, rich=False, regime_class=REGIME_CLASS,
    )
    for eq_name, terms in library.items():
        names = [t.name for t in terms]
        print(f"  {eq_name}: {len(terms)} terms: {names}")

    ell_grid = default_ell_grid(T_sub, nx, ny, n_points=n_ell)
    print(f"\n  ell_grid ({len(ell_grid)} configs): {ell_grid}")

    # ── Step 4: Model selection ──────────────────────────────────────────────
    print(f"\n[4/4] Model selection "
          f"({len(ell_grid)} ℓ configs × {len(field_data_list)} trajectories)")
    print("  (3 equations fit independently at each ℓ)\n")

    result, best_ell = model_selection_multifield(
        field_data_list,
        library,
        ell_grid,
        p=P,
        stride=STRIDE,
        lambdas=LAMBDAS,
        verbose=True,
    )

    print(f"\n  ─── Best ℓ = {best_ell} ───\n")
    print(result.summary())

    # ── Diagnostics per equation ─────────────────────────────────────────────
    TARGET_ACCESSORS = {
        "rho": lambda fd: fd.rho,
        "px":  lambda fd: fd.px,
        "py":  lambda fd: fd.py,
    }
    EQ_MODELS = {
        "rho": (result.rho_model, result.rho_terms),
        "px":  (result.px_model,  result.px_terms),
        "py":  (result.py_model,  result.py_terms),
    }

    summary = {
        "experiment":  "K6_v1_rawD19_p5_H37",
        "regime":      "gentle_clustering",
        "physics":     dict(MORSE_PARAMS, eta=0.2, speed=1.5),
        "grid":        {
            "nx": nx, "ny": ny, "T_sub": T_sub,
            "dt": DT, "Lx": LX, "Ly": LY,
        },
        "wsindy": {
            "n_train":  len(field_data_list),
            "subsample": SUBSAMPLE,
            "p":         list(P),
            "n_ell":     n_ell,
            "best_ell":  list(best_ell),
            "n_lambdas": len(LAMBDAS),
        },
        "equations":                    {},
        "post_regression_diagnostics":  {},
    }

    for eq_name, (model, lib_terms) in EQ_MODELS.items():
        print(f"\n{'='*50}")
        print(f"  Equation: {eq_name}_t")
        print(f"{'='*50}")
        print(f"  Discovered: {to_text(model)}")
        r2 = float(model.diagnostics.get("r2", np.nan))
        print(f"  R²_weak = {r2:.6f}   active terms = {model.n_active}")

        summary["equations"][eq_name] = {
            "model_text":   to_text(model),
            "active_terms": model.n_active,
            "r2":           r2,
            "best_lambda":  float(model.best_lambda),
            "coefficients": {
                n: float(model.w[model.col_names.index(n)])
                for n in model.active_terms
            },
            "col_names": model.col_names,
            "xi":        model.w.tolist(),
        }

        # Rebuild b, G at best_ell for this equation
        print(f"\n  Running post-regression diagnostics for '{eq_name}'...")
        try:
            _, eq_b, eq_G, _ = fit_equation_multifield(
                field_data_list,
                lib_terms,
                TARGET_ACCESSORS[eq_name],
                best_ell,
                P,
                STRIDE,
                LAMBDAS,
            )
            summary["post_regression_diagnostics"][eq_name] = run_diagnostics(
                eq_b, eq_G, model, eq_name,
            )
        except Exception as exc:
            print(f"  [WARN] Diagnostics for {eq_name} failed: {exc}")

    # ── Save summary ─────────────────────────────────────────────────────────
    summary_path = OUT_DIR / "discovery_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  Saved: {summary_path}")

    # ── Final discovery summary ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  DISCOVERY COMPLETE")
    print("=" * 65)
    for eq_name, (model, _) in EQ_MODELS.items():
        r2 = model.diagnostics.get("r2", 0)
        print(f"  {eq_name}_t = {to_text(model)}")
        print(f"    R²_weak={r2:.4f}  active={model.n_active}"
              f"  λ*={model.best_lambda:.2e}")
    print(f"\n  Output directory: {OUT_DIR}")

    # ── Rollout ─────────────────────────────────────────────────────────────
    print(f"\n[Rollout] n_steps={n_steps}")
    run_rollout(result, n_steps)

    return result, best_ell


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multifield WSINDy on K6 clustering data."
    )
    parser.add_argument(
        "--n_train", type=int, default=N_TRAIN_DEFAULT,
        help=f"Number of training trajectories (default: {N_TRAIN_DEFAULT})",
    )
    parser.add_argument(
        "--n_ell", type=int, default=N_ELL_DEFAULT,
        help=f"Number of ell configs to sweep (default: {N_ELL_DEFAULT})",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="List available training data without running discovery.",
    )
    parser.add_argument(
        "--n_steps", type=int, default=50,
        help="Number of rollout steps (default: 50)",
    )
    parser.add_argument(
        "--skip_discovery", action="store_true",
        help="Load existing discovery_summary.json and go straight to rollout.",
    )
    args = parser.parse_args()
    main(
        n_train=args.n_train,
        n_ell=args.n_ell,
        dry_run=args.dry_run,
        n_steps=args.n_steps,
        skip_discovery=args.skip_discovery,
    )
