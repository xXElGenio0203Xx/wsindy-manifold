#!/usr/bin/env python3
"""
Alvarez-Style ROM Hyperparameter Selection
===========================================

Replicates the principled lag-selection methodology from:
    Alvarez et al. (2025) — "Reduced-order modelling of particle-based …"

Steps:
  1. Reconstruct latent trajectories (y_trajs) from an experiment's saved
     training densities + POD basis.
  2. ADF stationarity test on each latent mode (Alvarez: p < 0.01).
  3. BIC / AIC lag selection by fitting VAR(w) for w ∈ [1, w_max]
     (Alvarez: w_max = 100; BIC chose w=4, AIC chose w=9).
  4. Print recommended MVAR + LSTM config with selected lags.
  5. Save diagnostic plots (BIC/AIC vs lag, ADF summary).

Usage:
  python -m rom_hyperparameters.alvarez_lag_selection \\
      --experiment_dir oscar_output/DYN1_gentle_v2 \\
      --w_max 50 \\
      --output_dir rom_hyperparameters/results
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Optional heavy imports — fail gracefully
# ---------------------------------------------------------------------------
try:
    from statsmodels.tsa.stattools import adfuller
    HAS_ADF = True
except ImportError:
    HAS_ADF = False

try:
    from statsmodels.tsa.api import VAR
    HAS_VAR = True
except ImportError:
    HAS_VAR = False


# ============================================================================
# 1. LATENT TRAJECTORY RECONSTRUCTION
# ============================================================================

def _load_config(exp_dir: Path) -> dict:
    """Load config_used.yaml from experiment directory."""
    import yaml
    cfg_path = exp_dir / 'config_used.yaml'
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config_used.yaml in {exp_dir}")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _count_train_runs(train_dir: Path) -> int:
    """Count train_XXX/ subdirectories."""
    runs = sorted(train_dir.glob('train_[0-9][0-9][0-9]'))
    return len(runs)


def _reconstruct_from_latent_dataset(
    exp_dir: Path,
    rom_common: Path,
    config: dict,
    max_realizations: int | None = None,
    verbose: bool = True,
) -> tuple[list[np.ndarray], dict]:
    """
    Fallback: approximate trajectory recovery from latent_dataset.npz.

    The windowed dataset stores (X_all, Y_all, lag) where X_all has shape
    (N_samples, lag, d) and Y_all has shape (N_samples, d). We can recover
    approximate trajectories by stitching windows from the same realisation.

    Because trajectory boundaries are not stored, we estimate T_rom from
    config and re-partition the samples.
    """
    ds = np.load(rom_common / 'latent_dataset.npz')
    X_all = ds['X_all']   # (N_samples, lag, d)
    Y_all = ds['Y_all']   # (N_samples, d)
    lag = int(ds['lag'])
    N_samples, _, d = X_all.shape

    rom_cfg = config.get('rom', {})
    subsample = rom_cfg.get('subsample', rom_cfg.get('rom_subsample', 1))
    sim_dt = config.get('sim', {}).get('dt', 0.04)
    sim_T = config.get('sim', {}).get('T', 20.0)
    T_total = int(round(sim_T / sim_dt))
    T_rom = T_total // subsample  # after subsampling

    # Number of samples per realisation = T_rom - lag
    samples_per_traj = T_rom - lag
    if samples_per_traj <= 0:
        raise ValueError(f"T_rom ({T_rom}) <= lag ({lag}), cannot recover trajectories")

    M_total = N_samples // samples_per_traj
    M = min(M_total, max_realizations) if max_realizations else M_total

    if verbose:
        print(f"\n  Experiment: {exp_dir.name}")
        print(f"  Recovering from latent_dataset.npz (lag={lag}, d={d})")
        print(f"  N_samples={N_samples}, estimated T_rom={T_rom}, "
              f"samples_per_traj={samples_per_traj}")
        print(f"  Estimated M={M_total} realizations, using {M}")

    y_trajs = []
    for m in range(M):
        start = m * samples_per_traj
        end = start + samples_per_traj
        if end > N_samples:
            break

        # Recover trajectory: first window gives the initial `lag` steps,
        # then each Y_all gives one more step
        first_window = X_all[start]  # (lag, d)
        targets = Y_all[start:end]   # (samples_per_traj, d)
        traj = np.vstack([first_window, targets])  # (T_rom, d)
        y_trajs.append(traj)

    dt_latent = sim_dt * subsample

    info = {
        'M': len(y_trajs),
        'T_rom': T_rom,
        'R_POD': d,
        'dt_latent': dt_latent,
        'subsample': subsample,
        'density_transform': rom_cfg.get('density_transform', 'raw'),
        'shift_align': rom_cfg.get('shift_align', False),
        'config': config,
        'source': 'latent_dataset.npz',
    }

    if verbose:
        print(f"  ✓ Recovered {len(y_trajs)} trajectories, "
              f"each ({T_rom}, {d}), dt_latent={dt_latent:.4f}s")
        print(f"    NOTE: approximate recovery from windowed data")

    return y_trajs, info


def reconstruct_latent_trajectories(
    exp_dir: Path,
    max_realizations: int | None = None,
    verbose: bool = True,
    use_unaligned: bool = False,
) -> tuple[list[np.ndarray], dict]:
    """
    Reconstruct per-realisation latent trajectories from saved experiment data.

    Tries two approaches in order:
      1. From raw densities: density → (subsample) → (shift_align) →
         (density_transform) → center → project
      2. From latent_dataset.npz: approximate recovery by stitching windows

    Parameters
    ----------
    exp_dir : Path
        Experiment output directory (e.g. oscar_output/DYN1_gentle_v2)
    max_realizations : int or None
        Cap number of training runs loaded (for speed). None = all.
    verbose : bool
        Print progress.
    use_unaligned : bool
        If True, skip shift alignment and compute a fresh POD from unaligned
        data. Used for aligned-vs-unaligned stationarity comparison.

    Returns
    -------
    y_trajs : list[np.ndarray]
        Each element shape (T_rom, R_POD).
    info : dict
        Metadata: M, T_rom, R_POD, dt_latent, config, ...
    """
    exp_dir = Path(exp_dir)
    config = _load_config(exp_dir)
    rom_cfg = config.get('rom', {})

    rom_common = exp_dir / 'rom_common'
    if not rom_common.exists():
        raise FileNotFoundError(f"No rom_common/ directory in {exp_dir}")

    # --- Count available density files ---
    train_dir = exp_dir / 'train'
    n_density_files = 0
    n_train_dirs = 0
    if train_dir.exists():
        train_runs = sorted(train_dir.glob('train_[0-9][0-9][0-9]'))
        n_train_dirs = len(train_runs)
        n_density_files = sum(
            1 for r in train_runs if (r / 'density.npz').exists()
        )

    # --- Decide reconstruction strategy ---
    latent_ds_path = rom_common / 'latent_dataset.npz'
    has_latent_ds = latent_ds_path.exists()

    min_needed = min(max_realizations or 999999, 3)

    if n_density_files >= min_needed:
        # Strategy A: from raw densities (exact reconstruction)
        if verbose:
            mode_str = "UNALIGNED" if use_unaligned else "aligned"
            print(f"  Strategy: reconstruct from raw densities [{mode_str}] "
                  f"({n_density_files}/{n_train_dirs} available)")
        return _reconstruct_from_densities(
            exp_dir, train_dir, rom_common, config,
            n_density_available=n_density_files,
            max_realizations=max_realizations,
            verbose=verbose,
            use_unaligned=use_unaligned,
        )
    elif has_latent_ds:
        # Strategy B: approximate recovery from windowed dataset
        if verbose:
            print(f"  Strategy: recover from latent_dataset.npz "
                  f"(only {n_density_files} density files available)")
        return _reconstruct_from_latent_dataset(
            exp_dir, rom_common, config,
            max_realizations=max_realizations,
            verbose=verbose,
        )
    else:
        raise FileNotFoundError(
            f"Cannot reconstruct latent trajectories for {exp_dir.name}:\n"
            f"  - Only {n_density_files} density files in train/\n"
            f"  - No latent_dataset.npz in rom_common/\n"
            f"Sync more data from Oscar or run the experiment first."
        )


def _reconstruct_from_densities(
    exp_dir: Path,
    train_dir: Path,
    rom_common: Path,
    config: dict,
    n_density_available: int,
    max_realizations: int | None = None,
    verbose: bool = True,
    use_unaligned: bool = False,
) -> tuple[list[np.ndarray], dict]:
    """Reconstruct from raw density files (exact).

    If use_unaligned=True, skip shift alignment and compute a fresh POD
    basis from the unaligned (raw) data.  This lets us compare stationarity
    with vs. without alignment.
    """
    rom_cfg = config.get('rom', {})

    # --- Config params ---
    subsample = rom_cfg.get('subsample', rom_cfg.get('rom_subsample', 1))
    density_transform = rom_cfg.get('density_transform', 'raw')
    density_transform_eps = rom_cfg.get('density_transform_eps', 1e-8)
    density_key = 'rho'

    # --- Decide alignment mode ---
    config_wants_shift = rom_cfg.get('shift_align', False)
    do_shift = config_wants_shift and (not use_unaligned)

    # --- Load POD basis (only when using aligned path) ---
    U_r = None
    X_mean = None
    R_POD = None
    if not use_unaligned:
        pod = np.load(rom_common / 'pod_basis.npz')
        U_r = pod['U']              # (N_spatial, R_POD)
        R_POD = U_r.shape[1]
        X_mean = np.load(rom_common / 'X_train_mean.npy')  # (N_spatial,)

    # --- Load shift alignment data if applicable ---
    sa_shifts = None
    if do_shift:
        sa_path = rom_common / 'shift_align.npz'
        if sa_path.exists():
            sa = np.load(sa_path)
            sa_shifts = sa['shifts']  # (M*T_rom, 2)
            if verbose:
                print(f"  Loaded shift alignment: {sa_shifts.shape[0]} frames")
        else:
            print(f"  WARNING: shift_align=True but no shift_align.npz found. "
                  "Proceeding without alignment (results may differ).")
            do_shift = False

    M_target = min(n_density_available, max_realizations) if max_realizations else n_density_available
    if verbose:
        mode_str = "UNALIGNED (fresh POD)" if use_unaligned else "aligned"
        print(f"\n  Experiment: {exp_dir.name}")
        print(f"  Mode: {mode_str}")
        print(f"  Training runs to load: {M_target} (of {n_density_available} with density files)")
        print(f"  Subsample: {subsample}, transform: {density_transform}")
        print(f"  Shift-align: {do_shift}")
        if R_POD is not None:
            print(f"  POD modes (R_POD): {R_POD}")

    # --- Import shift alignment ---
    apply_shifts_fn = None
    if do_shift:
        try:
            from rectsim.shift_align import apply_shifts
            apply_shifts_fn = apply_shifts
        except ImportError:
            print(f"  WARNING: cannot import rectsim.shift_align, skipping alignment")
            do_shift = False

    # --- Load, align, transform each available run ---
    all_X_flat = []   # Collect for SVD when use_unaligned
    per_run_X = []    # Keep per-run data for per-trajectory projection
    y_trajs = []
    T_rom = None
    loaded = 0

    # Find runs that actually have density.npz
    train_runs = sorted(train_dir.glob('train_[0-9][0-9][0-9]'))
    for run_dir in train_runs:
        if loaded >= M_target:
            break
        density_path = run_dir / 'density.npz'
        if not density_path.exists():
            continue

        # Extract run index from directory name for shift alignment
        run_idx = int(run_dir.name.split('_')[1])

        data = np.load(density_path)
        density = data[density_key]  # (T, Ny, Nx)

        # Subsample in time
        if subsample > 1:
            density = density[::subsample]

        T_sub = density.shape[0]
        if T_rom is None:
            T_rom = T_sub

        # Shift alignment (apply pre-computed shifts for this run's frames)
        if do_shift and sa_shifts is not None and apply_shifts_fn is not None:
            frame_start = run_idx * T_rom
            frame_end = frame_start + T_rom
            if frame_end <= sa_shifts.shape[0]:
                run_shifts = sa_shifts[frame_start:frame_end]
                density = apply_shifts_fn(density, run_shifts)
            else:
                if verbose:
                    print(f"  WARNING: shift data too short for run {run_idx}, skipping alignment")

        # Flatten to (T_rom, N_spatial)
        X_run = density.reshape(T_sub, -1)

        # Density transform
        if density_transform == 'log':
            X_run = np.log(X_run + density_transform_eps)
        elif density_transform == 'sqrt':
            X_run = np.sqrt(X_run + density_transform_eps)
        elif density_transform == 'meansub':
            X_run = X_run - X_run.mean(axis=1, keepdims=True)

        if use_unaligned:
            # Defer projection — collect flat data for joint SVD
            per_run_X.append(X_run)  # (T_rom, N_spatial)
            all_X_flat.append(X_run)
        else:
            # Project immediately with saved aligned basis
            X_centered = X_run - X_mean[np.newaxis, :]
            Y_latent = X_centered @ U_r  # (T_rom, R_POD)
            y_trajs.append(Y_latent)

        loaded += 1

    if loaded == 0:
        raise RuntimeError("No density files could be loaded")

    # --- If unaligned: compute fresh POD and project ---
    if use_unaligned:
        if verbose:
            print(f"  Computing fresh POD from unaligned data ({loaded} runs)...")
        X_all = np.vstack(all_X_flat)  # (M*T_rom, N_spatial)
        X_mean = X_all.mean(axis=0)
        X_centered_all = X_all - X_mean[np.newaxis, :]

        # Aligned basis has R_POD modes — use same count for fair comparison
        # Load aligned basis to get R_POD
        aligned_pod = np.load(rom_common / 'pod_basis.npz')
        R_POD = aligned_pod['U'].shape[1]

        # Truncated SVD: X^T = U S Vt where U columns are spatial modes
        U_full, S, _Vt = np.linalg.svd(X_centered_all.T, full_matrices=False)
        U_r = U_full[:, :R_POD]  # (N_spatial, R_POD)

        if verbose:
            total_energy = np.sum(S**2)
            captured = np.sum(S[:R_POD]**2) / total_energy
            print(f"  ✓ Unaligned POD: {R_POD} modes capture {captured:.4f} energy")

        # Project each run
        for X_run in per_run_X:
            X_centered = X_run - X_mean[np.newaxis, :]
            Y_latent = X_centered @ U_r  # (T_rom, R_POD)
            y_trajs.append(Y_latent)

        del X_all, X_centered_all, all_X_flat, per_run_X

    if not y_trajs:
        raise RuntimeError("No density files could be loaded")

    # Compute dt in latent space
    sim_dt = config.get('sim', {}).get('dt', 0.04)
    dt_latent = sim_dt * subsample

    info = {
        'M': len(y_trajs),
        'T_rom': T_rom,
        'R_POD': R_POD,
        'dt_latent': dt_latent,
        'subsample': subsample,
        'density_transform': density_transform,
        'shift_align': do_shift,
        'alignment_mode': 'unaligned' if use_unaligned else ('aligned' if config_wants_shift else 'none'),
        'config': config,
        'source': 'raw_densities',
    }

    if verbose:
        print(f"  ✓ Reconstructed {len(y_trajs)} trajectories, "
              f"each ({T_rom}, {R_POD}), dt_latent={dt_latent:.4f}s")

    return y_trajs, info


# ============================================================================
# 2. ADF STATIONARITY TEST
# ============================================================================

def run_adf_tests(
    y_trajs: list[np.ndarray],
    alpha: float = 0.01,
    max_realizations: int = 20,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Augmented Dickey-Fuller stationarity test per latent mode.

    Following Alvarez et al.: test each POD coefficient time series for
    a unit root. Significance threshold p < alpha (default 0.01).

    We concatenate a subset of realisations and test each mode's series.

    Parameters
    ----------
    y_trajs : list[np.ndarray]
        Per-realisation latent trajectories, each (T_rom, d).
    alpha : float
        Significance level for rejecting H0 (unit root).
    max_realizations : int
        Max realisations to use (concatenated). Too many → slow ADF.
    verbose : bool
        Print results table.

    Returns
    -------
    pd.DataFrame
        Columns: mode, adf_stat, p_value, critical_1pct, critical_5pct,
                 n_lags_used, n_obs, stationary
    """
    if not HAS_ADF:
        raise ImportError(
            "statsmodels is required for ADF tests. "
            "Install with: pip install statsmodels"
        )

    d = y_trajs[0].shape[1]
    M_use = min(len(y_trajs), max_realizations)

    # Concatenate selected realisations per mode
    concat = np.concatenate([y_trajs[i] for i in range(M_use)], axis=0)
    # concat shape: (M_use * T_rom, d)

    rows = []
    for j in range(d):
        series = concat[:, j]
        result = adfuller(series, autolag='AIC')
        adf_stat, p_val, lags_used, nobs = result[0], result[1], result[2], result[3]
        crit = result[4]  # dict: {'1%': ..., '5%': ..., '10%': ...}

        rows.append({
            'mode': j,
            'adf_stat': adf_stat,
            'p_value': p_val,
            'critical_1pct': crit['1%'],
            'critical_5pct': crit['5%'],
            'n_lags_used': lags_used,
            'n_obs': nobs,
            'stationary': p_val < alpha,
        })

    df = pd.DataFrame(rows)

    if verbose:
        n_pass = df['stationary'].sum()
        print(f"\n{'=' * 65}")
        print(f"ADF STATIONARITY TEST  (α = {alpha}, {M_use} realisations concatenated)")
        print(f"{'=' * 65}")
        print(f"  {'Mode':>4s}  {'ADF stat':>10s}  {'p-value':>10s}  "
              f"{'1% crit':>10s}  {'Stationary':>10s}")
        print(f"  {'─' * 4}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 10}")
        for _, r in df.iterrows():
            tag = '  ✓' if r['stationary'] else '  ✗'
            print(f"  {int(r['mode']):>4d}  {r['adf_stat']:>10.3f}  "
                  f"{r['p_value']:>10.4e}  {r['critical_1pct']:>10.3f}  {tag:>10s}")
        print(f"\n  Result: {n_pass}/{d} modes are stationary at p < {alpha}")
        if n_pass < d:
            non_stat = df[~df['stationary']]['mode'].tolist()
            print(f"  ⚠  Non-stationary modes: {non_stat}")
            print(f"     (Alvarez requires all modes stationary before VAR fitting)")
        else:
            print(f"  ✓ All modes stationary — safe to proceed with VAR lag selection")

    return df


# ============================================================================
# 3. BIC / AIC LAG SELECTION (VAR)
# ============================================================================

def select_lag_bic_aic(
    y_trajs: list[np.ndarray],
    w_max: int = 100,
    max_realizations: int = 10,
    verbose: bool = True,
) -> dict:
    """
    Select MVAR lag order using BIC and AIC on a VAR(w) model.

    Following Alvarez et al. Appendix D: fit VAR(w) for w ∈ [1, w_max],
    report the lag that minimises BIC and AIC respectively.

    Parameters
    ----------
    y_trajs : list[np.ndarray]
        Per-realisation latent trajectories, each (T_rom, d).
    w_max : int
        Maximum lag to consider (Alvarez: 100).
    max_realizations : int
        Number of realisations to concatenate (more → more data,
        but with d=19 and large w VAR fitting gets expensive).
    verbose : bool
        Print results.

    Returns
    -------
    dict with keys:
        lag_bic : int — lag selected by BIC
        lag_aic : int — lag selected by AIC
        lag_hqic : int — lag selected by HQIC
        lag_fpe : int — lag selected by FPE
        ic_table : pd.DataFrame — all IC values per lag
        var_result : statsmodels SelectOrderResult
    """
    if not HAS_VAR:
        raise ImportError(
            "statsmodels is required for VAR lag selection. "
            "Install with: pip install statsmodels"
        )

    d = y_trajs[0].shape[1]
    T_rom = y_trajs[0].shape[0]
    M_use = min(len(y_trajs), max_realizations)

    # Cap w_max at what the data can support
    # VAR(w) needs at least w + d*w + 1 observations per equation
    max_feasible = (T_rom * M_use - 1) // (d + 1) - 1
    # More conservatively: statsmodels needs nobs > w + d*w
    # But practically, T_rom * M_use >> w_max for our data
    if w_max > T_rom - 2:
        old_max = w_max
        w_max = T_rom - 2
        if verbose:
            print(f"  ⚠ Capping w_max from {old_max} to {w_max} "
                  f"(T_rom={T_rom})")

    # Concatenate realisations into one long multivariate time series.
    # NOTE: Concatenation introduces artificial jumps at boundaries.
    # This is acceptable because:
    #  (a) Alvarez concatenated multiple realisations the same way
    #  (b) With T_rom >> w_max, boundary effects are negligible
    #  (c) We only use this for IC computation, not forecasting
    concat = np.concatenate([y_trajs[i] for i in range(M_use)], axis=0)
    # shape: (M_use * T_rom, d)

    if verbose:
        print(f"\n{'=' * 65}")
        print(f"VAR LAG SELECTION (BIC / AIC)")
        print(f"{'=' * 65}")
        print(f"  Latent dim d = {d}")
        print(f"  Realisations used: {M_use}")
        print(f"  Total observations: {concat.shape[0]}")
        print(f"  w_max = {w_max}")
        print(f"  Fitting VAR(1) … VAR({w_max})  — this may take a moment …")

    t0 = time.time()

    # Fit VAR and compute information criteria
    model = VAR(concat)
    try:
        result = model.select_order(maxlags=w_max)
    except Exception as e:
        print(f"\n  ERROR in VAR.select_order: {e}")
        print(f"  Try reducing --w_max or --max_realizations")
        raise

    elapsed = time.time() - t0

    # Extract selected lags
    lag_bic = result.bic
    lag_aic = result.aic
    lag_hqic = result.hqic
    lag_fpe = result.fpe

    # Build IC table from the summary
    # result.ics is a dict: {'aic': OrderedDict, 'bic': ...}
    ic_data = []
    for w in range(1, w_max + 1):
        row = {'lag': w}
        for ic_name in ['aic', 'bic', 'hqic', 'fpe']:
            ics_dict = getattr(result, f'{ic_name}s', None)
            if ics_dict is not None and w in ics_dict:
                row[ic_name] = ics_dict[w]
            else:
                # fallback: try the summary table
                row[ic_name] = np.nan
        ic_data.append(row)
    ic_table = pd.DataFrame(ic_data)

    if verbose:
        print(f"\n  ✓ VAR lag selection complete in {elapsed:.1f}s")
        print(f"\n  {'Criterion':>10s}  {'Selected lag':>12s}")
        print(f"  {'─' * 10}  {'─' * 12}")
        print(f"  {'BIC':>10s}  {lag_bic:>12d}")
        print(f"  {'AIC':>10s}  {lag_aic:>12d}")
        print(f"  {'HQIC':>10s}  {lag_hqic:>12d}")
        print(f"  {'FPE':>10s}  {lag_fpe:>12d}")

        print(f"\n  → Alvarez used BIC → w={lag_bic}, AIC → w={lag_aic}")
        print(f"     (Alvarez 2025 reported BIC=4, AIC=9 for their dataset)")

    return {
        'lag_bic': lag_bic,
        'lag_aic': lag_aic,
        'lag_hqic': lag_hqic,
        'lag_fpe': lag_fpe,
        'ic_table': ic_table,
        'var_result': result,
        'elapsed': elapsed,
    }


# ============================================================================
# 4. DIAGNOSTIC PLOTS
# ============================================================================

def plot_bic_aic_vs_lag(ic_table: pd.DataFrame, lag_bic: int, lag_aic: int,
                        output_dir: Path, exp_name: str = ''):
    """BIC and AIC as function of lag, with selected lag markers."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, ic_name, sel_lag in zip(
        axes, ['bic', 'aic'], [lag_bic, lag_aic]
    ):
        vals = ic_table[ic_name].values
        lags = ic_table['lag'].values
        valid = ~np.isnan(vals)

        ax.plot(lags[valid], vals[valid], '-', lw=1.5, color='#1f77b4')
        if sel_lag in lags:
            idx = np.where(lags == sel_lag)[0][0]
            if valid[idx]:
                ax.axvline(sel_lag, color='red', ls='--', lw=1, alpha=0.7)
                ax.scatter([sel_lag], [vals[idx]], color='red', s=80, zorder=5,
                           marker='*', label=f'Selected w={sel_lag}')
                ax.legend(fontsize=10)

        ax.set_xlabel('Lag order (w)', fontsize=11)
        ax.set_ylabel(ic_name.upper(), fontsize=11)
        ax.set_title(f'{ic_name.upper()} vs Lag Order', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

    title = 'Alvarez-Style Lag Selection: BIC & AIC'
    if exp_name:
        title += f'\n({exp_name})'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'bic_aic_vs_lag.{ext}',
                    bbox_inches='tight', dpi=200 if ext == 'png' else None)
    plt.close(fig)
    print(f"  Saved: bic_aic_vs_lag.pdf/png")


def plot_adf_summary(adf_df: pd.DataFrame, alpha: float,
                     output_dir: Path, exp_name: str = ''):
    """Bar chart of ADF test statistics per mode, with critical values."""
    fig, ax = plt.subplots(figsize=(max(8, adf_df.shape[0] * 0.5), 5))

    modes = adf_df['mode'].values
    stats = adf_df['adf_stat'].values
    crit1 = adf_df['critical_1pct'].values
    stationary = adf_df['stationary'].values

    colors = ['#2ca02c' if s else '#d62728' for s in stationary]
    bars = ax.bar(modes, stats, color=colors, edgecolor='white', linewidth=0.5)

    # Critical value line (1%)
    ax.axhline(crit1[0], color='black', ls='--', lw=1.2, alpha=0.7,
               label=f'1% critical value ({crit1[0]:.2f})')

    ax.set_xlabel('Latent Mode', fontsize=11)
    ax.set_ylabel('ADF Test Statistic', fontsize=11)
    ax.set_xticks(modes)

    title = f'ADF Stationarity Test (α = {alpha})'
    if exp_name:
        title += f'\n({exp_name})'
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Add pass/fail annotations
    for m, s, stat in zip(modes, stationary, stats):
        label = '✓' if s else '✗'
        ax.annotate(label, (m, stat), textcoords="offset points",
                    xytext=(0, -12 if stat < 0 else 8), ha='center',
                    fontsize=8, fontweight='bold',
                    color='green' if s else 'red')

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'adf_stationarity.{ext}',
                    bbox_inches='tight', dpi=200 if ext == 'png' else None)
    plt.close(fig)
    print(f"  Saved: adf_stationarity.pdf/png")


def plot_latent_timeseries_sample(
    y_trajs: list[np.ndarray],
    dt_latent: float,
    output_dir: Path,
    n_modes: int = 6,
    n_trajs: int = 5,
    exp_name: str = '',
):
    """Plot a few latent modes over time for visual stationarity check."""
    d = y_trajs[0].shape[1]
    n_modes = min(n_modes, d)
    n_trajs = min(n_trajs, len(y_trajs))
    T_rom = y_trajs[0].shape[0]
    times = np.arange(T_rom) * dt_latent

    fig, axes = plt.subplots(n_modes, 1, figsize=(12, 2.5 * n_modes),
                              sharex=True)
    if n_modes == 1:
        axes = [axes]

    for j, ax in enumerate(axes):
        for i in range(n_trajs):
            ax.plot(times, y_trajs[i][:, j], alpha=0.6, lw=0.8)
        ax.set_ylabel(f'Mode {j}', fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)', fontsize=11)
    title = 'Latent Mode Time Series (sample realisations)'
    if exp_name:
        title += f'\n({exp_name})'
    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'latent_timeseries_sample.{ext}',
                    bbox_inches='tight', dpi=200 if ext == 'png' else None)
    plt.close(fig)
    print(f"  Saved: latent_timeseries_sample.pdf/png")


# ============================================================================
# 5. RECOMMENDED CONFIG OUTPUT
# ============================================================================

def print_recommended_config(lag_bic: int, lag_aic: int, current_config: dict):
    """Print YAML-style recommended configuration."""
    rom_cfg = current_config.get('rom', {})
    models = rom_cfg.get('models', {})
    current_mvar_lag = models.get('mvar', {}).get('lag', '?')
    current_lstm_lag = models.get('lstm', {}).get('lag', '?')
    ridge_alpha = models.get('mvar', {}).get('ridge_alpha', 1e-6)

    print(f"\n{'=' * 65}")
    print("RECOMMENDED CONFIGURATION (Alvarez-style)")
    print(f"{'=' * 65}")

    print(f"\n  Current config:  MVAR lag = {current_mvar_lag},  LSTM lag = {current_lstm_lag}")
    print(f"  BIC selection:   w = {lag_bic}")
    print(f"  AIC selection:   w = {lag_aic}")

    print(f"\n  # --- Option A: Conservative (BIC, fewer parameters) ---")
    print(f"  rom:")
    print(f"    models:")
    print(f"      mvar:")
    print(f"        lag: {lag_bic}")
    print(f"        ridge_alpha: {ridge_alpha}")
    print(f"      lstm:")
    print(f"        lag: {lag_bic}")

    print(f"\n  # --- Option B: Flexible (AIC, more temporal context) ---")
    print(f"  rom:")
    print(f"    models:")
    print(f"      mvar:")
    print(f"        lag: {lag_aic}")
    print(f"        ridge_alpha: {ridge_alpha}")
    print(f"      lstm:")
    print(f"        lag: {lag_aic}")

    print(f"\n  # --- Option C: Alvarez-style (both, 4 models) ---")
    print(f"  # Train MVAR({lag_bic}), MVAR({lag_aic}), LSTM({lag_bic}), LSTM({lag_aic})")

    print(f"\n  Alvarez note: LSTM uses the SAME lag as MVAR for comparison.")
    print(f"  The lag is NOT separately optimized for LSTM.")
    print(f"{'=' * 65}")


# ============================================================================
# 6. MAIN ORCHESTRATOR
# ============================================================================

def run_alvarez_selection(
    experiment_dir: str | Path,
    w_max: int = 100,
    adf_alpha: float = 0.01,
    max_realizations_latent: int | None = None,
    max_realizations_adf: int = 20,
    max_realizations_var: int = 10,
    output_dir: str | Path = 'rom_hyperparameters/results',
    verbose: bool = True,
    use_unaligned: bool = False,
):
    """
    Run the full Alvarez-style hyperparameter selection pipeline.

    Parameters
    ----------
    experiment_dir : path-like
        Experiment output directory with train/ and rom_common/.
    w_max : int
        Maximum lag to search (Alvarez: 100).
    adf_alpha : float
        ADF test significance level (Alvarez: 0.01).
    max_realizations_latent : int or None
        Cap on training runs to load for latent reconstruction.
    max_realizations_adf : int
        Cap on realisations for ADF test.
    max_realizations_var : int
        Cap on realisations for VAR fitting (d=19, w_max=100 is expensive).
    output_dir : path-like
        Where to save plots and tables.
    verbose : bool
        Print progress.
    """
    experiment_dir = Path(experiment_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exp_name = experiment_dir.name

    t0_total = time.time()

    align_label = "UNALIGNED" if use_unaligned else "aligned"
    print("\n" + "=" * 70)
    print(f"ALVAREZ-STYLE ROM HYPERPARAMETER SELECTION [{align_label}]")
    print(f"  Experiment:  {exp_name}")
    print(f"  w_max:       {w_max}")
    print(f"  ADF α:       {adf_alpha}")
    print(f"  Alignment:   {align_label}")
    print("=" * 70)

    # ── Step 1: Reconstruct latent trajectories ─────────────────────────
    print(f"\n--- Step 1: Reconstruct Latent Trajectories ---")
    y_trajs, info = reconstruct_latent_trajectories(
        experiment_dir,
        max_realizations=max_realizations_latent,
        verbose=verbose,
        use_unaligned=use_unaligned,
    )

    # ── Step 1b: Visual check — sample time series ──────────────────────
    plot_latent_timeseries_sample(
        y_trajs, info['dt_latent'], output_dir,
        exp_name=exp_name,
    )

    # ── Step 2: ADF stationarity test ───────────────────────────────────
    print(f"\n--- Step 2: ADF Stationarity Test ---")
    adf_df = run_adf_tests(
        y_trajs, alpha=adf_alpha,
        max_realizations=max_realizations_adf,
        verbose=verbose,
    )
    plot_adf_summary(adf_df, adf_alpha, output_dir, exp_name=exp_name)

    # Save ADF table
    adf_df.to_csv(output_dir / 'adf_results.csv', index=False, float_format='%.6f')
    print(f"  Saved: adf_results.csv")

    # ── Step 3: BIC / AIC lag selection ─────────────────────────────────
    print(f"\n--- Step 3: BIC / AIC Lag Selection ---")
    lag_result = select_lag_bic_aic(
        y_trajs, w_max=w_max,
        max_realizations=max_realizations_var,
        verbose=verbose,
    )

    plot_bic_aic_vs_lag(
        lag_result['ic_table'],
        lag_result['lag_bic'], lag_result['lag_aic'],
        output_dir, exp_name=exp_name,
    )

    # Save IC table
    lag_result['ic_table'].to_csv(
        output_dir / 'ic_values.csv', index=False, float_format='%.4f'
    )
    print(f"  Saved: ic_values.csv")

    # ── Step 4: Print recommended config ────────────────────────────────
    print_recommended_config(
        lag_result['lag_bic'], lag_result['lag_aic'], info['config']
    )

    # ── Summary ─────────────────────────────────────────────────────────
    elapsed_total = time.time() - t0_total
    summary = {
        'experiment': exp_name,
        'alignment': info.get('alignment_mode', 'aligned'),
        'M_used_latent': int(info['M']),
        'T_rom': int(info['T_rom']),
        'R_POD': int(info['R_POD']),
        'dt_latent': float(info['dt_latent']),
        'adf_alpha': float(adf_alpha),
        'n_stationary': int(adf_df['stationary'].sum()),
        'n_modes': int(adf_df.shape[0]),
        'all_stationary': bool(adf_df['stationary'].all()),
        'w_max': int(w_max),
        'lag_bic': int(lag_result['lag_bic']),
        'lag_aic': int(lag_result['lag_aic']),
        'lag_hqic': int(lag_result['lag_hqic']),
        'lag_fpe': int(lag_result['lag_fpe']),
        'var_elapsed_s': float(lag_result['elapsed']),
        'total_elapsed_s': float(elapsed_total),
    }

    # Save summary JSON
    import json
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: summary.json")

    print(f"\n  Total elapsed: {elapsed_total:.1f}s")
    print("=" * 70)

    return summary


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Alvarez-style lag selection via BIC/AIC on VAR model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on a single experiment
  python -m rom_hyperparameters.alvarez_lag_selection \\
      --experiment_dir oscar_output/DYN1_gentle_v2

  # Higher lag bound, fewer realisations for speed
  python -m rom_hyperparameters.alvarez_lag_selection \\
      --experiment_dir oscar_output/DO_CS01_swarm_C01_l05 \\
      --w_max 50 --max_realizations_var 5

  # Run on multiple experiments
  python -m rom_hyperparameters.alvarez_lag_selection \\
      --experiment_dir oscar_output/DYN1_gentle_v2 \\
                       oscar_output/DYN2_hypervelocity_v2
        """
    )
    parser.add_argument('--experiment_dir', nargs='+', required=True,
                        help='Experiment directory(ies) with train/ and rom_common/')
    parser.add_argument('--w_max', type=int, default=100,
                        help='Maximum lag to search (Alvarez: 100)')
    parser.add_argument('--adf_alpha', type=float, default=0.01,
                        help='ADF significance level (Alvarez: 0.01)')
    parser.add_argument('--max_realizations', type=int, default=None,
                        help='Cap on training runs to load for latent reconstruction')
    parser.add_argument('--max_realizations_adf', type=int, default=20,
                        help='Cap on realisations for ADF test')
    parser.add_argument('--max_realizations_var', type=int, default=10,
                        help='Cap on realisations for VAR fitting')
    parser.add_argument('--output_dir', type=str,
                        default='rom_hyperparameters/results',
                        help='Output directory for plots and tables')
    parser.add_argument('--unaligned', action='store_true',
                        help='Skip shift alignment and compute fresh POD from '
                             'unaligned data (for alignment comparison)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress detailed output')
    args = parser.parse_args()

    # Check dependencies
    missing = []
    if not HAS_ADF:
        missing.append('statsmodels (for adfuller)')
    if not HAS_VAR:
        missing.append('statsmodels (for VAR)')
    if missing:
        print(f"ERROR: Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install statsmodels")
        sys.exit(1)

    # Add src/ to path so rectsim is importable
    src_dir = Path(__file__).resolve().parent.parent / 'src'
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # Run for each experiment
    all_summaries = []
    for exp_path in args.experiment_dir:
        exp_dir = Path(exp_path)
        if not exp_dir.exists():
            print(f"\n  ⚠ Skipping {exp_path}: directory not found")
            continue

        # Per-experiment output subdirectory
        if len(args.experiment_dir) > 1:
            out = Path(args.output_dir) / exp_dir.name
        else:
            out = Path(args.output_dir)

        summary = run_alvarez_selection(
            experiment_dir=exp_dir,
            w_max=args.w_max,
            adf_alpha=args.adf_alpha,
            max_realizations_latent=args.max_realizations,
            max_realizations_adf=args.max_realizations_adf,
            max_realizations_var=args.max_realizations_var,
            output_dir=out,
            verbose=not args.quiet,
            use_unaligned=args.unaligned,
        )
        all_summaries.append(summary)

    # Print comparison table if multiple experiments
    if len(all_summaries) > 1:
        print(f"\n\n{'=' * 70}")
        print("CROSS-EXPERIMENT LAG COMPARISON")
        print(f"{'=' * 70}")
        align_col = all_summaries[0].get('alignment', 'aligned')
        print(f"  Alignment mode: {align_col}")
        print(f"  {'Experiment':>35s}  {'BIC':>5s}  {'AIC':>5s}  "
              f"{'HQIC':>5s}  {'Stationary':>11s}")
        print(f"  {'─' * 35}  {'─' * 5}  {'─' * 5}  {'─' * 5}  {'─' * 11}")
        for s in all_summaries:
            stat = f"{s['n_stationary']}/{s['n_modes']}"
            print(f"  {s['experiment']:>35s}  {s['lag_bic']:>5d}  "
                  f"{s['lag_aic']:>5d}  {s['lag_hqic']:>5d}  {stat:>11s}")
        print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
