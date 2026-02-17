"""
Shift Alignment Module
======================

Computes and applies translational alignment of 2D density fields
on periodic domains using FFT cross-correlation.

The alignment removes phase drift (translational motion of clusters)
so that POD/MVAR can focus on structural changes rather than spatial transport.

Usage:
    In config YAML:
        rom:
          shift_align: true       # Enable shift-aligned POD
          shift_align_ref: "mean" # Reference: "mean" (default), "first", or "median"

    The pipeline will:
    1. Compute per-frame shifts relative to reference density
    2. Roll all density fields to align
    3. Build POD on aligned fields
    4. At test time: align test density → forecast in aligned space → un-align predictions
"""

import numpy as np
from typing import Tuple, Optional


def fft_cross_correlation_shift(ref: np.ndarray, target: np.ndarray) -> Tuple[int, int]:
    """
    Find the integer pixel shift (dy, dx) that best aligns target to ref
    via FFT cross-correlation on a periodic domain.

    Maximizes: sum_xy ref(x,y) * target(x+dx, y+dy)

    Parameters
    ----------
    ref : np.ndarray, shape (Ny, Nx)
        Reference density field
    target : np.ndarray, shape (Ny, Nx)
        Target density field to align

    Returns
    -------
    dy, dx : int, int
        Shift to apply: aligned = np.roll(np.roll(target, dy, axis=0), dx, axis=1)
    """
    Ny, Nx = ref.shape

    # Cross-correlation via FFT (periodic)
    F_ref = np.fft.fft2(ref)
    F_target = np.fft.fft2(target)
    cross = np.real(np.fft.ifft2(F_ref * np.conj(F_target)))

    # Find peak
    idx = np.unravel_index(np.argmax(cross), cross.shape)

    # Convert to signed shifts (wrap around)
    dy = idx[0] if idx[0] <= Ny // 2 else idx[0] - Ny
    dx = idx[1] if idx[1] <= Nx // 2 else idx[1] - Nx

    return int(dy), int(dx)


def compute_reference_field(densities: np.ndarray, method: str = "mean") -> np.ndarray:
    """
    Compute reference density field for alignment.

    Parameters
    ----------
    densities : np.ndarray, shape (T, Ny, Nx)
        Stack of density fields
    method : str
        "mean" - temporal mean (default, stable)
        "first" - first frame
        "median" - per-pixel temporal median

    Returns
    -------
    ref : np.ndarray, shape (Ny, Nx)
    """
    if method == "mean":
        return densities.mean(axis=0)
    elif method == "first":
        return densities[0].copy()
    elif method == "median":
        return np.median(densities, axis=0)
    else:
        raise ValueError(f"Unknown shift_align_ref: '{method}'. Use 'mean', 'first', or 'median'.")


def compute_shifts(densities: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Compute per-frame alignment shifts relative to reference.

    Parameters
    ----------
    densities : np.ndarray, shape (T, Ny, Nx)
    ref : np.ndarray, shape (Ny, Nx)

    Returns
    -------
    shifts : np.ndarray, shape (T, 2)
        shifts[t] = (dy, dx) to align frame t to ref
    """
    T = densities.shape[0]
    shifts = np.zeros((T, 2), dtype=np.int32)

    for t in range(T):
        dy, dx = fft_cross_correlation_shift(ref, densities[t])
        shifts[t] = [dy, dx]

    return shifts


def apply_shifts(densities: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """
    Apply alignment shifts to density fields (periodic roll).

    Parameters
    ----------
    densities : np.ndarray, shape (T, Ny, Nx)
    shifts : np.ndarray, shape (T, 2)  — (dy, dx) per frame

    Returns
    -------
    aligned : np.ndarray, shape (T, Ny, Nx)
    """
    T = densities.shape[0]
    aligned = np.empty_like(densities)

    for t in range(T):
        dy, dx = shifts[t]
        aligned[t] = np.roll(np.roll(densities[t], int(dy), axis=0), int(dx), axis=1)

    return aligned


def undo_shifts(densities: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """
    Reverse alignment shifts (apply negative shifts).

    Parameters
    ----------
    densities : np.ndarray, shape (T, Ny, Nx) — in aligned space
    shifts : np.ndarray, shape (T, 2) — (dy, dx) that were used to align

    Returns
    -------
    unaligned : np.ndarray, shape (T, Ny, Nx) — back in original space
    """
    return apply_shifts(densities, -shifts)


def align_training_data(all_densities: np.ndarray, M: int, T_rom: int,
                        ref_method: str = "mean") -> dict:
    """
    Align all training density data for POD.

    Takes the stacked (M*T_rom, Ny, Nx) array, computes a global reference,
    aligns all frames, and returns aligned data + metadata.

    Parameters
    ----------
    all_densities : np.ndarray, shape (M*T_rom, Ny, Nx)
        All training density fields (M runs × T_rom frames each)
    M : int
        Number of training runs
    T_rom : int
        Frames per run (after subsampling)
    ref_method : str
        Reference computation method

    Returns
    -------
    dict with keys:
        - 'aligned': np.ndarray (M*T_rom, Ny, Nx) — aligned densities
        - 'ref': np.ndarray (Ny, Nx) — reference field
        - 'shifts': np.ndarray (M*T_rom, 2) — per-frame shifts
        - 'ref_method': str
    """
    # Compute global reference from all training data
    ref = compute_reference_field(all_densities, method=ref_method)

    # Compute per-frame shifts
    shifts = compute_shifts(all_densities, ref)

    # Apply alignment
    aligned = apply_shifts(all_densities, shifts)

    print(f"  Shift alignment: ref_method='{ref_method}'")
    print(f"  Mean |shift|: dy={np.abs(shifts[:, 0]).mean():.1f}, dx={np.abs(shifts[:, 1]).mean():.1f} pixels")
    print(f"  Max  |shift|: dy={np.abs(shifts[:, 0]).max()}, dx={np.abs(shifts[:, 1]).max()} pixels")

    return {
        'aligned': aligned,
        'ref': ref,
        'shifts': shifts,
        'ref_method': ref_method,
    }


def align_test_sequence(test_density: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align a single test density sequence to the training reference.

    Parameters
    ----------
    test_density : np.ndarray, shape (T, Ny, Nx)
    ref : np.ndarray, shape (Ny, Nx) — training reference

    Returns
    -------
    aligned : np.ndarray, shape (T, Ny, Nx)
    shifts : np.ndarray, shape (T, 2)
    """
    shifts = compute_shifts(test_density, ref)
    aligned = apply_shifts(test_density, shifts)
    return aligned, shifts


def predict_shifts_linear(known_shifts: np.ndarray, n_forecast: int) -> np.ndarray:
    """
    Predict future shifts by linear extrapolation of known teacher-forced shifts.

    Uses a simple linear fit to the last few known shifts.

    Parameters
    ----------
    known_shifts : np.ndarray, shape (T_known, 2)
        Shifts from teacher-forced period
    n_forecast : int
        Number of forecast steps

    Returns
    -------
    predicted_shifts : np.ndarray, shape (n_forecast, 2)
    """
    T_known = known_shifts.shape[0]

    if T_known < 2:
        # Can't extrapolate with < 2 points; use last known shift
        return np.tile(known_shifts[-1], (n_forecast, 1))

    # Linear fit to each component
    t_known = np.arange(T_known, dtype=np.float64)
    t_forecast = np.arange(T_known, T_known + n_forecast, dtype=np.float64)

    predicted = np.zeros((n_forecast, 2), dtype=np.int32)
    for dim in range(2):
        coeffs = np.polyfit(t_known, known_shifts[:, dim].astype(np.float64), 1)
        predicted[:, dim] = np.round(np.polyval(coeffs, t_forecast)).astype(np.int32)

    return predicted
