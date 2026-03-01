"""
Build the WSINDy weak linear system  **b = G w**  via FFT convolutions.

Given spatiotemporal field data ``U(t, x, y)`` and a test-function bundle
from Part 1, this module:

1. Generates query-point indices (uniform subsampling with optional margins).
2. Computes 3-D convolutions ``(kernel * U)`` using FFT, handling periodic
   spatial axes and (optionally) non-periodic time.
3. Evaluates nonlinear features ``f_j(U)`` and pairs them with differential
   operators ``D_i ψ`` to assemble the columns of **G**.
4. Extracts ``b`` and **G** at the query points.

All arrays follow the **(t, x, y)** axis convention.

Convolution convention
----------------------
Discrete convolution with centered kernel:

    result[i] = Σ_j kernel[j] · data[i − j + center]

where ``center = kernel_size // 2`` on each axis.  This is computed via
``IFFT(FFT(kernel) · FFT(data))``, which gives the raw circular convolution
``raw[n] = Σ_j kernel[j] · data[n − j]``.  To re-centre:

* **Periodic axis** — ``np.roll(raw, −center)``
* **Non-periodic axis** — zero-pad to ``n + m − 1``, then crop
  ``raw[center : center + n]``
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from .grid import GridSpec


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Query-point generation
# ═══════════════════════════════════════════════════════════════════════════

def make_query_indices(
    T: int,
    nx: int,
    ny: int,
    stride_t: int = 1,
    stride_x: int = 1,
    stride_y: int = 1,
    t_margin: int = 0,
) -> NDArray[np.intp]:
    """Return uniformly strided query-point indices.

    Parameters
    ----------
    T, nx, ny : int
        Data dimensions (time steps, spatial grid points).
    stride_t, stride_x, stride_y : int
        Strides for subsampling along each axis.
    t_margin : int
        Number of time steps to skip at the start **and** end of the
        time axis (to avoid boundary artefacts when time is non-periodic).
        Space is periodic, so no spatial margin is applied.

    Returns
    -------
    idx : ndarray of shape ``(K, 3)``, dtype ``intp``
        Each row is ``(t, x, y)``.
    """
    ts = np.arange(t_margin, T - t_margin, stride_t)
    xs = np.arange(0, nx, stride_x)
    ys = np.arange(0, ny, stride_y)

    gt, gx, gy = np.meshgrid(ts, xs, ys, indexing="ij")
    return np.column_stack([gt.ravel(), gx.ravel(), gy.ravel()])


def default_t_margin(psi_bundle: dict) -> int:
    """Sensible default time margin: the kernel half-width ``ellt``."""
    ellt: int = psi_bundle["ell"][0]
    return ellt


# ═══════════════════════════════════════════════════════════════════════════
# 2.  FFT convolution  (periodic / non-periodic per axis)
# ═══════════════════════════════════════════════════════════════════════════

def fft_convolve3d_same(
    data: NDArray[np.floating],
    kernel: NDArray[np.floating],
    periodic: Tuple[bool, bool, bool],
) -> NDArray[np.floating]:
    """3-D convolution returning an array the same shape as *data*.

    Parameters
    ----------
    data : ndarray, shape ``(T, nx, ny)``
        Input field.
    kernel : ndarray, shape ``(kt, kx, ky)``
        Compact-support convolution kernel (odd extents).
    periodic : (bool, bool, bool)
        Per-axis periodicity flags ``(periodic_t, periodic_x, periodic_y)``.

    Returns
    -------
    result : ndarray, same shape as *data*, dtype ``float64``
        ``result[i] = Σ_j kernel[j] · data[i − j + center]`` with
        periodic or zero-boundary handling per axis.
    """
    data = np.asarray(data, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if data.ndim != 3 or kernel.ndim != 3:
        raise ValueError("data and kernel must be 3-D")

    dshape = np.array(data.shape)
    kshape = np.array(kernel.shape)
    centers = kshape // 2

    # Determine FFT length per axis
    fft_shape = tuple(
        int(d) if p else int(d + k - 1)
        for d, k, p in zip(dshape, kshape, periodic)
    )

    # Forward FFT, multiply, inverse FFT  (rfftn for real data)
    D = np.fft.rfftn(data, s=fft_shape)
    K = np.fft.rfftn(kernel, s=fft_shape)
    raw = np.fft.irfftn(D * K, s=fft_shape)

    # Re-centre: roll periodic axes, crop non-periodic axes
    result = raw
    for ax in range(3):
        if periodic[ax]:
            result = np.roll(result, -int(centers[ax]), axis=ax)
        else:
            sl = [slice(None)] * 3
            sl[ax] = slice(int(centers[ax]), int(centers[ax]) + int(dshape[ax]))
            result = result[tuple(sl)]

    return np.ascontiguousarray(result)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Library: nonlinear features  f_j(U)
# ═══════════════════════════════════════════════════════════════════════════

#: Registry of scalar features.  Each maps U → f(U) element-wise.
FEATURES: Dict[str, object] = {
    "1":  lambda U: np.ones_like(U),
    "u":  lambda U: U.copy(),
    "u2": lambda U: U ** 2,
    "u3": lambda U: U ** 3,
}


def eval_feature(U: NDArray[np.floating], name: str) -> NDArray[np.floating]:
    """Evaluate a nonlinear feature on the field *U*.

    Parameters
    ----------
    U : ndarray, shape ``(T, nx, ny)``
    name : str
        One of ``"1"``, ``"u"``, ``"u2"``, ``"u3"``.

    Returns
    -------
    F : ndarray, same shape and dtype as *U*.
    """
    if name not in FEATURES:
        raise ValueError(
            f"Unknown feature '{name}'. Available: {sorted(FEATURES)}"
        )
    return np.asarray(FEATURES[name](U), dtype=np.float64)  # type: ignore[operator]


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Library: differential operators  → kernel selection
# ═══════════════════════════════════════════════════════════════════════════

_OPERATOR_MAP: Dict[str, str] = {
    "I":   "psi",       # identity (no spatial derivative)
    "dx":  "psi_x",
    "dy":  "psi_y",
    "dxx": "psi_xx",
    "dyy": "psi_yy",
}


def get_kernel(psi_bundle: dict, op: str) -> NDArray[np.floating]:
    """Return the convolution kernel for operator *op*.

    Parameters
    ----------
    psi_bundle : dict
        Output of :func:`make_separable_psi`.
    op : str
        One of ``"I"``, ``"dx"``, ``"dy"``, ``"dxx"``, ``"dyy"``,
        ``"lap"`` (Laplacian = ``psi_xx + psi_yy``).

    Returns
    -------
    kernel : 3-D ndarray
    """
    if op == "lap":
        return np.asarray(
            psi_bundle["psi_xx"] + psi_bundle["psi_yy"],
            dtype=np.float64,
        )
    if op not in _OPERATOR_MAP:
        raise ValueError(
            f"Unknown operator '{op}'. "
            f"Available: {sorted(_OPERATOR_MAP) + ['lap']}"
        )
    return np.asarray(psi_bundle[_OPERATOR_MAP[op]], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Build weak system  b = G w
# ═══════════════════════════════════════════════════════════════════════════

def build_weak_system(
    U: NDArray[np.floating],
    grid: GridSpec,
    psi_bundle: dict,
    library_terms: List[Tuple[str, str]],
    query_idx: NDArray[np.intp],
) -> Tuple[NDArray[np.floating], NDArray[np.floating], List[str]]:
    """Assemble the weak linear system at the given query points.

    Parameters
    ----------
    U : ndarray, shape ``(T, nx, ny)``
        Spatiotemporal field data.
    grid : GridSpec
        Grid metadata (used for periodicity flags).
    psi_bundle : dict
        Output of :func:`make_separable_psi`.
    library_terms : list of (op, feature) pairs
        Each pair ``(op, feature)`` specifies one column of **G**:
        ``G[:, m] = conv(get_kernel(op), eval_feature(U, feature))``
        evaluated at the query points.
    query_idx : ndarray, shape ``(K, 3)``
        Integer indices ``(t, x, y)`` of query points.

    Returns
    -------
    b : ndarray, shape ``(K,)``
        Left-hand side: ``(ψ_t * U)`` at query points.
    G : ndarray, shape ``(K, M)``
        Right-hand side matrix.
    col_names : list of *M* strings
        Human-readable column labels, e.g. ``"dxx:u"``.
    """
    U = np.asarray(U, dtype=np.float64)
    periodic = (grid.periodic_time, grid.periodic_space, grid.periodic_space)
    qt, qx, qy = query_idx[:, 0], query_idx[:, 1], query_idx[:, 2]
    K = query_idx.shape[0]

    # ── LHS: b = conv(psi_t, U) at query points ────────────────────────
    psi_t_kernel = np.asarray(psi_bundle["psi_t"], dtype=np.float64)
    b_full = fft_convolve3d_same(U, psi_t_kernel, periodic)
    b = b_full[qt, qx, qy]

    # ── RHS: G columns ─────────────────────────────────────────────────
    M = len(library_terms)
    G = np.empty((K, M), dtype=np.float64)
    col_names: List[str] = []

    for m, (op, feat) in enumerate(library_terms):
        F = eval_feature(U, feat)
        kern = get_kernel(psi_bundle, op)
        col_full = fft_convolve3d_same(F, kern, periodic)
        G[:, m] = col_full[qt, qx, qy]
        col_names.append(f"{op}:{feat}")

    return b, G, col_names
