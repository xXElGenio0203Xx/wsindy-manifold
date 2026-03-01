"""
Multi-field WSINDy: library, weak system and forecast for coupled PDEs.
=======================================================================

Fits the **3-equation Vicsek–Morse system**:

    ρ_t  = Σ_j  c_j · Θ_j(ρ, p, Φ)
    p_x_t = Σ_k  d_k · Ξ_k(ρ, p, Φ)
    p_y_t = Σ_k  e_k · Ξ_k(ρ, p, Φ)        (sharing library structure with p_x)

where each library term ``Θ_j`` or ``Ξ_k`` is a *pre-computed scalar field*
evaluated from :class:`~wsindy.fields.FieldData`.

Architecture
~~~~~~~~~~~~
Each library term is:

    (term_name, callable(FieldData, t) → (ny, nx) array)

The weak system is built by convolving each library-term field with the
``psi`` test-function kernel and sampling at query points — *exactly*
the same machinery as the scalar case, just with richer columns.

The **p_x** and **p_y** equations share the same library *structure*
but use different target fields (p_x vs p_y) on the LHS and adjust
direction-sensitive terms (e.g., ∇ρ splits into ∂_x ρ vs ∂_y ρ).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .grid import GridSpec
from .fields import FieldData, _dx, _dy, _dxx, _dyy, _lap, _bilap, _div_vec
from .system import fft_convolve3d_same, make_query_indices, default_t_margin
from .test_functions import make_separable_psi
from .fit import wsindy_fit_regression
from .model import WSINDyModel


# Term names that are pure-linear in the target field.
# ETDRK4 treats these exactly in Fourier space.
_LINEAR_TERM_NAMES = frozenset({
    "px", "py",                  # linear decay / growth
    "lap_rho", "lap_px", "lap_py",  # diffusion
    "bilap_px", "bilap_py",      # hyper-viscosity
})


# ═══════════════════════════════════════════════════════════════════
#  Term: a named callable  FieldData → (T, ny, nx)
# ═══════════════════════════════════════════════════════════════════

# Type alias for library term evaluator
TermEval = Callable[[FieldData], np.ndarray]


class LibraryTerm:
    """One candidate term in the multi-field library.

    Parameters
    ----------
    name : str
        Human-readable label, e.g. ``"div_p"`` or ``"rho_grad_Phi_x"``.
    evaluator : callable
        ``f(fd: FieldData) → ndarray (T, ny, nx)``
    latex : str, optional
        LaTeX representation.
    equation : str
        Which equation this term belongs to: ``"rho"``, ``"px"``, ``"py"``.
    dangerous : bool
        If True, this term is prone to overfitting (informational flag).
    """
    __slots__ = ("name", "evaluator", "latex", "equation", "dangerous")

    def __init__(
        self,
        name: str,
        evaluator: TermEval,
        *,
        latex: str = "",
        equation: str = "rho",
        dangerous: bool = False,
    ):
        self.name = name
        self.evaluator = evaluator
        self.latex = latex or name
        self.equation = equation
        self.dangerous = dangerous

    def __repr__(self) -> str:
        return f"LibraryTerm({self.name!r}, eq={self.equation!r})"


# ═══════════════════════════════════════════════════════════════════
#  Pre-built term factories
# ═══════════════════════════════════════════════════════════════════

# ── ρ-equation terms ────────────────────────────────────────────

def rho_terms_core() -> List[LibraryTerm]:
    """C1: Conservative continuity core — almost always needed."""
    return [
        LibraryTerm(
            "div_p", lambda fd: fd.div_p(),
            latex=r"\nabla\cdot \mathbf{p}", equation="rho"),
        LibraryTerm(
            "lap_rho", lambda fd: fd.lap_rho(),
            latex=r"\Delta\rho", equation="rho"),
    ]


def rho_terms_morse() -> List[LibraryTerm]:
    """C2: Morse-driven aggregation."""
    return [
        LibraryTerm(
            "div_rho_gradPhi", lambda fd: fd.div_rho_gradPhi(),
            latex=r"\nabla\cdot(\rho\nabla\Phi)", equation="rho"),
    ]


def rho_terms_nonlinear_diff() -> List[LibraryTerm]:
    """C3: Nonlinear diffusion / crowd pressure surrogates."""
    return [
        LibraryTerm(
            "lap_rho2", lambda fd: fd.lap_rho2(),
            latex=r"\Delta(\rho^2)", equation="rho"),
        LibraryTerm(
            "lap_rho3", lambda fd: fd.lap_rho3(),
            latex=r"\Delta(\rho^3)", equation="rho"),
    ]


def rho_terms_coupling() -> List[LibraryTerm]:
    """C4: Coupling to alignment intensity (optional, dangerous)."""
    return [
        LibraryTerm(
            "lap_p_sq", lambda fd: fd.lap_p_sq(),
            latex=r"\Delta(|\mathbf{p}|^2)", equation="rho",
            dangerous=True),
    ]


# ── p_x-equation terms ─────────────────────────────────────────

def px_terms_linear() -> List[LibraryTerm]:
    """D1: Linear + saturation (alignment)."""
    return [
        LibraryTerm(
            "px", lambda fd: fd.px,
            latex=r"p_x", equation="px"),
        LibraryTerm(
            "p_sq_px", lambda fd: fd.p_sq() * fd.px,
            latex=r"|\mathbf{p}|^2 p_x", equation="px"),
    ]


def px_terms_pressure() -> List[LibraryTerm]:
    """D2: Pressure / density coupling."""
    return [
        LibraryTerm(
            "dx_rho", lambda fd: fd.dx_rho(),
            latex=r"\partial_x \rho", equation="px"),
    ]


def px_terms_pressure_extended() -> List[LibraryTerm]:
    """D2 extended: ∇(ρ²)."""
    return [
        LibraryTerm(
            "dx_rho2", lambda fd: fd.dx_rho2(),
            latex=r"\partial_x(\rho^2)", equation="px",
            dangerous=True),
    ]


def px_terms_viscosity() -> List[LibraryTerm]:
    """D3: Spatial smoothing / viscosity."""
    return [
        LibraryTerm(
            "lap_px", lambda fd: fd.lap_px(),
            latex=r"\Delta p_x", equation="px"),
    ]


def px_terms_hyperviscosity() -> List[LibraryTerm]:
    """D3 extended: Δ²p stabiliser."""
    return [
        LibraryTerm(
            "bilap_px", lambda fd: fd.bilap_px(),
            latex=r"\Delta^2 p_x", equation="px",
            dangerous=True),
    ]


def px_terms_morse() -> List[LibraryTerm]:
    """D5: Morse forcing on polarization."""
    return [
        LibraryTerm(
            "rho_dx_Phi", lambda fd: fd.rho_grad_Phi_x(),
            latex=r"\rho\,\partial_x\Phi", equation="px"),
    ]


def px_terms_selfadvect() -> List[LibraryTerm]:
    """D4: Self-advection (risky but sometimes key)."""
    return [
        LibraryTerm(
            "p_dot_grad_px", lambda fd: fd.p_dot_grad_px(),
            latex=r"(\mathbf{p}\cdot\nabla)p_x", equation="px",
            dangerous=True),
    ]


# ── p_y-equation terms (symmetric to p_x) ──────────────────────

def py_terms_linear() -> List[LibraryTerm]:
    return [
        LibraryTerm(
            "py", lambda fd: fd.py,
            latex=r"p_y", equation="py"),
        LibraryTerm(
            "p_sq_py", lambda fd: fd.p_sq() * fd.py,
            latex=r"|\mathbf{p}|^2 p_y", equation="py"),
    ]


def py_terms_pressure() -> List[LibraryTerm]:
    return [
        LibraryTerm(
            "dy_rho", lambda fd: fd.dy_rho(),
            latex=r"\partial_y \rho", equation="py"),
    ]


def py_terms_pressure_extended() -> List[LibraryTerm]:
    return [
        LibraryTerm(
            "dy_rho2", lambda fd: fd.dy_rho2(),
            latex=r"\partial_y(\rho^2)", equation="py",
            dangerous=True),
    ]


def py_terms_viscosity() -> List[LibraryTerm]:
    return [
        LibraryTerm(
            "lap_py", lambda fd: fd.lap_py(),
            latex=r"\Delta p_y", equation="py"),
    ]


def py_terms_hyperviscosity() -> List[LibraryTerm]:
    return [
        LibraryTerm(
            "bilap_py", lambda fd: fd.bilap_py(),
            latex=r"\Delta^2 p_y", equation="py",
            dangerous=True),
    ]


def py_terms_morse() -> List[LibraryTerm]:
    return [
        LibraryTerm(
            "rho_dy_Phi", lambda fd: fd.rho_grad_Phi_y(),
            latex=r"\rho\,\partial_y\Phi", equation="py"),
    ]


def py_terms_selfadvect() -> List[LibraryTerm]:
    return [
        LibraryTerm(
            "p_dot_grad_py", lambda fd: fd.p_dot_grad_py(),
            latex=r"(\mathbf{p}\cdot\nabla)p_y", equation="py",
            dangerous=True),
    ]


# ═══════════════════════════════════════════════════════════════════
#  Recommended library builder
# ═══════════════════════════════════════════════════════════════════

def build_default_library(
    *,
    morse: bool = True,
    rich: bool = False,
) -> Dict[str, List[LibraryTerm]]:
    """Build the recommended thesis-grade library.

    Parameters
    ----------
    morse : bool
        Include Morse-derived terms (requires Φ field).
    rich : bool
        Include "richer" optional terms (Δ(|p|²), (p·∇)p, Δ²p, ∇(ρ²)).

    Returns
    -------
    dict mapping ``"rho"``, ``"px"``, ``"py"`` to their term lists.
    """
    # ── ρ equation ──
    rho_lib = rho_terms_core()
    if morse:
        rho_lib += rho_terms_morse()
    rho_lib += rho_terms_nonlinear_diff()
    if rich:
        rho_lib += rho_terms_coupling()

    # ── p_x equation ──
    px_lib = px_terms_linear() + px_terms_pressure() + px_terms_viscosity()
    if morse:
        px_lib += px_terms_morse()
    if rich:
        px_lib += px_terms_selfadvect() + px_terms_hyperviscosity()
        px_lib += px_terms_pressure_extended()

    # ── p_y equation ──
    py_lib = py_terms_linear() + py_terms_pressure() + py_terms_viscosity()
    if morse:
        py_lib += py_terms_morse()
    if rich:
        py_lib += py_terms_selfadvect() + py_terms_hyperviscosity()
        py_lib += py_terms_pressure_extended()

    return {"rho": rho_lib, "px": px_lib, "py": py_lib}


def library_from_config_multifield(cfg: dict) -> Dict[str, List[LibraryTerm]]:
    """Build multi-field library from YAML config.

    Config structure::

        multifield_library:
          morse: true
          rich: false
          rho_extras: []
          px_extras: []
          py_extras: []

    Returns dict of ``{eq_name: [LibraryTerm, ...]}``.
    """
    morse = cfg.get("morse", True)
    rich = cfg.get("rich", False)
    lib = build_default_library(morse=morse, rich=rich)

    # Could add custom extras here in later expansion
    return lib


# ═══════════════════════════════════════════════════════════════════
#  Multi-field weak system builder
# ═══════════════════════════════════════════════════════════════════

def build_weak_system_multifield(
    fd: FieldData,
    psi_bundle: dict,
    library: List[LibraryTerm],
    query_idx: NDArray[np.intp],
    target_field: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build weak system ``b = G w`` for one equation of the multi-field PDE.

    This is the multi-field analogue of :func:`system.build_weak_system`.
    Instead of computing ``eval_feature(U, name)`` then applying a
    differential-operator kernel, each library term *directly provides*
    the full (T, ny, nx) field to convolve with ``ψ`` (identity kernel).

    Parameters
    ----------
    fd : FieldData
        Precomputed field container with ρ, p_x, p_y, Φ, etc.
    psi_bundle : dict
        Output of :func:`make_separable_psi`.
    library : list of LibraryTerm
        Candidate terms for this equation.
    query_idx : ndarray (K, 3)
        Query-point indices ``(t, x, y)``.
    target_field : ndarray (T, ny, nx)
        The field whose time derivative is the LHS (ρ for ρ_t, etc.).

    Returns
    -------
    b : (K,)
    G : (K, M)
    col_names : list of str
    """
    periodic = (False, True, True)  # time non-periodic, space periodic
    qt, qx, qy = query_idx[:, 0], query_idx[:, 1], query_idx[:, 2]
    K = query_idx.shape[0]
    M = len(library)

    # ── LHS: b = conv(ψ_t, target_field) ───────────────────────
    psi_t = np.asarray(psi_bundle["psi_t"], dtype=np.float64)
    b_full = fft_convolve3d_same(target_field.astype(np.float64), psi_t, periodic)
    b = b_full[qt, qx, qy]

    # ── RHS: conv(ψ, Θ_j) where ψ is the identity test function ──
    # Each term is convolved with ψ (no differential operator on ψ —
    # derivatives are already baked into the term field).
    psi_kern = np.asarray(psi_bundle["psi"], dtype=np.float64)
    G = np.empty((K, M), dtype=np.float64)
    col_names = []

    for m, term in enumerate(library):
        # Evaluate the term field across all (t, x, y)
        field_m = term.evaluator(fd)
        field_m = np.asarray(field_m, dtype=np.float64)
        conv_m = fft_convolve3d_same(field_m, psi_kern, periodic)
        G[:, m] = conv_m[qt, qx, qy]
        col_names.append(term.name)

    return b, G, col_names


# ═══════════════════════════════════════════════════════════════════
#  Stacked multi-trajectory weak system
# ═══════════════════════════════════════════════════════════════════

def build_stacked_multifield(
    field_list: List[FieldData],
    psi_bundle: dict,
    library: List[LibraryTerm],
    target_accessor: Callable[[FieldData], np.ndarray],
    stride: Tuple[int, int, int] = (2, 2, 2),
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Stack weak systems from multiple trajectories for one equation.

    Parameters
    ----------
    field_list : list of FieldData
        One per training trajectory.
    psi_bundle : dict
        Shared test-function bundle.
    library : list of LibraryTerm
        Candidate terms.
    target_accessor : callable
        ``f(fd) → ndarray (T, ny, nx)`` — extracts the target field.
        E.g., ``lambda fd: fd.rho`` for the ρ equation.
    stride : (int, int, int)
        Query-point strides.

    Returns
    -------
    b, G, col_names — concatenated across trajectories.
    """
    all_b, all_G = [], []
    col_names = None
    t_margin = default_t_margin(psi_bundle)

    for fd in field_list:
        T, ny, nx = fd.shape
        if 2 * t_margin >= T:
            continue

        qi = make_query_indices(
            T, nx, ny,
            stride_t=stride[0], stride_x=stride[1], stride_y=stride[2],
            t_margin=t_margin,
        )
        if qi.shape[0] < len(library) + 1:
            continue

        target = target_accessor(fd)
        b_k, G_k, cn = build_weak_system_multifield(
            fd, psi_bundle, library, qi, target,
        )
        all_b.append(b_k)
        all_G.append(G_k)
        if col_names is None:
            col_names = cn

    if not all_b:
        raise ValueError("No valid query points from any trajectory")

    return np.concatenate(all_b), np.vstack(all_G), col_names


# ═══════════════════════════════════════════════════════════════════
#  Model selection for one equation (multi-field)
# ═══════════════════════════════════════════════════════════════════

def fit_equation_multifield(
    field_list: List[FieldData],
    library: List[LibraryTerm],
    target_accessor: Callable[[FieldData], np.ndarray],
    ell: Tuple[int, int, int],
    p: Tuple[int, int, int] = (2, 2, 2),
    stride: Tuple[int, int, int] = (2, 2, 2),
    lambdas: Optional[np.ndarray] = None,
    max_iter: int = 25,
) -> Tuple[WSINDyModel, np.ndarray, np.ndarray, List[str]]:
    """Fit one equation of the multi-field system at given ℓ.

    Returns ``(model, b, G, col_names)``.
    """
    grid = field_list[0].grid
    psi_bundle = make_separable_psi(
        grid,
        ellt=ell[0], ellx=ell[1], elly=ell[2],
        pt=p[0], px=p[1], py=p[2],
    )

    b, G, col_names = build_stacked_multifield(
        field_list, psi_bundle, library, target_accessor, stride,
    )

    model = wsindy_fit_regression(
        b, G, col_names, lambdas=lambdas, max_iter=max_iter,
    )
    return model, b, G, col_names


# ═══════════════════════════════════════════════════════════════════
#  Full 3-equation discovery pipeline
# ═══════════════════════════════════════════════════════════════════

class MultiFieldResult:
    """Result of the 3-equation multi-field WSINDy discovery."""

    def __init__(
        self,
        rho_model: WSINDyModel,
        px_model: WSINDyModel,
        py_model: WSINDyModel,
        rho_terms: List[LibraryTerm],
        px_terms: List[LibraryTerm],
        py_terms: List[LibraryTerm],
    ):
        self.rho_model = rho_model
        self.px_model = px_model
        self.py_model = py_model
        self.rho_terms = rho_terms
        self.px_terms = px_terms
        self.py_terms = py_terms

    def summary(self) -> str:
        lines = ["=" * 60, "  Multi-field WSINDy Discovery Results", "=" * 60]

        for eq_name, model in [
            ("ρ_t", self.rho_model),
            ("p_x_t", self.px_model),
            ("p_y_t", self.py_model),
        ]:
            r2 = model.diagnostics.get("r2", 0)
            n_act = model.n_active
            lines.append(f"\n  {eq_name}  (R²_weak={r2:.4f}, {n_act} active terms)")
            for i in range(len(model.w)):
                if model.active[i]:
                    lines.append(f"    {model.w[i]:+12.4e}  {model.col_names[i]}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialisable summary for JSON export."""
        out = {}
        for eq_name, model, terms in [
            ("rho", self.rho_model, self.rho_terms),
            ("px", self.px_model, self.px_terms),
            ("py", self.py_model, self.py_terms),
        ]:
            out[eq_name] = {
                "col_names": model.col_names,
                "w": model.w.tolist(),
                "active": model.active.tolist(),
                "active_terms": model.active_terms,
                "coefficients": {
                    n: float(model.w[model.col_names.index(n)])
                    for n in model.active_terms
                },
                "r2_weak": float(model.diagnostics.get("r2", 0)),
                "n_active": model.n_active,
                "best_lambda": float(model.best_lambda),
            }
        return out


def discover_multifield(
    field_list: List[FieldData],
    library: Dict[str, List[LibraryTerm]],
    ell: Tuple[int, int, int],
    p: Tuple[int, int, int] = (2, 2, 2),
    stride: Tuple[int, int, int] = (2, 2, 2),
    lambdas: Optional[np.ndarray] = None,
    max_iter: int = 25,
    verbose: bool = True,
) -> MultiFieldResult:
    """Discover all 3 equations at a given test-function scale ℓ.

    Parameters
    ----------
    field_list : list of FieldData
        Training trajectories.
    library : dict
        ``{"rho": [...], "px": [...], "py": [...]}`` from
        :func:`build_default_library`.
    ell, p, stride : tuples
        Test-function / query-point parameters.
    lambdas : ndarray, optional
        MSTLS regularisation grid.

    Returns
    -------
    MultiFieldResult
    """
    if lambdas is None:
        lambdas = np.logspace(-4, 1, 40)

    results = {}
    for eq_name, target_fn in [
        ("rho", lambda fd: fd.rho),
        ("px",  lambda fd: fd.px),
        ("py",  lambda fd: fd.py),
    ]:
        lib_eq = library[eq_name]
        if verbose:
            print(f"    Fitting {eq_name}_t  ({len(lib_eq)} candidates)...")

        model, b, G, cn = fit_equation_multifield(
            field_list, lib_eq, target_fn,
            ell=ell, p=p, stride=stride,
            lambdas=lambdas, max_iter=max_iter,
        )

        r2 = model.diagnostics.get("r2", 0)
        if verbose:
            print(f"      R²_weak={r2:.4f}, active={model.n_active}")
            for i in range(len(model.w)):
                if model.active[i]:
                    print(f"        {model.w[i]:+10.4e}  {cn[i]}")

        results[eq_name] = (model, b, G, cn)

    return MultiFieldResult(
        rho_model=results["rho"][0],
        px_model=results["px"][0],
        py_model=results["py"][0],
        rho_terms=library["rho"],
        px_terms=library["px"],
        py_terms=library["py"],
    )


# ═══════════════════════════════════════════════════════════════════
#  Model selection over ℓ for full 3-equation system
# ═══════════════════════════════════════════════════════════════════

def model_selection_multifield(
    field_list: List[FieldData],
    library: Dict[str, List[LibraryTerm]],
    ell_grid: List[Tuple[int, int, int]],
    p: Tuple[int, int, int] = (2, 2, 2),
    stride: Tuple[int, int, int] = (2, 2, 2),
    lambdas: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[MultiFieldResult, Tuple[int, int, int]]:
    """Model selection: try each ℓ and pick the best overall.

    "Best" is defined as the ℓ that maximises the average R²_weak
    across all three equations, penalising excessive active terms.

    Returns
    -------
    best_result : MultiFieldResult
    best_ell : tuple
    """
    import time as _time

    best_score = -np.inf
    best_result = None
    best_ell = None

    for idx, ell in enumerate(ell_grid):
        t0 = _time.perf_counter()
        try:
            result = discover_multifield(
                field_list, library, ell, p, stride, lambdas,
                verbose=False,
            )
        except Exception as exc:
            if verbose:
                print(f"  [{idx+1}/{len(ell_grid)}] ℓ={ell} FAILED: {exc}")
            continue

        elapsed = _time.perf_counter() - t0

        r2_rho = result.rho_model.diagnostics.get("r2", 0)
        r2_px = result.px_model.diagnostics.get("r2", 0)
        r2_py = result.py_model.diagnostics.get("r2", 0)
        n_act = (result.rho_model.n_active
                 + result.px_model.n_active
                 + result.py_model.n_active)
        n_lib = (len(library["rho"]) + len(library["px"]) + len(library["py"]))

        # Score: average R² minus complexity penalty
        avg_r2 = (r2_rho + r2_px + r2_py) / 3
        complexity_penalty = 0.05 * n_act / max(n_lib, 1)
        score = avg_r2 - complexity_penalty

        if verbose:
            print(
                f"  [{idx+1}/{len(ell_grid)}] ℓ={ell}  "
                f"R²(ρ)={r2_rho:.3f}  R²(px)={r2_px:.3f}  R²(py)={r2_py:.3f}  "
                f"active={n_act}  score={score:.4f}  ({elapsed:.1f}s)"
            )

        if score > best_score:
            best_score = score
            best_result = result
            best_ell = ell

    if best_result is None:
        raise RuntimeError("All model selection trials failed")

    if verbose:
        print(f"\n  Best ℓ = {best_ell}  (score = {best_score:.4f})")
        print(best_result.summary())

    return best_result, best_ell


# ═══════════════════════════════════════════════════════════════════
#  Bootstrap UQ for multi-field system
# ═══════════════════════════════════════════════════════════════════

def bootstrap_multifield(
    field_list: List[FieldData],
    library: Dict[str, List[LibraryTerm]],
    ell: Tuple[int, int, int],
    p: Tuple[int, int, int] = (2, 2, 2),
    stride: Tuple[int, int, int] = (2, 2, 2),
    lambdas: Optional[np.ndarray] = None,
    B: int = 50,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, dict]:
    """Bootstrap UQ for all 3 equations via trajectory resampling.

    Instead of row resampling (as in scalar), we resample *trajectories*
    to capture inter-trajectory variability.

    Returns
    -------
    dict mapping ``"rho", "px", "py"`` each to a dict with
    ``coeff_samples, coeff_mean, coeff_std, inclusion_probability``.
    """
    rng = np.random.default_rng(seed)
    N_traj = len(field_list)
    if lambdas is None:
        lambdas = np.logspace(-4, 1, 40)

    boot_data = {}
    for eq_name, target_fn in [
        ("rho", lambda fd: fd.rho),
        ("px",  lambda fd: fd.px),
        ("py",  lambda fd: fd.py),
    ]:
        lib_eq = library[eq_name]
        M = len(lib_eq)
        samples = np.zeros((B, M))
        active_counts = np.zeros(M)

        if verbose:
            print(f"  Bootstrap {eq_name}_t: {B} replicates...")

        for rep in range(B):
            idx = rng.choice(N_traj, size=N_traj, replace=True)
            fd_sub = [field_list[i] for i in idx]

            try:
                model, _, _, _ = fit_equation_multifield(
                    fd_sub, lib_eq, target_fn,
                    ell=ell, p=p, stride=stride,
                    lambdas=lambdas,
                )
                samples[rep] = model.w
                active_counts += model.active.astype(float)
            except Exception:
                pass  # skip failed replicates

        boot_data[eq_name] = {
            "coeff_samples": samples,
            "coeff_mean": samples.mean(axis=0),
            "coeff_std": samples.std(axis=0),
            "inclusion_probability": active_counts / B,
            "col_names": [t.name for t in lib_eq],
        }

    return boot_data


# ═══════════════════════════════════════════════════════════════════
#  Multi-field forecast (RK4 time integration of 3-field system)
# ═══════════════════════════════════════════════════════════════════

def multifield_rhs(
    rho: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    grid: GridSpec,
    Lx: float,
    Ly: float,
    rho_model: WSINDyModel,
    px_model: WSINDyModel,
    py_model: WSINDyModel,
    rho_terms: List[LibraryTerm],
    px_terms: List[LibraryTerm],
    py_terms: List[LibraryTerm],
    morse_params: Optional[Dict[str, float]] = None,
    xgrid: Optional[np.ndarray] = None,
    ygrid: Optional[np.ndarray] = None,
    use_spectral: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the RHS of the discovered 3-equation system.

    Parameters
    ----------
    rho, px, py : (ny, nx) — current state
    use_spectral : bool
        If True, use FFT-based spectral derivatives (machine-precision
        on periodic grids, unconditionally stable).  Default True for
        forecast; set False to match training FD derivatives.

    Returns d_rho/dt, d_px/dt, d_py/dt as (ny, nx) each.
    """
    # Build a single-frame FieldData (T=1)
    rho_3d = rho[np.newaxis]
    px_3d = px[np.newaxis]
    py_3d = py[np.newaxis]

    Phi = None
    if morse_params is not None and xgrid is not None and ygrid is not None:
        from .fields import compute_morse_potential
        Phi = compute_morse_potential(
            rho_3d, xgrid, ygrid, Lx, Ly,
            morse_params["Cr"], morse_params["Ca"],
            morse_params["lr"], morse_params["la"],
        )

    fd = FieldData(
        rho_3d, px_3d, py_3d, grid, Lx, Ly, Phi=Phi,
        use_spectral=use_spectral,
    )

    # Evaluate each equation's RHS
    def _eval_rhs(model, terms, fd):
        rhs = np.zeros((fd.shape[1], fd.shape[2]), dtype=np.float64)
        for i, t in enumerate(terms):
            if model.active[i]:
                field_val = t.evaluator(fd)[0]  # (ny, nx)
                rhs += model.w[i] * field_val
        return rhs

    d_rho = _eval_rhs(rho_model, rho_terms, fd)
    d_px = _eval_rhs(px_model, px_terms, fd)
    d_py = _eval_rhs(py_model, py_terms, fd)

    return d_rho, d_px, d_py


# ═══════════════════════════════════════════════════════════════════
#  ETDRK4 linear operator extraction
# ═══════════════════════════════════════════════════════════════════

def _extract_linear_spectral(
    model: WSINDyModel,
    terms: List[LibraryTerm],
    nx: int,
    ny: int,
    Lx: float,
    Ly: float,
) -> Optional[np.ndarray]:
    """Extract the diagonal linear operator L in Fourier space.

    Returns L_hat (ny, nx) such that  L[u] = ifft2(L_hat * fft2(u)),
    or ``None`` if no pure linear Laplacian/bilaplacian term is active.

    Currently recognises:
      - ``lap_*``   → coefficient * (-K²)
      - ``bilap_*`` → coefficient * (K⁴)
      - ``px``, ``py`` → coefficient * 1  (linear decay/growth)
    """
    from .fields import _spectral_wavenumbers_2d
    K2 = _spectral_wavenumbers_2d(ny, nx, Lx, Ly)
    L_hat = np.zeros((ny, nx), dtype=np.complex128)
    has_linear = False

    for i, t in enumerate(terms):
        if not model.active[i]:
            continue
        name = t.name
        c = model.w[i]
        if name.startswith("lap_") and "rho" not in name:
            # Δu_i  →  -K² in Fourier space
            L_hat += c * (-K2)
            has_linear = True
        elif name.startswith("bilap_"):
            # Δ²u_i  →  K⁴
            L_hat += c * (K2**2)
            has_linear = True
        elif name in ("px", "py"):
            # Linear decay/growth: c * u_i  →  c * 1
            L_hat += c
            has_linear = True

    return L_hat if has_linear else None


def _etdrk4_coefficients(
    L_hat: np.ndarray, dt: float,
) -> Tuple[np.ndarray, ...]:
    """ETDRK4 coefficients from Cox & Matthews (2002).

    Uses the contour-integral formulas of Kassam & Trefethen (2005)
    for numerical stability near L=0.

    Parameters
    ----------
    L_hat : (ny, nx) complex — diagonal of the linear operator
    dt : float

    Returns
    -------
    E, E2, a21, a31, a32, a41, a42, a43, b1, b2, b3, b4
        All shape (ny, nx) complex arrays.
    """
    M = 32  # contour integration points
    z = L_hat * dt
    r = np.exp(2j * np.pi * (np.arange(1, M + 1) - 0.5) / M)  # unit circle

    # Broadcast: z is (ny, nx), r is (M,) → (ny, nx, M)
    z_3d = z[..., np.newaxis]
    zr = z_3d + r  # (ny, nx, M)

    E = np.exp(z)
    E2 = np.exp(z / 2)

    # φ functions via contour integrals (Kassam & Trefethen)
    phi1 = np.mean((np.exp(zr) - 1) / zr, axis=-1)
    phi2 = np.mean((np.exp(zr) - 1 - zr) / zr**2, axis=-1)
    phi3 = np.mean((np.exp(zr) - 1 - zr - 0.5 * zr**2) / zr**3, axis=-1)

    # Half-step versions
    zr2 = z_3d / 2 + r
    phi1_h = np.mean((np.exp(zr2) - 1) / zr2, axis=-1)

    # ETDRK4 coefficients (Kassam & Trefethen 2005, Table 1)
    a21 = dt * phi1_h
    b1 = dt * (4 * phi3 - 3 * phi2 + phi1)
    b2 = dt * 2 * (phi2 - 2 * phi3)
    b3 = b2.copy()
    b4 = dt * (-phi2 + 4 * phi3)

    return E, E2, a21, b1, b2, b3, b4


def forecast_multifield(
    rho0: np.ndarray,
    px0: np.ndarray,
    py0: np.ndarray,
    result: MultiFieldResult,
    grid: GridSpec,
    Lx: float,
    Ly: float,
    n_steps: int,
    morse_params: Optional[Dict[str, float]] = None,
    xgrid: Optional[np.ndarray] = None,
    ygrid: Optional[np.ndarray] = None,
    clip_negative_rho: bool = True,
    mass_conserve: bool = True,
    method: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forecast the 3-field PDE system discovered by WSINDy.

    Integrators
    -----------
    * ``"rk4"``   — classical Runge–Kutta 4.  Simple, works if not stiff.
    * ``"etdrk4"`` — exponential time differencing (Cox & Matthews 2002,
      Kassam & Trefethen 2005).  Treats diffusion/Laplacian terms
      **exactly** in Fourier space, so the stiff part never limits Δt.
    * ``"auto"``  — choose ETDRK4 if any equation has an active
      Laplacian or bilaplacian term, otherwise RK4.

    Physical projections (applied every step)
    ------------------------------------------
    * Nonnegativity: ``ρ ← max(ρ, 0)``
    * Mass conservation: ``ρ ← ρ · M₀/M(ρ)``

    Parameters
    ----------
    rho0, px0, py0 : (ny, nx) — initial conditions
    result : MultiFieldResult — discovered PDE
    n_steps : int — time steps to forecast
    clip_negative_rho : bool — enforce ρ ≥ 0
    mass_conserve : bool — enforce ∫ρ dx = const each step
    method : {"rk4", "etdrk4", "auto"}

    Returns
    -------
    rho_hist, px_hist, py_hist : (n_steps+1, ny, nx)
    """
    ny, nx = rho0.shape
    dt = grid.dt

    rho_hist = np.zeros((n_steps + 1, ny, nx))
    px_hist = np.zeros((n_steps + 1, ny, nx))
    py_hist = np.zeros((n_steps + 1, ny, nx))

    rho_hist[0] = rho0.copy()
    px_hist[0] = px0.copy()
    py_hist[0] = py0.copy()

    # Initial mass for conservation
    M0 = float(np.sum(rho0))

    # ── Decide integrator ───────────────────────────────────────
    use_etdrk4 = False
    L_rho_hat = L_px_hat = L_py_hat = None

    if method in ("auto", "etdrk4"):
        L_rho_hat = _extract_linear_spectral(
            result.rho_model, result.rho_terms, nx, ny, Lx, Ly)
        L_px_hat = _extract_linear_spectral(
            result.px_model, result.px_terms, nx, ny, Lx, Ly)
        L_py_hat = _extract_linear_spectral(
            result.py_model, result.py_terms, nx, ny, Lx, Ly)

        has_stiff = any(L is not None for L in [L_rho_hat, L_px_hat, L_py_hat])

        if method == "etdrk4" or (method == "auto" and has_stiff):
            use_etdrk4 = True
            # Default zero for fields without linear part
            zero_L = np.zeros((ny, nx), dtype=np.complex128)
            L_rho_hat = L_rho_hat if L_rho_hat is not None else zero_L
            L_px_hat = L_px_hat if L_px_hat is not None else zero_L
            L_py_hat = L_py_hat if L_py_hat is not None else zero_L

    actual_method = "etdrk4" if use_etdrk4 else "rk4"

    # ── Build nonlinear RHS (excludes terms handled by L) ───────
    def nonlinear_rhs(rho, px_cur, py_cur, exclude_linear=False):
        """Evaluate the full (or nonlinear-only) RHS in spectral mode."""
        rho_3d = rho[np.newaxis]
        px_3d = px_cur[np.newaxis]
        py_3d = py_cur[np.newaxis]

        Phi = None
        if morse_params is not None and xgrid is not None and ygrid is not None:
            from .fields import compute_morse_potential
            Phi = compute_morse_potential(
                rho_3d, xgrid, ygrid, Lx, Ly,
                morse_params["Cr"], morse_params["Ca"],
                morse_params["lr"], morse_params["la"],
            )

        fd = FieldData(
            rho_3d, px_3d, py_3d, grid, Lx, Ly, Phi=Phi,
            use_spectral=True,  # Always spectral in forecast
        )

        def _eval_eq(model, terms, fd, skip_linear):
            val = np.zeros((ny, nx), dtype=np.float64)
            for i, t in enumerate(terms):
                if not model.active[i]:
                    continue
                if skip_linear and t.name in _LINEAR_TERM_NAMES:
                    continue
                val += model.w[i] * t.evaluator(fd)[0]
            return val

        d_rho = _eval_eq(result.rho_model, result.rho_terms, fd, exclude_linear)
        d_px = _eval_eq(result.px_model, result.px_terms, fd, exclude_linear)
        d_py = _eval_eq(result.py_model, result.py_terms, fd, exclude_linear)
        return d_rho, d_px, d_py

    def _project(rho, px_cur, py_cur):
        """Nonnegativity + mass conservation."""
        if clip_negative_rho:
            rho = np.maximum(rho, 0.0)
        if mass_conserve and M0 > 0:
            M = float(np.sum(rho))
            if M > 0:
                rho = rho * (M0 / M)
        return rho, px_cur, py_cur

    # ── RK4 loop ────────────────────────────────────────────────
    rho = rho0.copy()
    px_cur = px0.copy()
    py_cur = py0.copy()

    if not use_etdrk4:
        def rhs_full(r, p_x, p_y):
            return nonlinear_rhs(r, p_x, p_y, exclude_linear=False)

        for n in range(n_steps):
            k1r, k1x, k1y = rhs_full(rho, px_cur, py_cur)
            k2r, k2x, k2y = rhs_full(
                rho + 0.5 * dt * k1r,
                px_cur + 0.5 * dt * k1x,
                py_cur + 0.5 * dt * k1y,
            )
            k3r, k3x, k3y = rhs_full(
                rho + 0.5 * dt * k2r,
                px_cur + 0.5 * dt * k2x,
                py_cur + 0.5 * dt * k2y,
            )
            k4r, k4x, k4y = rhs_full(
                rho + dt * k3r,
                px_cur + dt * k3x,
                py_cur + dt * k3y,
            )

            rho = rho + (dt / 6) * (k1r + 2 * k2r + 2 * k3r + k4r)
            px_cur = px_cur + (dt / 6) * (k1x + 2 * k2x + 2 * k3x + k4x)
            py_cur = py_cur + (dt / 6) * (k1y + 2 * k2y + 2 * k3y + k4y)

            rho, px_cur, py_cur = _project(rho, px_cur, py_cur)

            if np.max(np.abs(rho)) > 1e10 or np.any(np.isnan(rho)):
                print(f"    WARNING: RK4 diverged at step {n+1}/{n_steps}")
                rho_hist[n + 1:] = rho_hist[n]
                px_hist[n + 1:] = px_hist[n]
                py_hist[n + 1:] = py_hist[n]
                break

            rho_hist[n + 1] = rho
            px_hist[n + 1] = px_cur
            py_hist[n + 1] = py_cur

    # ── ETDRK4 loop ─────────────────────────────────────────────
    else:
        # Precompute ETDRK4 coefficients per field
        coeff_rho = _etdrk4_coefficients(L_rho_hat, dt)
        coeff_px = _etdrk4_coefficients(L_px_hat, dt)
        coeff_py = _etdrk4_coefficients(L_py_hat, dt)

        def _etdrk4_step(u_hat, N_fns, coeffs):
            """One ETDRK4 step for a single field in Fourier space.

            u_hat : current field in Fourier space
            N_fns : list of 4 nonlinear RHS evaluations in Fourier space
            coeffs : (E, E2, a21, b1, b2, b3, b4)
            """
            E, E2, a21, b1, b2, b3, b4 = coeffs
            Na, Nb, Nc, Nd = N_fns
            # a = E2 * u_hat + a21 * Na  (first half-step)
            a = E2 * u_hat + a21 * Na
            # b = E2 * u_hat + a21 * Nb
            b = E2 * u_hat + a21 * Nb
            # c = E2 * a + a21 * (2*Nc - Na)
            c = E2 * a + a21 * (2 * Nc - Na)
            # Final
            return E * u_hat + b1 * Na + b2 * (Nb + Nc) + b4 * Nd

        rho_hat = np.fft.fft2(rho)
        px_hat = np.fft.fft2(px_cur)
        py_hat = np.fft.fft2(py_cur)

        for n in range(n_steps):
            # Stage 1: N(u_n)
            Nr1, Nx1, Ny1 = nonlinear_rhs(rho, px_cur, py_cur, exclude_linear=True)
            Nr1_hat = np.fft.fft2(Nr1)
            Nx1_hat = np.fft.fft2(Nx1)
            Ny1_hat = np.fft.fft2(Ny1)

            # Stage 2: u_a = E2*u + a21*N1
            E_r, E2_r, a21_r, b1_r, b2_r, b3_r, b4_r = coeff_rho
            E_x, E2_x, a21_x, b1_x, b2_x, b3_x, b4_x = coeff_px
            E_y, E2_y, a21_y, b1_y, b2_y, b3_y, b4_y = coeff_py

            rho_a_hat = E2_r * rho_hat + a21_r * Nr1_hat
            px_a_hat = E2_x * px_hat + a21_x * Nx1_hat
            py_a_hat = E2_y * py_hat + a21_y * Ny1_hat

            rho_a = np.real(np.fft.ifft2(rho_a_hat))
            px_a = np.real(np.fft.ifft2(px_a_hat))
            py_a = np.real(np.fft.ifft2(py_a_hat))

            Nr2, Nx2, Ny2 = nonlinear_rhs(rho_a, px_a, py_a, exclude_linear=True)
            Nr2_hat = np.fft.fft2(Nr2)
            Nx2_hat = np.fft.fft2(Nx2)
            Ny2_hat = np.fft.fft2(Ny2)

            # Stage 3
            rho_b_hat = E2_r * rho_hat + a21_r * Nr2_hat
            px_b_hat = E2_x * px_hat + a21_x * Nx2_hat
            py_b_hat = E2_y * py_hat + a21_y * Ny2_hat

            rho_b = np.real(np.fft.ifft2(rho_b_hat))
            px_b = np.real(np.fft.ifft2(px_b_hat))
            py_b = np.real(np.fft.ifft2(py_b_hat))

            Nr3, Nx3, Ny3 = nonlinear_rhs(rho_b, px_b, py_b, exclude_linear=True)
            Nr3_hat = np.fft.fft2(Nr3)
            Nx3_hat = np.fft.fft2(Nx3)
            Ny3_hat = np.fft.fft2(Ny3)

            # Stage 4
            rho_c_hat = E2_r * rho_a_hat + a21_r * (2 * Nr3_hat - Nr1_hat)
            px_c_hat = E2_x * px_a_hat + a21_x * (2 * Nx3_hat - Nx1_hat)
            py_c_hat = E2_y * py_a_hat + a21_y * (2 * Ny3_hat - Ny1_hat)

            rho_c = np.real(np.fft.ifft2(rho_c_hat))
            px_c = np.real(np.fft.ifft2(px_c_hat))
            py_c = np.real(np.fft.ifft2(py_c_hat))

            Nr4, Nx4, Ny4 = nonlinear_rhs(rho_c, px_c, py_c, exclude_linear=True)
            Nr4_hat = np.fft.fft2(Nr4)
            Nx4_hat = np.fft.fft2(Nx4)
            Ny4_hat = np.fft.fft2(Ny4)

            # Final combination
            rho_hat = (E_r * rho_hat
                       + b1_r * Nr1_hat + b2_r * (Nr2_hat + Nr3_hat) + b4_r * Nr4_hat)
            px_hat = (E_x * px_hat
                      + b1_x * Nx1_hat + b2_x * (Nx2_hat + Nx3_hat) + b4_x * Nx4_hat)
            py_hat = (E_y * py_hat
                      + b1_y * Ny1_hat + b2_y * (Ny2_hat + Ny3_hat) + b4_y * Ny4_hat)

            rho = np.real(np.fft.ifft2(rho_hat))
            px_cur = np.real(np.fft.ifft2(px_hat))
            py_cur = np.real(np.fft.ifft2(py_hat))

            rho, px_cur, py_cur = _project(rho, px_cur, py_cur)

            # After projection, update Fourier coefficients
            rho_hat = np.fft.fft2(rho)
            px_hat = np.fft.fft2(px_cur)
            py_hat = np.fft.fft2(py_cur)

            if np.max(np.abs(rho)) > 1e10 or np.any(np.isnan(rho)):
                print(f"    WARNING: ETDRK4 diverged at step {n+1}/{n_steps}")
                rho_hist[n + 1:] = rho_hist[n]
                px_hist[n + 1:] = px_hist[n]
                py_hist[n + 1:] = py_hist[n]
                break

            rho_hist[n + 1] = rho
            px_hist[n + 1] = px_cur
            py_hist[n + 1] = py_cur

    return rho_hist, px_hist, py_hist
