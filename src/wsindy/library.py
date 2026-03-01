"""
Composable library builder for WSINDy PDE discovery.

Extends the basic ``(op, feature)`` library with:

* **Nonlinear diffusion:** :math:`\\Delta(u^p)`, :math:`\\nabla\\cdot(u^p \\nabla u)`
* **Conservative forms:** :math:`\\partial_x(f)`, :math:`\\nabla\\cdot(\\mathbf{F})`
* **Gradient nonlinearity:** :math:`|\\nabla u|^2`
* **Custom features and operators**

The builder produces a ``library_terms`` list of ``(op, feature)`` pairs
compatible with :func:`system.build_weak_system`, plus the feature and
operator registrations needed for :func:`system.eval_feature` and
:func:`rhs.eval_feature_pointwise`.

YAML integration
----------------
Library specs can be loaded from YAML config dicts::

    wsindy:
      library:
        max_poly: 3         # u, u^2, u^3
        operators: [I, dx, dy, lap]
        nonlinear_diffusion:
          enabled: true
          powers: [2]        # Δ(u²)
        gradient_nonlinearity:
          enabled: true      # |∇u|²
        conservative:
          enabled: true
          terms: ["dx:u2", "dy:u2"]   # ∂_x(u²), ∂_y(u²)
        custom_terms: []     # additional (op, feature) pairs

The ROM pipeline ignores unrecognised top-level keys, so ``wsindy:``
can coexist with ``rom:`` without conflict.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


# ═══════════════════════════════════════════════════════════════════
#  Feature/operator registries  (extend the ones in system.py/rhs.py)
# ═══════════════════════════════════════════════════════════════════

# Runtime registries — seeded from the defaults in system.py
_EXTRA_FEATURES: Dict[str, Callable[[NDArray], NDArray]] = {}
_EXTRA_OPERATORS: Dict[str, str] = {}  # op_name → psi_bundle key


def register_feature(name: str, fn: Callable[[NDArray], NDArray]) -> None:
    """Register a custom nonlinear feature ``f(U) → F``.

    Parameters
    ----------
    name : str
        Short name used in ``(op, feature)`` pairs (e.g. ``"u4"``).
    fn : callable
        Element-wise map ``U → f(U)``.
    """
    _EXTRA_FEATURES[name] = fn


def register_operator(name: str, kernel_key: str) -> None:
    """Register a custom operator with its psi_bundle key.

    Parameters
    ----------
    name : str
        Operator name (e.g. ``"dxy"``).
    kernel_key : str
        Key into ``psi_bundle`` dict (e.g. ``"psi_xy"``).
    """
    _EXTRA_OPERATORS[name] = kernel_key


def get_registered_features() -> Dict[str, Callable]:
    """Return all registered custom features."""
    return dict(_EXTRA_FEATURES)


def get_registered_operators() -> Dict[str, str]:
    """Return all registered custom operators."""
    return dict(_EXTRA_OPERATORS)


def clear_registrations() -> None:
    """Remove all custom features and operators."""
    _EXTRA_FEATURES.clear()
    _EXTRA_OPERATORS.clear()


# ═══════════════════════════════════════════════════════════════════
#  Standard library presets
# ═══════════════════════════════════════════════════════════════════

#: Default operators available out of the box
STANDARD_OPERATORS = ("I", "dx", "dy", "dxx", "dyy", "lap")

#: Feature names for polynomial powers of u
POLY_FEATURES = {
    0: "1",
    1: "u",
    2: "u2",
    3: "u3",
}


def _poly_feature_name(power: int) -> str:
    """Return feature name for u^power, creating it if needed."""
    if power in POLY_FEATURES:
        return POLY_FEATURES[power]
    name = f"u{power}"
    # Register the higher-order polynomial if not yet known
    if name not in _EXTRA_FEATURES:
        p = power  # capture
        _EXTRA_FEATURES[name] = lambda U, _p=p: U ** _p
    return name


# ═══════════════════════════════════════════════════════════════════
#  LibraryBuilder
# ═══════════════════════════════════════════════════════════════════

class LibraryBuilder:
    """Fluent builder for WSINDy library term lists.

    Example
    -------
    >>> lib = (LibraryBuilder()
    ...        .add_polynomials(max_power=3, operators=["I", "lap"])
    ...        .add_nonlinear_diffusion(powers=[2])
    ...        .add_gradient_nonlinearity()
    ...        .build())
    """

    def __init__(self) -> None:
        self._terms: List[Tuple[str, str]] = []
        self._seen: set = set()  # avoid duplicates

    def _add(self, op: str, feat: str) -> "LibraryBuilder":
        key = (op, feat)
        if key not in self._seen:
            self._terms.append(key)
            self._seen.add(key)
        return self

    # ── polynomial terms ───────────────────────────────────────────

    def add_polynomials(
        self,
        max_power: int = 3,
        operators: Sequence[str] = ("I", "dx", "dy", "dxx", "dyy", "lap"),
        include_constant: bool = True,
    ) -> "LibraryBuilder":
        """Add ``op:u^p`` for p=0..max_power and each operator.

        Parameters
        ----------
        max_power : int
            Highest polynomial power (e.g. 3 → 1, u, u², u³).
        operators : sequence of str
            Which operators to pair with each feature.
        include_constant : bool
            If True, include the constant feature ``"1"`` (only with
            operator ``"I"``).
        """
        start = 0 if include_constant else 1
        for p in range(start, max_power + 1):
            feat = _poly_feature_name(p)
            for op in operators:
                # Constant feature only makes sense with identity
                if feat == "1" and op != "I":
                    continue
                self._add(op, feat)
        return self

    # ── nonlinear diffusion ────────────────────────────────────────

    def add_nonlinear_diffusion(
        self,
        powers: Sequence[int] = (2,),
    ) -> "LibraryBuilder":
        r"""Add :math:`\Delta(u^p)` terms.

        In the weak formulation this becomes a column with operator
        ``lap`` applied to feature ``u^p``.  The ``"op:feature"``
        encoding is ``"lap:u2"`` for :math:`\Delta(u^2)`.

        Parameters
        ----------
        powers : sequence of int
            Polynomial powers (e.g. ``[2, 3]``).
        """
        for p in powers:
            feat = _poly_feature_name(p)
            self._add("lap", feat)
        return self

    # ── conservative forms ─────────────────────────────────────────

    def add_conservative(
        self,
        features: Sequence[str] = ("u2",),
        directions: Sequence[str] = ("dx", "dy"),
    ) -> "LibraryBuilder":
        r"""Add conservative flux terms :math:`\partial_x(f)`, etc.

        Parameters
        ----------
        features : sequence of str
            Features inside the divergence (e.g. ``["u2", "u3"]``).
        directions : sequence of str
            Derivative operators (e.g. ``["dx", "dy"]``).
        """
        for feat in features:
            _ = _poly_feature_name(int(feat.replace("u", "") or "1"))  # ensure registered
            for d in directions:
                self._add(d, feat)
        return self

    # ── gradient nonlinearity |∇u|² ────────────────────────────────

    def add_gradient_nonlinearity(self) -> "LibraryBuilder":
        r"""Add :math:`|\nabla u|^2 = u_x^2 + u_y^2`.

        This registers a custom feature ``"grad_u_sq"`` and pairs it
        with the identity operator.  It also registers the feature
        function so ``eval_feature`` can compute it.
        """
        def _grad_u_sq(U: NDArray) -> NDArray:
            """Compute |∇u|² at each snapshot using finite differences.

            This is a *feature-level* computation.  For the weak-form
            convolution, U is the full (T, nx, ny) field; for strong-form
            RHS evaluation, U is a single (nx, ny) snapshot.
            """
            # Use spectral derivatives for accuracy on periodic grids.
            # The grid info is not available here, so we use unit spacing
            # as a placeholder — the coefficient will absorb the scale.
            # In practice this feature should be paired with I operator.
            if U.ndim == 3:
                # (T, nx, ny) — per-snapshot
                result = np.zeros_like(U)
                for t in range(U.shape[0]):
                    u = U[t]
                    u_hat = np.fft.fft2(u)
                    nx, ny = u.shape
                    kx = 2 * np.pi * np.fft.fftfreq(nx)
                    ky = 2 * np.pi * np.fft.fftfreq(ny)
                    ux = np.fft.ifft2(1j * kx[:, None] * u_hat).real
                    uy = np.fft.ifft2(1j * ky[None, :] * u_hat).real
                    result[t] = ux ** 2 + uy ** 2
                return result
            else:
                # (nx, ny) single snapshot
                u_hat = np.fft.fft2(U)
                nx, ny = U.shape
                kx = 2 * np.pi * np.fft.fftfreq(nx)
                ky = 2 * np.pi * np.fft.fftfreq(ny)
                ux = np.fft.ifft2(1j * kx[:, None] * u_hat).real
                uy = np.fft.ifft2(1j * ky[None, :] * u_hat).real
                return ux ** 2 + uy ** 2

        register_feature("grad_u_sq", _grad_u_sq)
        self._add("I", "grad_u_sq")
        return self

    # ── custom term ────────────────────────────────────────────────

    def add_term(self, op: str, feat: str) -> "LibraryBuilder":
        """Add a single ``(op, feature)`` term."""
        self._add(op, feat)
        return self

    def add_terms(self, terms: Sequence[Tuple[str, str]]) -> "LibraryBuilder":
        """Add multiple ``(op, feature)`` pairs."""
        for op, feat in terms:
            self._add(op, feat)
        return self

    # ── build ──────────────────────────────────────────────────────

    def build(self) -> List[Tuple[str, str]]:
        """Return the final list of ``(op, feature)`` term specifications."""
        return list(self._terms)

    def __len__(self) -> int:
        return len(self._terms)

    def __repr__(self) -> str:
        return f"LibraryBuilder({len(self._terms)} terms)"


# ═══════════════════════════════════════════════════════════════════
#  YAML configuration interface
# ═══════════════════════════════════════════════════════════════════

def library_from_config(cfg: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Build a library term list from a YAML config dict.

    Parameters
    ----------
    cfg : dict
        The ``wsindy.library`` section of a YAML config, containing
        keys such as ``max_poly``, ``operators``,
        ``nonlinear_diffusion``, ``gradient_nonlinearity``,
        ``conservative``, ``custom_terms``.

    Returns
    -------
    list of ``(op, feature)`` tuples
    """
    builder = LibraryBuilder()

    # Polynomial terms
    max_poly = cfg.get("max_poly", 3)
    operators = cfg.get("operators", list(STANDARD_OPERATORS))
    include_constant = cfg.get("include_constant", True)
    builder.add_polynomials(
        max_power=max_poly,
        operators=operators,
        include_constant=include_constant,
    )

    # Nonlinear diffusion
    nld = cfg.get("nonlinear_diffusion", {})
    if nld.get("enabled", False):
        powers = nld.get("powers", [2])
        builder.add_nonlinear_diffusion(powers=powers)

    # Conservative forms
    cons = cfg.get("conservative", {})
    if cons.get("enabled", False):
        terms_raw = cons.get("terms", [])
        features = []
        directions = []
        for t in terms_raw:
            parts = t.split(":")
            if len(parts) == 2:
                directions.append(parts[0])
                features.append(parts[1])
        if features and directions:
            builder.add_conservative(
                features=list(set(features)),
                directions=list(set(directions)),
            )

    # Gradient nonlinearity
    gn = cfg.get("gradient_nonlinearity", {})
    if gn.get("enabled", False):
        builder.add_gradient_nonlinearity()

    # Custom terms
    custom = cfg.get("custom_terms", [])
    for item in custom:
        if isinstance(item, str) and ":" in item:
            op, feat = item.rsplit(":", 1)
            builder.add_term(op, feat)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            builder.add_term(item[0], item[1])

    return builder.build()


def default_library(
    max_poly: int = 3,
    operators: Sequence[str] = ("I", "dx", "dy", "lap"),
) -> List[Tuple[str, str]]:
    """Quick default library for common use cases.

    Returns polynomial features × operators, excluding nonsensical
    combinations like ``dx:1``.
    """
    return (
        LibraryBuilder()
        .add_polynomials(max_power=max_poly, operators=operators)
        .build()
    )


# ═══════════════════════════════════════════════════════════════════
#  Hook into system.py feature evaluation
# ═══════════════════════════════════════════════════════════════════

def patch_feature_registries() -> None:
    """Inject custom features into ``system.FEATURES`` and
    ``rhs.eval_feature_pointwise`` so that :func:`build_weak_system`
    and :func:`wsindy_rhs` can evaluate them.

    Call this after building the library (any ``register_feature``
    calls) and before running the pipeline.
    """
    from . import system as _sys
    from . import rhs as _rhs

    # Extend system.FEATURES dict
    for name, fn in _EXTRA_FEATURES.items():
        if name not in _sys.FEATURES:
            _sys.FEATURES[name] = fn

    # Monkey-patch rhs.eval_feature_pointwise to handle extras
    _original_efp = _rhs.eval_feature_pointwise

    def _extended_eval_feature_pointwise(u, name):
        if name in _EXTRA_FEATURES:
            return np.asarray(_EXTRA_FEATURES[name](u), dtype=np.float64)
        return _original_efp(u, name)

    # Only patch once
    if not getattr(_rhs.eval_feature_pointwise, "_patched", False):
        _rhs.eval_feature_pointwise = _extended_eval_feature_pointwise
        _rhs.eval_feature_pointwise._patched = True  # type: ignore[attr-defined]
