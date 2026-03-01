"""
WSINDy — Weak SINDy for PDE discovery from spatiotemporal field data.

Following Minor et al. (2025), "Learning Physically Interpretable
Atmospheric Models from Data with WSINDy."

Part 1: Grid specification and separable compact test-function generation.
Part 2: FFT-accelerated convolutions and weak-system assembly (b = Gw).
Part 3: Preconditioned MSTLS sparse regression and model container.
Part 4: Spectral RHS, time integrators (RK4/ETDRK4), forecasting, evaluation.
Part 5: Automated model selection across test-function scales.
Part 6: Interpretability — LaTeX/text rendering and term grouping.
Part 7: Uncertainty quantification and stability selection.
Part 8: Composable library builder with YAML config integration.
"""

from .grid import GridSpec
from .test_functions import make_1d_phi, make_separable_psi
from .utils import finite_diff_1d
from .system import (
    build_weak_system,
    default_t_margin,
    eval_feature,
    fft_convolve3d_same,
    get_kernel,
    make_query_indices,
)
from .regression import mstls, precondition_columns, solve_ls
from .model import WSINDyModel
from .metrics import r2_score, wsindy_fit_metrics
from .fit import wsindy_fit_regression

# Part 4
from .operators import (
    fft_wavenumbers,
    grad_spectral,
    dxx_spectral,
    dyy_spectral,
    laplacian_spectral,
)
from .rhs import (
    apply_operator_pointwise,
    eval_feature_pointwise,
    mass,
    parse_term,
    wsindy_rhs,
)
from .integrators import (
    build_L_hat,
    etdrk4_integrate,
    rk4_integrate,
    rk4_step,
    split_linear_nonlinear,
)
from .forecast import wsindy_forecast
from .eval import r2_per_snapshot, relative_l2, rollout_metrics

# Part 5
from .select import (
    SelectionResult,
    TrialResult,
    default_ell_grid,
    wsindy_model_selection,
)

# Part 6
from .pretty import group_terms, to_latex, to_text

# Part 7
from .uncertainty import BootstrapResult, bootstrap_wsindy
from .stability import StabilityResult, stability_selection

# Part 8
from .library import (
    LibraryBuilder,
    clear_registrations,
    default_library,
    library_from_config,
    patch_feature_registries,
    register_feature,
    register_operator,
)

# Part 9 — Multi-field PDE discovery
from .fields import (
    FieldData,
    build_field_data,
    build_field_data_rho_only,
    compute_flux_kde,
    compute_morse_potential,
)
from .multifield import (
    LibraryTerm,
    MultiFieldResult,
    bootstrap_multifield,
    build_default_library,
    build_stacked_multifield,
    build_weak_system_multifield,
    discover_multifield,
    forecast_multifield,
    library_from_config_multifield,
    model_selection_multifield,
)

__all__ = [
    # Part 1
    "GridSpec",
    "make_1d_phi",
    "make_separable_psi",
    "finite_diff_1d",
    # Part 2
    "build_weak_system",
    "default_t_margin",
    "eval_feature",
    "fft_convolve3d_same",
    "get_kernel",
    "make_query_indices",
    # Part 3
    "mstls",
    "precondition_columns",
    "solve_ls",
    "WSINDyModel",
    "r2_score",
    "wsindy_fit_metrics",
    "wsindy_fit_regression",
    # Part 4
    "fft_wavenumbers",
    "grad_spectral",
    "dxx_spectral",
    "dyy_spectral",
    "laplacian_spectral",
    "apply_operator_pointwise",
    "eval_feature_pointwise",
    "mass",
    "parse_term",
    "wsindy_rhs",
    "build_L_hat",
    "etdrk4_integrate",
    "rk4_integrate",
    "rk4_step",
    "split_linear_nonlinear",
    "wsindy_forecast",
    "r2_per_snapshot",
    "relative_l2",
    "rollout_metrics",
    # Part 5
    "SelectionResult",
    "TrialResult",
    "default_ell_grid",
    "wsindy_model_selection",
    # Part 6
    "group_terms",
    "to_latex",
    "to_text",
    # Part 7
    "BootstrapResult",
    "bootstrap_wsindy",
    "StabilityResult",
    "stability_selection",
    # Part 8
    "LibraryBuilder",
    "clear_registrations",
    "default_library",
    "library_from_config",
    "patch_feature_registries",
    "register_feature",
    "register_operator",
    # Part 9 — Multi-field
    "FieldData",
    "build_field_data",
    "build_field_data_rho_only",
    "compute_flux_kde",
    "compute_morse_potential",
    "LibraryTerm",
    "MultiFieldResult",
    "bootstrap_multifield",
    "build_default_library",
    "build_stacked_multifield",
    "build_weak_system_multifield",
    "discover_multifield",
    "forecast_multifield",
    "library_from_config_multifield",
    "model_selection_multifield",
]
