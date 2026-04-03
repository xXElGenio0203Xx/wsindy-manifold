import numpy as np
import pytest

from wsindy.grid import GridSpec
from wsindy.model import WSINDyModel
from wsindy.fields import FieldData
from wsindy.multifield import (
    LibraryTerm,
    MultiFieldForecastError,
    MultiFieldResult,
    build_default_library,
    fit_equation_multifield,
    forecast_multifield,
    model_selection_multifield,
    resolve_regime_aware_library_settings,
)


def _model(col_names, weights, active=None, *, r2=0.9):
    weights = np.asarray(weights, dtype=np.float64)
    if active is None:
        active = weights != 0.0
    return WSINDyModel(
        col_names=list(col_names),
        w=weights,
        active=np.asarray(active, dtype=bool),
        best_lambda=1e-3,
        col_scale=np.ones(len(col_names), dtype=np.float64),
        diagnostics={"r2": float(r2), "lambda_history": []},
    )


def _empty_model():
    return _model([], np.zeros(0, dtype=np.float64), active=np.zeros(0, dtype=bool), r2=0.0)


def _simple_field_data(T=12, ny=6, nx=6):
    x = np.linspace(0.0, 2.0 * np.pi, nx, endpoint=False)
    y = np.linspace(0.0, 2.0 * np.pi, ny, endpoint=False)
    X, Y = np.meshgrid(x, y)
    rho = np.stack([
        1.0 + 0.2 * np.sin(X + 0.1 * t) + 0.1 * np.cos(2 * Y - 0.05 * t)
        for t in range(T)
    ])
    px = np.stack([0.1 * np.cos(X - 0.03 * t) for t in range(T)])
    py = np.stack([0.1 * np.sin(Y + 0.02 * t) for t in range(T)])
    grid = GridSpec(dt=0.05, dx=1.0, dy=1.0)
    return FieldData(rho, px, py, grid, Lx=float(nx), Ly=float(ny), use_spectral=False)


def test_build_default_library_continuity_first_rho_terms_are_conservative():
    lib = build_default_library(
        morse=True,
        rich=True,
        rho_strategy="continuity_first",
        regime_class="attractive",
    )
    rho_names = {term.name for term in lib["rho"]}
    assert "div_p" in rho_names
    assert "lap_rho" not in rho_names
    assert "div_rho_gradPhi" in rho_names
    assert "lap_rho2" not in rho_names
    assert "lap_rho3" not in rho_names
    assert "lap_p_sq" not in rho_names


def test_build_default_library_attractive_rich_includes_class_specific_terms():
    lib = build_default_library(
        morse=True,
        rich=True,
        rho_strategy="continuity_first",
        regime_class="attractive",
    )
    rho_names = {term.name for term in lib["rho"]}
    px_names = {term.name for term in lib["px"]}
    py_names = {term.name for term in lib["py"]}

    assert "div_p" in rho_names
    assert "div_rho_gradPhi" in rho_names
    assert {"rho_dx_Phi", "div_px_p", "dx_rho2"} <= px_names
    assert {"rho_dy_Phi", "div_py_p", "dy_rho2"} <= py_names


def test_build_default_library_repulsive_rich_excludes_attraction_specific_terms():
    lib = build_default_library(
        morse=True,
        rich=True,
        rho_strategy="continuity_first",
        regime_class="repulsive",
    )
    rho_names = {term.name for term in lib["rho"]}
    px_names = {term.name for term in lib["px"]}
    py_names = {term.name for term in lib["py"]}

    assert "div_p" in rho_names
    assert "div_rho_gradPhi" not in rho_names
    assert {"rho_dx_Phi", "div_px_p"} <= px_names
    assert {"rho_dy_Phi", "div_py_p"} <= py_names
    assert "dx_rho2" not in px_names
    assert "dy_rho2" not in py_names


def test_build_default_library_pure_vicsek_repulsive_excludes_all_morse_terms():
    lib = build_default_library(
        morse=False,
        rich=True,
        rho_strategy="continuity_first",
        regime_class="repulsive",
    )
    all_names = {term.name for eq_terms in lib.values() for term in eq_terms}
    assert "div_rho_gradPhi" not in all_names
    assert "rho_dx_Phi" not in all_names
    assert "rho_dy_Phi" not in all_names


def test_resolve_multifield_regime_settings_auto_classifies_regimes():
    blackhole = resolve_regime_aware_library_settings(
        forces_enabled=True,
        Ca=20.0,
        Cr=0.05,
        morse_requested=True,
    )
    assert blackhole["regime_class"] == "attractive"
    assert blackhole["regime_class_source"] == "auto"
    assert blackhole["effective_morse"] is True
    assert blackhole["ca_cr_ratio"] == pytest.approx(400.0)

    supernova = resolve_regime_aware_library_settings(
        forces_enabled=True,
        Ca=0.05,
        Cr=15.0,
        morse_requested=True,
    )
    assert supernova["regime_class"] == "repulsive"
    assert supernova["effective_morse"] is True

    pure_vicsek = resolve_regime_aware_library_settings(
        forces_enabled=False,
        Ca=1.5,
        Cr=0.5,
        morse_requested=True,
    )
    assert pure_vicsek["regime_class"] == "repulsive"
    assert pure_vicsek["forces_enabled"] is False
    assert pure_vicsek["effective_morse"] is False
    assert pure_vicsek["morse_requested"] is True


def test_resolve_multifield_regime_settings_override_takes_precedence():
    resolved = resolve_regime_aware_library_settings(
        forces_enabled=True,
        Ca=0.05,
        Cr=15.0,
        morse_requested=True,
        regime_class="attractive",
    )
    assert resolved["regime_class"] == "attractive"
    assert resolved["regime_class_source"] == "override"


def test_fit_equation_multifield_forces_required_divergence_term(monkeypatch):
    fd = _simple_field_data()
    library = [
        LibraryTerm("div_p", lambda field: field.div_p(), equation="rho"),
        LibraryTerm("lap_rho", lambda field: field.lap_rho(), equation="rho"),
    ]

    def fake_fit(b, G, col_names, lambdas=None, max_iter=25):
        return _model(col_names, [0.0, 1.0], active=[False, True], r2=0.95)

    monkeypatch.setattr("wsindy.multifield.wsindy_fit_regression", fake_fit)
    model, *_ = fit_equation_multifield(
        [fd],
        library,
        lambda field: field.rho,
        ell=(2, 2, 2),
        required_terms=("div_p",),
    )

    assert "div_p" in model.active_terms
    assert model.diagnostics["required_terms_forced"] == ["div_p"]


def test_fit_equation_multifield_drops_unphysical_positive_py_terms(monkeypatch):
    fd = _simple_field_data()
    library = [
        LibraryTerm("py", lambda field: field.py, equation="py"),
        LibraryTerm("lap_py", lambda field: field.lap_py(), equation="py"),
        LibraryTerm("dy_rho2", lambda field: field.dy_rho2(), equation="py"),
    ]

    def fake_fit(b, G, col_names, lambdas=None, max_iter=25):
        return _model(col_names, [0.4, 0.3, 1.2], active=[True, True, True], r2=0.9)

    monkeypatch.setattr("wsindy.multifield.wsindy_fit_regression", fake_fit)
    model, *_ = fit_equation_multifield(
        [fd],
        library,
        lambda field: field.py,
        ell=(2, 2, 2),
    )

    assert "py" not in model.active_terms
    assert "lap_py" not in model.active_terms
    assert "dy_rho2" not in model.active_terms
    assert model.diagnostics["sign_constraints_dropped"] == ["py", "lap_py", "dy_rho2"]


def test_fit_equation_multifield_drops_oversized_coefficients(monkeypatch):
    fd = _simple_field_data()
    library = [
        LibraryTerm("dy_rho", lambda field: field.dy_rho(), equation="py"),
        LibraryTerm("p_sq_py", lambda field: field.p_sq() * field.py, equation="py"),
    ]

    def fake_fit(b, G, col_names, lambdas=None, max_iter=25):
        return _model(col_names, [-8.0, 0.4], active=[True, True], r2=0.85)

    monkeypatch.setattr("wsindy.multifield.wsindy_fit_regression", fake_fit)
    model, *_ = fit_equation_multifield(
        [fd],
        library,
        lambda field: field.py,
        ell=(2, 2, 2),
    )

    assert "dy_rho" not in model.active_terms
    assert "p_sq_py" in model.active_terms
    assert model.diagnostics["sign_constraints_dropped"] == ["dy_rho"]


def test_forecast_multifield_raises_on_nan_rhs():
    nan_term = LibraryTerm(
        "bad_term",
        lambda fd: np.full(fd.rho.shape, np.nan, dtype=np.float64),
        equation="rho",
    )
    result = MultiFieldResult(
        rho_model=_model(["bad_term"], [1.0]),
        px_model=_empty_model(),
        py_model=_empty_model(),
        rho_terms=[nan_term],
        px_terms=[],
        py_terms=[],
        metadata={"use_spectral": False},
    )
    grid = GridSpec(dt=0.1, dx=1.0, dy=1.0)
    rho0 = np.ones((4, 4), dtype=np.float64)
    px0 = np.zeros_like(rho0)
    py0 = np.zeros_like(rho0)

    with pytest.raises(MultiFieldForecastError):
        forecast_multifield(
            rho0,
            px0,
            py0,
            result,
            grid,
            Lx=4.0,
            Ly=4.0,
            n_steps=2,
            clip_negative_rho=False,
            mass_conserve=False,
            method="rk4",
        )


def test_forecast_multifield_etdrk4_uses_active_lap_rho():
    lap_term = LibraryTerm("lap_rho", lambda fd: fd.lap_rho(), equation="rho")
    result = MultiFieldResult(
        rho_model=_model(["lap_rho"], [0.2], r2=0.8),
        px_model=_empty_model(),
        py_model=_empty_model(),
        rho_terms=[lap_term],
        px_terms=[],
        py_terms=[],
        metadata={"use_spectral": False},
    )
    grid = GridSpec(dt=0.02, dx=1.0, dy=1.0)
    x = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    y = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    X, Y = np.meshgrid(x, y)
    rho0 = 1.0 + 0.3 * np.sin(X) * np.cos(Y)
    px0 = np.zeros_like(rho0)
    py0 = np.zeros_like(rho0)

    rho_hist, _, _ = forecast_multifield(
        rho0,
        px0,
        py0,
        result,
        grid,
        Lx=8.0,
        Ly=8.0,
        n_steps=1,
        clip_negative_rho=False,
        mass_conserve=False,
        method="etdrk4",
    )

    assert not np.allclose(rho_hist[1], rho_hist[0])


def test_forecast_multifield_uses_training_backend_metadata():
    dx_term = LibraryTerm("dx_rho_custom", lambda fd: fd.dx_rho(), equation="rho")
    base_kwargs = dict(
        rho_model=_model(["dx_rho_custom"], [0.4], r2=0.7),
        px_model=_empty_model(),
        py_model=_empty_model(),
        rho_terms=[dx_term],
        px_terms=[],
        py_terms=[],
    )
    grid = GridSpec(dt=0.05, dx=1.0, dy=1.0)
    x = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    y = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    X, Y = np.meshgrid(x, y)
    rho0 = np.sin(3 * X) + 0.5 * np.cos(5 * Y)
    px0 = np.zeros_like(rho0)
    py0 = np.zeros_like(rho0)

    result_fd = MultiFieldResult(metadata={"use_spectral": False}, **base_kwargs)
    result_sp = MultiFieldResult(metadata={"use_spectral": True}, **base_kwargs)

    rho_fd, _, _ = forecast_multifield(
        rho0, px0, py0, result_fd, grid,
        Lx=16.0, Ly=16.0, n_steps=1,
        clip_negative_rho=False, mass_conserve=False, method="rk4",
    )
    rho_sp, _, _ = forecast_multifield(
        rho0, px0, py0, result_sp, grid,
        Lx=16.0, Ly=16.0, n_steps=1,
        clip_negative_rho=False, mass_conserve=False, method="rk4",
    )

    assert not np.allclose(rho_fd[1], rho_sp[1])


def test_model_selection_multifield_prefers_motion_aware_transport(monkeypatch):
    ell_static = (7, 3, 3)
    ell_transport = (10, 5, 5)

    rho_terms = [LibraryTerm("div_p", lambda fd: fd.div_p(), equation="rho")]
    px_terms = [LibraryTerm("px", lambda fd: fd.px, equation="px")]
    py_terms = [LibraryTerm("py", lambda fd: fd.py, equation="py")]
    library = {"rho": rho_terms, "px": px_terms, "py": py_terms}

    def fake_result(tag, weak_r2, rho_active_terms):
        rho_model = _model(["div_p"], [1.0 if "div_p" in rho_active_terms else 0.0], active=["div_p" in rho_active_terms], r2=weak_r2)
        px_model = _model(["px"], [1.0], r2=weak_r2)
        py_model = _model(["py"], [1.0], r2=weak_r2)
        return MultiFieldResult(
            rho_model=rho_model,
            px_model=px_model,
            py_model=py_model,
            rho_terms=rho_terms,
            px_terms=px_terms,
            py_terms=py_terms,
            metadata={"tag": tag, "use_spectral": False},
        )

    def fake_discover(field_list, library, ell, p=(2, 2, 2), stride=(2, 2, 2), lambdas=None, max_iter=25, rho_strategy="legacy", verbose=True):
        if ell == ell_static:
            return fake_result("static", 0.96, [])
        return fake_result("transport", 0.82, ["div_p"])

    def fake_rollout(fd, result, *, n_steps, morse_params):
        if result.metadata["tag"] == "static":
            return {
                "status": "ok",
                "r2_rho": 0.1,
                "r2_px": 0.1,
                "r2_py": 0.1,
                "mass_drift": 0.2,
                "motion_ratio": 0.0,
            }
        return {
            "status": "ok",
            "r2_rho": 0.95,
            "r2_px": 0.9,
            "r2_py": 0.9,
            "mass_drift": 0.0,
            "motion_ratio": 1.0,
        }

    monkeypatch.setattr("wsindy.multifield.discover_multifield", fake_discover)
    monkeypatch.setattr("wsindy.multifield._short_rollout_diagnostics", fake_rollout)

    fd = _simple_field_data(T=20)
    best_result, best_ell = model_selection_multifield(
        [fd],
        library,
        [ell_static, ell_transport],
        rho_strategy="continuity_first",
        validation_trajectories=1,
        validation_steps=5,
        verbose=False,
    )

    assert best_ell == ell_transport
    assert "div_p" in best_result.rho_model.active_terms
