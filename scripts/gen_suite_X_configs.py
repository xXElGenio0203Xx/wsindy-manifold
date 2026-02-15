#!/usr/bin/env python3
"""Generate 12 Suite X config YAMLs for cross-regime validation."""
import yaml
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_DIR = os.path.join(ROOT, "configs")

# ============================================================================
# COMMON BLOCKS
# ============================================================================

density = dict(nx=48, ny=48, bandwidth=4.0)
outputs = dict(density_resolution=48, density_bandwidth=4.0)
alignment = dict(enabled=True)

# V1 ICs (from KNEE suite)
v1_train_ic = dict(
    type="mixed_comprehensive",
    gaussian=dict(enabled=True, n_runs=120,
                  positions_x=[3.0, 6.0, 9.0, 12.0],
                  positions_y=[3.0, 6.0, 9.0, 12.0],
                  variances=[0.8, 1.5, 2.5],
                  n_samples_per_config=2),
    uniform=dict(enabled=True, n_runs=50, n_samples=50),
    two_clusters=dict(enabled=True, n_runs=30,
                      separations=[3.0, 5.0, 7.0],
                      sigmas=[1.0, 2.0],
                      n_samples_per_config=5),
    ring=dict(enabled=False),
)

# V3.3/V3.4 ICs (larger training set)
v33_train_ic = dict(
    type="mixed_comprehensive",
    gaussian=dict(enabled=True, n_runs=180,
                  positions_x=[3.0, 6.0, 9.0, 12.0],
                  positions_y=[3.0, 6.0, 9.0, 12.0],
                  variances=[0.8, 1.5, 2.5],
                  n_samples_per_config=3),
    uniform=dict(enabled=True, n_runs=75, n_samples=75),
    two_clusters=dict(enabled=True, n_runs=45,
                      separations=[3.0, 5.0, 7.0],
                      sigmas=[1.0, 2.0],
                      n_samples_per_config=7),
    ring=dict(enabled=False),
)

# Shared test ICs (26 runs)
test_ic = dict(
    type="mixed_test_comprehensive",
    gaussian=dict(enabled=True, n_runs=15,
                  test_positions_x=[4.5, 7.5, 10.5],
                  test_positions_y=[4.5, 7.5, 10.5],
                  test_variances=[1.2],
                  n_samples_per_config=1,
                  extrapolation_positions=[[2.0, 2.0], [13.0, 2.0],
                                           [2.0, 13.0], [13.0, 13.0],
                                           [7.5, 1.5], [1.5, 7.5]],
                  extrapolation_variance=[1.2]),
    uniform=dict(enabled=True, n_runs=8),
    two_clusters=dict(enabled=True, n_runs=7,
                      test_separations=[4.0, 6.0],
                      test_sigmas=[1.5],
                      n_samples_per_config=3,
                      extrapolation_separations=[2.0],
                      extrapolation_sigma=[1.5]),
    ring=dict(enabled=False),
)

# ============================================================================
# REGIME DEFINITIONS
# ============================================================================
regimes = {
    "V1": dict(
        sim=dict(N=100, T=5.0, dt=0.04, Lx=15.0, Ly=15.0, bc="periodic"),
        model=dict(type="discrete", speed=1.5, speed_mode="constant_with_forces"),
        params=dict(R=2.5),
        noise=dict(kind="gaussian", eta=0.2, match_variance=True),
        forces=dict(enabled=True, type="morse",
                    params=dict(Ca=0.8, Cr=0.3, la=1.5, lr=0.5, mu_t=0.3, rcut_factor=5.0)),
        forecast_start=0.60, d=19, p=5, alpha=1e-4,
        train_ic=v1_train_ic,
    ),
    "V3.3": dict(
        sim=dict(N=100, T=6.0, dt=0.04, Lx=15.0, Ly=15.0, bc="periodic"),
        model=dict(type="discrete", speed=3.0, speed_mode="constant_with_forces"),
        params=dict(R=4.0),
        noise=dict(kind="gaussian", eta=0.15, match_variance=True),
        forces=dict(enabled=True, type="morse",
                    params=dict(Ca=2.0, Cr=0.3, la=1.5, lr=0.5, mu_t=0.3, rcut_factor=5.0)),
        forecast_start=0.36, d=20, p=3, alpha=10.0,
        train_ic=v33_train_ic,
    ),
    "V3.4": dict(
        sim=dict(N=100, T=6.0, dt=0.04, Lx=15.0, Ly=15.0, bc="periodic"),
        model=dict(type="discrete", speed=5.0, speed_mode="constant_with_forces"),
        params=dict(R=5.0),
        noise=dict(kind="gaussian", eta=0.1, match_variance=True),
        forces=dict(enabled=True, type="morse",
                    params=dict(Ca=3.0, Cr=0.5, la=1.5, lr=0.5, mu_t=0.3, rcut_factor=5.0)),
        forecast_start=0.36, d=10, p=3, alpha=10.0,
        train_ic=v33_train_ic,  # same larger IC set
    ),
}

# ============================================================================
# 12 EXPERIMENTS
# ============================================================================
experiments = [
    # V1 regime
    dict(name="X1_V1_raw_H37",           regime="V1",   transform="raw",  postprocess="none",    horizon=37,  test_T=None),
    dict(name="X2_V1_sqrtSimplex_H37",   regime="V1",   transform="sqrt", postprocess="simplex", horizon=37,  test_T=None),
    dict(name="X3_V1_raw_H162",          regime="V1",   transform="raw",  postprocess="none",    horizon=162, test_T=20.04),
    dict(name="X4_V1_sqrtSimplex_H162",  regime="V1",   transform="sqrt", postprocess="simplex", horizon=162, test_T=20.04),
    # V3.3 regime
    dict(name="X5_V33_raw_H37",          regime="V3.3", transform="raw",  postprocess="none",    horizon=37,  test_T=None),
    dict(name="X6_V33_sqrtSimplex_H37",  regime="V3.3", transform="sqrt", postprocess="simplex", horizon=37,  test_T=None),
    dict(name="X7_V33_raw_H162",         regime="V3.3", transform="raw",  postprocess="none",    horizon=162, test_T=19.80),
    dict(name="X8_V33_sqrtSimplex_H162", regime="V3.3", transform="sqrt", postprocess="simplex", horizon=162, test_T=19.80),
    # V3.4 regime
    dict(name="X9_V34_raw_H37",          regime="V3.4", transform="raw",  postprocess="none",    horizon=37,  test_T=None),
    dict(name="X10_V34_sqrtSimplex_H37", regime="V3.4", transform="sqrt", postprocess="simplex", horizon=37,  test_T=None),
    dict(name="X11_V34_raw_H162",        regime="V3.4", transform="raw",  postprocess="none",    horizon=162, test_T=19.80),
    dict(name="X12_V34_sqrtSimplex_H162",regime="V3.4", transform="sqrt", postprocess="simplex", horizon=162, test_T=19.80),
]

# ============================================================================
# GENERATE
# ============================================================================
for exp in experiments:
    r = regimes[exp["regime"]]

    cfg = {}
    cfg["experiment_name"] = exp["name"]

    # sim
    cfg["sim"] = dict(r["sim"])

    # test_sim (only for H162 — extends test simulations beyond train T)
    if exp["test_T"] is not None:
        cfg["test_sim"] = dict(T=exp["test_T"])

    cfg["model"] = dict(r["model"])
    cfg["params"] = dict(r["params"])
    cfg["noise"] = dict(r["noise"])
    cfg["forces"] = dict(r["forces"])
    cfg["alignment"] = dict(alignment)
    cfg["density"] = dict(density)
    cfg["outputs"] = dict(outputs)
    cfg["train_ic"] = dict(r["train_ic"])
    cfg["test_ic"] = dict(test_ic)

    # ROM block
    rom = dict(subsample=3, fixed_modes=r["d"])
    if exp["transform"] == "sqrt":
        rom["density_transform"] = "sqrt"
        rom["density_transform_eps"] = 1e-10
    else:
        rom["density_transform"] = "raw"
    rom["models"] = dict(
        mvar=dict(enabled=True, lag=r["p"], ridge_alpha=r["alpha"]),
        lstm=dict(enabled=False),
    )
    cfg["rom"] = rom

    # Eval block
    ev = dict(
        metrics=["r2", "rmse"],
        save_forecasts=True,
        save_time_resolved=True,
        forecast_start=r["forecast_start"],
        clamp_negative=True,
    )
    if exp["postprocess"] != "none":
        ev["mass_postprocess"] = exp["postprocess"]
    cfg["eval"] = ev

    # Write
    path = os.path.join(CFG_DIR, f"{exp['name']}.yaml")
    with open(path, "w") as f:
        f.write("---\n")
        label = "simplex postprocess" if exp["postprocess"] == "simplex" else "no postprocess"
        f.write(f"# Suite X: {exp['regime']} regime, {exp['transform']} transform, {label}, H{exp['horizon']}\n")
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"  created {path}")

print(f"\n==> {len(experiments)} Suite X configs created")
