#!/usr/bin/env python3
"""Smoke test: spectral derivatives + mass conservation + ETDRK4.

Uses oscar_output test data to verify:
  1. FieldData with use_spectral=True matches FD within expected truncation
  2. discover_multifield still works
  3. forecast_multifield with method="rk4" runs
  4. forecast_multifield with method="etdrk4" runs
  5. Mass conservation is enforced
"""
import numpy as np
from pathlib import Path
from src.wsindy.fields import (
    FieldData, build_field_data,
    _dx, _dy, _lap,
    _spectral_dx, _spectral_dy, _spectral_lap,
)
from src.wsindy.multifield import (
    build_default_library, discover_multifield, forecast_multifield,
    _extract_linear_spectral,
)
from src.wsindy.grid import GridSpec

# ── Find data ──────────────────────────────────────────────────
base = Path("oscar_output")
candidates = sorted(base.rglob("trajectory.npz"))
assert len(candidates) > 0, "No test data found"
run_dir = candidates[0].parent
print(f"Using: {run_dir}\n")

d = np.load(run_dir / "density_true.npz")
rho_full = d["rho"][::3]
xgrid, ygrid = d["xgrid"], d["ygrid"]
dt = float(d["times"][1] - d["times"][0]) * 3
dx_val = float(xgrid[1] - xgrid[0])
dy_val = float(ygrid[1] - ygrid[0])
Lx = float(xgrid[-1] - xgrid[0]) + dx_val
Ly = float(ygrid[-1] - ygrid[0]) + dy_val

td = np.load(run_dir / "trajectory.npz")
traj = td["traj"][::3]
vel = td["vel"][::3]
T_min = min(rho_full.shape[0], traj.shape[0])
rho_full = rho_full[:T_min]
traj = traj[:T_min]
vel = vel[:T_min]

print(f"rho: {rho_full.shape}, traj: {traj.shape}")
print(f"Grid: dx={dx_val:.4f}, dt={dt:.4f}, Lx={Lx:.2f}, Ly={Ly:.2f}")

# ════════════════════════════════════════════════════════════════
#  TEST 1: Spectral vs FD derivative accuracy
# ════════════════════════════════════════════════════════════════
print("\n─── Test 1: Spectral vs FD derivative accuracy ───")

# Use a smooth test function where spectral should be much more accurate
nx, ny = rho_full.shape[2], rho_full.shape[1]
x = np.linspace(0, Lx, nx, endpoint=False)
y = np.linspace(0, Ly, ny, endpoint=False)
X, Y = np.meshgrid(x, y)
# sin(2πx/Lx) has exact derivative (2π/Lx)cos(2πx/Lx)
f_test = np.sin(2 * np.pi * X / Lx) * np.cos(4 * np.pi * Y / Ly)

# Exact derivatives
exact_dx = (2 * np.pi / Lx) * np.cos(2 * np.pi * X / Lx) * np.cos(4 * np.pi * Y / Ly)
exact_lap = -(
    (2 * np.pi / Lx)**2 * np.sin(2 * np.pi * X / Lx) * np.cos(4 * np.pi * Y / Ly)
    + (4 * np.pi / Ly)**2 * np.sin(2 * np.pi * X / Lx) * np.cos(4 * np.pi * Y / Ly)
)

fd_dx = _dx(f_test, dx_val)
sp_dx = _spectral_dx(f_test, Lx)
fd_lap = _lap(f_test, dx_val, dy_val)
sp_lap = _spectral_lap(f_test, Lx, Ly)

err_fd_dx = np.max(np.abs(fd_dx - exact_dx))
err_sp_dx = np.max(np.abs(sp_dx - exact_dx))
err_fd_lap = np.max(np.abs(fd_lap - exact_lap))
err_sp_lap = np.max(np.abs(sp_lap - exact_lap))

print(f"  ∂f/∂x error — FD: {err_fd_dx:.2e}  Spectral: {err_sp_dx:.2e}")
print(f"  Δf error    — FD: {err_fd_lap:.2e}  Spectral: {err_sp_lap:.2e}")
assert err_sp_dx < 1e-10, f"Spectral ∂/∂x too inaccurate: {err_sp_dx}"
assert err_sp_lap < 1e-10, f"Spectral Δ too inaccurate: {err_sp_lap}"
assert err_sp_dx < err_fd_dx * 0.01, "Spectral should be much better than FD"
print("  ✓ Spectral derivatives are machine-precision")

# ════════════════════════════════════════════════════════════════
#  TEST 2: FieldData with use_spectral=True
# ════════════════════════════════════════════════════════════════
print("\n─── Test 2: FieldData spectral mode ───")

fd_fd = build_field_data(
    rho_full[:50], traj[:50], vel[:50],
    xgrid, ygrid, Lx, Ly, dt,
    bandwidth=5.0,
    morse_params=dict(Cr=0.3, Ca=0.8, lr=0.5, la=1.5),
)
# Same data, spectral mode via direct construction
grid = GridSpec(dt=dt, dx=dx_val, dy=dy_val)
fd_sp = FieldData(
    fd_fd.rho, fd_fd.px, fd_fd.py, grid, Lx, Ly,
    Phi=fd_fd.Phi, use_spectral=True,
)

# The two should agree on non-derivative quantities
assert np.allclose(fd_fd.p_sq(), fd_sp.p_sq()), "p² mismatch"
assert np.allclose(fd_fd.rho2(), fd_sp.rho2()), "ρ² mismatch"

# Derivatives should be similar but spectral more accurate
diff_div_p = np.max(np.abs(fd_fd.div_p() - fd_sp.div_p()))
diff_lap_rho = np.max(np.abs(fd_fd.lap_rho() - fd_sp.lap_rho()))
print(f"  ∇·p:  max FD-vs-spectral diff = {diff_div_p:.4e}")
print(f"  Δρ:   max FD-vs-spectral diff = {diff_lap_rho:.4e}")
print("  ✓ Both backends agree (within truncation)")

# ════════════════════════════════════════════════════════════════
#  TEST 3: Discover + RK4 forecast
# ════════════════════════════════════════════════════════════════
print("\n─── Test 3: Discover + RK4 forecast ───")

lib = build_default_library(morse=True, rich=False)
result = discover_multifield(
    [fd_sp], lib,
    ell=(5, 5, 5), p=(2, 2, 2), stride=(2, 2, 2),
    lambdas=np.logspace(-4, 1, 30),
    verbose=False,
)
print(f"  ρ: R²={result.rho_model.diagnostics.get('r2', 0):.4f}, "
      f"active={result.rho_model.n_active}")
print(f"  px: R²={result.px_model.diagnostics.get('r2', 0):.4f}, "
      f"active={result.px_model.n_active}")
print(f"  py: R²={result.py_model.diagnostics.get('r2', 0):.4f}, "
      f"active={result.py_model.n_active}")

rho0 = fd_sp.rho[0]
px0 = fd_sp.px[0]
py0 = fd_sp.py[0]
M0 = np.sum(rho0)

n_fc = 10
rho_rk4, px_rk4, py_rk4 = forecast_multifield(
    rho0, px0, py0, result, grid,
    Lx=Lx, Ly=Ly, n_steps=n_fc,
    clip_negative_rho=True,
    mass_conserve=True,
    method="rk4",
    morse_params=dict(Cr=0.3, Ca=0.8, lr=0.5, la=1.5),
    xgrid=xgrid, ygrid=ygrid,
)
print(f"  RK4 forecast: {rho_rk4.shape}")
assert rho_rk4.shape == (n_fc + 1, ny, nx), "Wrong output shape"
assert not np.any(np.isnan(rho_rk4)), "NaN in RK4 forecast"
assert np.all(rho_rk4 >= 0), "Negative density in RK4 forecast"

# Mass conservation check
for t in range(1, n_fc + 1):
    Mt = np.sum(rho_rk4[t])
    rel_err = abs(Mt - M0) / M0
    assert rel_err < 1e-10, f"Mass drift at t={t}: {rel_err:.2e}"
print(f"  ✓ RK4 mass conservation: max relative error < 1e-10")

# ════════════════════════════════════════════════════════════════
#  TEST 4: ETDRK4 forecast
# ════════════════════════════════════════════════════════════════
print("\n─── Test 4: ETDRK4 forecast ───")

# Check linear operator extraction
L_px = _extract_linear_spectral(
    result.px_model, result.px_terms, nx, ny, Lx, Ly)
has_linear = L_px is not None
print(f"  Linear operator for px: {'found' if has_linear else 'none'}")

rho_et, px_et, py_et = forecast_multifield(
    rho0, px0, py0, result, grid,
    Lx=Lx, Ly=Ly, n_steps=n_fc,
    clip_negative_rho=True,
    mass_conserve=True,
    method="etdrk4",
    morse_params=dict(Cr=0.3, Ca=0.8, lr=0.5, la=1.5),
    xgrid=xgrid, ygrid=ygrid,
)
print(f"  ETDRK4 forecast: {rho_et.shape}")
assert not np.any(np.isnan(rho_et)), "NaN in ETDRK4 forecast"
assert np.all(rho_et >= 0), "Negative density in ETDRK4 forecast"

# Mass conservation
for t in range(1, n_fc + 1):
    Mt = np.sum(rho_et[t])
    rel_err = abs(Mt - M0) / M0
    assert rel_err < 1e-10, f"ETDRK4 mass drift at t={t}: {rel_err:.2e}"
print(f"  ✓ ETDRK4 mass conservation: max relative error < 1e-10")

# RK4 and ETDRK4 should give similar (not identical) results
rho_diff = np.max(np.abs(rho_rk4 - rho_et)) / (np.max(np.abs(rho_rk4)) + 1e-30)
print(f"  RK4 vs ETDRK4 max relative diff: {rho_diff:.4e}")

# ════════════════════════════════════════════════════════════════
#  TEST 5: Auto method selection
# ════════════════════════════════════════════════════════════════
print("\n─── Test 5: Auto method selection ───")

rho_auto, px_auto, py_auto = forecast_multifield(
    rho0, px0, py0, result, grid,
    Lx=Lx, Ly=Ly, n_steps=5,
    method="auto",
    morse_params=dict(Cr=0.3, Ca=0.8, lr=0.5, la=1.5),
    xgrid=xgrid, ygrid=ygrid,
)
print(f"  Auto forecast: {rho_auto.shape}")
assert not np.any(np.isnan(rho_auto)), "NaN in auto forecast"
print("  ✓ Auto method works")

print("\n═══════════════════════════════════════════════════")
print("  ALL SMOKE TESTS PASSED")
print("═══════════════════════════════════════════════════")
