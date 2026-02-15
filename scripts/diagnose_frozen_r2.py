#!/usr/bin/env python3
"""Diagnose the R2_1step anomaly in the frozen identity experiment."""
import numpy as np
from pathlib import Path

exp = Path('oscar_output/suite_S0_frozen_identity_mvar_p1')
run0 = exp / 'test' / 'test_000'

# Load test density
d = np.load(run0 / 'density_true.npz')
rho = d['rho']
times = d['times']
print(f"rho shape: {rho.shape}, times: [{times[0]:.3f}, {times[-1]:.3f}]")

# Check how much density changes between frames
diffs = np.diff(rho, axis=0)
print(f"Frame diffs: max={np.abs(diffs).max():.6e}, mean={np.abs(diffs).mean():.6e}")
print(f"rho[0] == rho[-1]: {np.allclose(rho[0], rho[-1], atol=1e-10)}")
print(f"Max diff first vs last: {np.abs(rho[0] - rho[-1]).max():.6e}")
print()

# Subsample by 3
rho_sub = rho[::3]
print(f"Subsampled: {rho_sub.shape}")
rho_flat = rho_sub.reshape(len(rho_sub), -1)

# Load POD
pod = np.load(exp / 'rom_common' / 'pod_basis.npz')
U_r = pod['U']
X_mean = np.load(exp / 'rom_common' / 'X_train_mean.npy')

# Project to latent
test_latent = (rho_flat - X_mean) @ U_r
print(f"test_latent shape: {test_latent.shape}")
print(f"test_latent[0]: {test_latent[0]}")
print(f"test_latent[-1]: {test_latent[-1]}")
print(f"Max latent diff consecutive: {np.abs(np.diff(test_latent, axis=0)).max():.6e}")
print(f"Latent std per component: {test_latent.std(axis=0)}")
print()

# T_train computation (from test_evaluator)
T_train = int(0.12 / 0.04 / 3)
print(f"T_train = {T_train}")
n_forecast = len(test_latent) - T_train
print(f"Forecast region: test_latent[{T_train}:] = {n_forecast} steps")
print()

# The targets for R2_1step
targets = test_latent[T_train:]
print(f"targets shape: {targets.shape}")
print(f"targets std per component: {targets.std(axis=0)}")
ss_tot = np.sum((targets - targets.mean(axis=0))**2)
print(f"ss_tot_1s = {ss_tot:.6e}")
print()

# Since A = I and intercept = 0:
# pred for step t_1s: feed window = test_latent[T_train+t_1s-1:T_train+t_1s]
# model predicts: A @ y = I @ y = y  (same as input)
# So pred[t_1s] = test_latent[T_train + t_1s - 1]
# target[t_1s] = test_latent[T_train + t_1s]
# For frozen sim: all latent states are identical
# => pred == target (both the same constant)
# => ss_res = 0, ss_tot = 0 => R2 = 0/0

# BUT: let's check if density is ACTUALLY constant
print("=== DENSITY CONSTANCY CHECK ===")
for i in range(min(5, len(rho_sub))):
    for j in range(i+1, min(5, len(rho_sub))):
        diff = np.abs(rho_flat[i] - rho_flat[j]).max()
        print(f"  max |rho[{i}] - rho[{j}]| = {diff:.6e}")
print()

# Manual 1-step: simulate exactly what test_evaluator does
P_LAG = 1
mvar = np.load(exp / 'MVAR' / 'mvar_model.npz', allow_pickle=True)
A = mvar['A_matrices']  # (1, 7, 7)
# No intercept saved, check if Ridge model has one
# Ridge with fit_intercept=True stores an intercept
# But mvar_model.npz stores A_matrices which are the weight matrices
# The forecast_fn uses mvar_model.predict() which includes the intercept
# Let's see what happens with just A @ y

true_latent_forecast_1s = test_latent[T_train:]
onestep_preds = []
for t_1s in range(len(true_latent_forecast_1s)):
    window_end = T_train + t_1s
    if window_end < P_LAG:
        continue
    true_window = test_latent[window_end - P_LAG:window_end]
    # What forecast_fn does: x_hist = true_window.flatten(), pred = model.predict(x_hist)
    # Which is: pred = A[0] @ x_hist + intercept_from_Ridge
    # With frozen data, Ridge learned: y(t) = A*y(t-1) + b
    # All y identical => y = A*y + b => (I-A)y = b
    # If A=I => b = 0. So pred = y. 
    # But Ridge with regularization may give slightly different A...
    
    pred_1s = A[0] @ true_window[0]  # no intercept (it was 0)
    onestep_preds.append(pred_1s)

onestep_preds = np.array(onestep_preds)
onestep_targets = true_latent_forecast_1s[:len(onestep_preds)]

print(f"onestep_preds shape: {onestep_preds.shape}")
print(f"onestep_targets shape: {onestep_targets.shape}")
print()

residuals = onestep_targets - onestep_preds
print(f"Residuals: max_abs={np.abs(residuals).max():.6e}, mean_abs={np.abs(residuals).mean():.6e}")
print()

ss_res_1s = np.sum(residuals**2)
ss_tot_1s = np.sum((onestep_targets - onestep_targets.mean(axis=0))**2)
print(f"ss_res_1s = {ss_res_1s:.6e}")
print(f"ss_tot_1s = {ss_tot_1s:.6e}")
if ss_tot_1s > 0:
    ratio = ss_res_1s / ss_tot_1s
    r2 = 1 - ratio
    print(f"ratio = {ratio:.6e}")
    print(f"R2_1step = {r2:.6e}")
else:
    print("ss_tot_1s = 0 => R2 undefined!")
print()

# ===== KEY QUESTION: is the ACTUAL code path different? =====
# The actual test_evaluator uses forecast_fn which calls sklearn Ridge.predict()
# Ridge.predict includes intercept. Let's check if there's an intercept issue.
print("=== CHECKING ACTUAL CODE PATH ===")
print(f"Ridge alpha from MVAR: {mvar['alpha']}")
print(f"Train R2: {mvar['train_r2']}")
print(f"A[0] - I (should be ~0): max_abs={np.abs(A[0] - np.eye(7)).max():.6e}")
print(f"rho_before from file: {mvar['rho_before']}")
print()

# The REAL issue: onestep_targets alignment!
# t_1s=0: window_end=T_train=1, window=test_latent[0:1], pred=model(y(0)), this predicts y(1)
#          BUT onestep_targets[0] = true_latent_forecast_1s[0] = test_latent[T_train] = test_latent[1]
#          So target IS y(1). Alignment is CORRECT.
# 
# For frozen sim: y(0) = y(1) = ... = const
# pred = A @ y(0) = I @ y(0) = y(0) = y(1) = target
# => residual = 0, ss_res = 0
# BUT ss_tot = 0 too (all targets identical)
# => R2 = 1 - 0/0 = undefined!

# Actually ss_tot uses targets.mean(axis=0), which IS the constant value
# So ss_tot = sum of (target - mean)^2 = 0 exactly (or near-zero with float noise)
# And the code does: r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0

# With float noise: ss_tot might be e.g. 1e-28, ss_res might be 1e-26
# => ratio = 100 => R2 = -99

# BUT the actual reported R2_1step is -2.5e9 to -1.6e10!
# That means ss_res >> ss_tot by a factor of ~1e10

# This could happen if the Ridge model is actually slightly different from I
# (due to regularization) and the predicted values drift from the constant

print("=== ACTUAL SCENARIO ANALYSIS ===")
print(f"All targets identical? std={onestep_targets.std():.6e}")
print(f"Target mean per dim: {onestep_targets.mean(axis=0)}")
target_centered = onestep_targets - onestep_targets.mean(axis=0)
print(f"Target centered max abs: {np.abs(target_centered).max():.6e}")
print()

# Check if test density actually varies due to KDE numerical issues
print("=== KDE NUMERICAL CHECK ===")
traj = np.load(run0 / 'trajectory.npz')
positions = traj['traj']  # (126, 200, 2)
velocities = traj['vel']  # (126, 200, 2)
print(f"Positions shape: {positions.shape}")
print(f"Velocities range: [{velocities.min():.6e}, {velocities.max():.6e}]")
print(f"Position diff between frames: {np.abs(np.diff(positions, axis=0)).max():.6e}")
# If positions are truly frozen, then KDE should give identical density at each frame
# But floating point in KDE computation might introduce tiny variations
