#!/usr/bin/env python3
"""Analyze why R² lifted >> R² latent despite high POD ceiling.

Also analyze the mass drop at the start of CUR3 forecasts.
"""
import numpy as np
import sys

experiments = [
    'CUR1_N500_fast_aligned_sqrtSimplex_H300',
    'CUR3_N500_fast_aligned_sqrtSimplex_H300_highNoise',
]

# ========== PART 1: Latent vs Lifted R² gap ==========
print("=" * 72)
print("  PART 1: Why R² lifted >> R² latent")
print("=" * 72)

for name in experiments:
    label = name.split('_')[0]
    print(f"\n{'─' * 72}")
    print(f"  {label}")
    print(f"{'─' * 72}")

    pod = np.load(f'oscar_output/{name}/rom_common/pod_basis.npz')
    U = pod['U']
    sv = pod['singular_values']
    X_mean = np.load(f'oscar_output/{name}/rom_common/X_train_mean.npy')

    d = U.shape[1]
    G = U.shape[0]

    # Load test_000
    true_d = np.load(f'oscar_output/{name}/test/test_000/density_true.npz')
    pred_d = np.load(f'oscar_output/{name}/test/test_000/density_pred_mvar.npz')
    rho_true_all = true_d['rho']
    rho_pred_all = pred_d['rho']

    T_pred = rho_pred_all.shape[0]
    rho_true = rho_true_all[-T_pred:]

    # sqrt transform
    true_sqrt = np.sqrt(np.maximum(rho_true, 0))
    pred_sqrt = np.sqrt(np.maximum(rho_pred_all, 0))

    # Flatten to [T, G]
    true_flat = true_sqrt.reshape(-1, G)
    pred_flat = pred_sqrt.reshape(-1, G)

    # Project to latent
    true_lat = (true_flat - X_mean) @ U
    pred_lat = (pred_flat - X_mean) @ U

    # --- Per-mode R² ---
    total_energy = np.sum(sv ** 2)
    print(f"\n  Per-mode R² in latent space:")
    print(f"  {'Mode':>5s}  {'R²':>8s}  {'Energy%':>8s}  {'SV':>8s}")
    print(f"  {'─' * 40}")

    r2_modes = []
    for i in range(d):
        ss_res = np.sum((true_lat[:, i] - pred_lat[:, i]) ** 2)
        ss_tot = np.sum((true_lat[:, i] - true_lat[:, i].mean()) ** 2)
        r2_i = 1 - ss_res / ss_tot if ss_tot > 1e-10 else float('nan')
        w_i = sv[i] ** 2 / total_energy * 100
        r2_modes.append(r2_i)
        print(f"  {i:5d}  {r2_i:+8.4f}  {w_i:7.2f}%  {sv[i]:8.1f}")

    r2_modes = np.array(r2_modes)
    weights = sv ** 2 / total_energy

    # --- Key metrics ---
    # SS_res is preserved by orthogonality: ||Ua - Ub||² = ||a - b||²
    lat_ss_res = np.sum((true_lat - pred_lat) ** 2)
    phys_ss_res = np.sum((true_flat - pred_flat) ** 2)

    # SS_tot is NOT preserved — different centering
    lat_ss_tot = np.sum((true_lat.flatten() - true_lat.flatten().mean()) ** 2)
    phys_ss_tot = np.sum((true_flat - true_flat.mean()) ** 2)

    # Also compute per-mode SS_tot (sum across modes)
    lat_ss_tot_permode = sum(
        np.sum((true_lat[:, i] - true_lat[:, i].mean()) ** 2)
        for i in range(d)
    )

    r2_flat_lat = 1 - lat_ss_res / lat_ss_tot
    r2_flat_phys = 1 - phys_ss_res / phys_ss_tot
    r2_energy_wt = np.sum(r2_modes * weights)

    # The residual in physical space includes out-of-subspace component
    # pred is lifted through POD so it stays in subspace
    # but true has components outside the d-dim subspace
    # So phys_ss_res = lat_ss_res + out-of-subspace error (from true)
    # Actually no: both true_flat and pred_flat are full G-dim vectors.
    # pred_flat = pred_lat @ U.T + X_mean (stays in subspace)
    # true_flat has components outside the subspace.
    # But we compute phys_ss_res = ||true_flat - pred_flat||²
    #   = || (component in subspace) + (component outside) ||²
    # The in-subspace part = lat_ss_res (by Parseval)
    # The out-of-subspace part = truncation error
    # These are orthogonal, so phys_ss_res = lat_ss_res + truncation_error
    # This makes phys_ss_res >= lat_ss_res, which HURTS R² lifted.
    # Yet R² lifted is BETTER. Why? Because phys_ss_tot is much bigger.

    # Verify Parseval:
    # In-subspace residual
    resid_insub = ((true_flat - pred_flat) @ U) # project residual to latent
    ss_res_insub = np.sum(resid_insub ** 2)
    ss_res_outsub = phys_ss_res - ss_res_insub

    print(f"\n  Key quantities:")
    print(f"    SS_res (latent flat):     {lat_ss_res:14.1f}")
    print(f"    SS_res (physical flat):   {phys_ss_res:14.1f}")
    print(f"      in-subspace component:  {ss_res_insub:14.1f}  (≈ lat_ss_res)")
    print(f"      out-of-subspace:        {ss_res_outsub:14.1f}  (truncation error)")
    print(f"    SS_tot (latent flat):     {lat_ss_tot:14.1f}")
    print(f"    SS_tot (physical flat):   {phys_ss_tot:14.1f}")
    print(f"    Ratio SS_tot phys/lat:    {phys_ss_tot / lat_ss_tot:14.2f}x")
    print(f"    Ratio SS_res phys/lat:    {phys_ss_res / lat_ss_res:14.2f}x")
    print()
    print(f"  R² metrics:")
    print(f"    R² latent (flat):         {r2_flat_lat:+.4f}")
    print(f"    R² latent (energy-wt):    {r2_energy_wt:+.4f}")
    print(f"    R² lifted (sqrt-density): {r2_flat_phys:+.4f}")
    print()
    print(f"  >>> The gap comes from SS_tot being {phys_ss_tot / lat_ss_tot:.1f}x larger")
    print(f"      in physical space while SS_res only grows {phys_ss_res / lat_ss_res:.1f}x.")
    print()
    print(f"  WHY SS_tot is bigger in physical space:")
    print(f"    Latent SS_tot uses a GLOBAL scalar mean across all {d} modes × {T_pred} timesteps.")
    print(f"    Physical SS_tot uses a GLOBAL scalar mean across {G} grid cells × {T_pred} timesteps.")
    print(f"    The mean structure (X_mean) contributes huge variance in grid space")
    print(f"    because cells far from the mean density vary a lot — but that variance")
    print(f"    is already captured by the POD basis, inflating the denominator.")
    print()
    
    # The real insight: SS_tot(latent,flat) uses mean over ALL d*T values.
    # Most latent coefficients are near zero (centered by construction).
    # The global mean ≈ 0, so SS_tot ≈ sum of variances.
    # But SS_tot(physical,flat) mean is the AVERAGE cell value over time,
    # which is ~mean density. Cells that are far from mean density
    # (e.g., in clusters) contribute huge (value - mean)^2 to SS_tot.
    
    lat_global_mean = true_lat.flatten().mean()
    phys_global_mean = true_flat.flatten().mean()
    print(f"    Latent global mean:  {lat_global_mean:.4f}  (near 0, by POD centering)")
    print(f"    Physical global mean: {phys_global_mean:.4f}")
    

# ========== PART 2: CUR3 Mass Drop Analysis ==========
print("\n\n")
print("=" * 72)
print("  PART 2: CUR3 Mass Drop at Forecast Start")
print("=" * 72)

name = 'CUR3_N500_fast_aligned_sqrtSimplex_H300_highNoise'
pod = np.load(f'oscar_output/{name}/rom_common/pod_basis.npz')
U = pod['U']
sv = pod['singular_values']
X_mean = np.load(f'oscar_output/{name}/rom_common/X_train_mean.npy')
d = U.shape[1]
G = U.shape[0]

# Check several test runs
import os
test_dir = f'oscar_output/{name}/test'
test_runs = sorted([x for x in os.listdir(test_dir) if x.startswith('test_')])[:5]

for run_name in test_runs:
    run_dir = f'{test_dir}/{run_name}'
    true_d = np.load(f'{run_dir}/density_true.npz')
    pred_d = np.load(f'{run_dir}/density_pred_mvar.npz')

    rho_true_all = true_d['rho']
    rho_pred_all = pred_d['rho']
    T_pred = rho_pred_all.shape[0]
    rho_true = rho_true_all[-T_pred:]

    # Mass = sum of density over grid
    mass_true = rho_true.sum(axis=(1, 2))
    mass_pred = rho_pred_all.sum(axis=(1, 2))

    # Also check in sqrt space
    sqrt_true = np.sqrt(np.maximum(rho_true, 0))
    sqrt_pred = np.sqrt(np.maximum(rho_pred_all, 0))
    mass_sqrt_true = sqrt_true.sum(axis=(1, 2))
    mass_sqrt_pred = sqrt_pred.sum(axis=(1, 2))

    # Negative fraction
    neg_frac = np.array([(rho_pred_all[t] < 0).sum() / rho_pred_all[t].size 
                         for t in range(T_pred)])

    print(f"\n  {run_name}:")
    print(f"    True mass:  start={mass_true[0]:.1f}  end={mass_true[-1]:.1f}  "
          f"range=[{mass_true.min():.1f}, {mass_true.max():.1f}]")
    print(f"    Pred mass:  start={mass_pred[0]:.1f}  end={mass_pred[-1]:.1f}  "
          f"range=[{mass_pred.min():.1f}, {mass_pred.max():.1f}]")
    print(f"    Mass ratio pred/true at t=0: {mass_pred[0] / mass_true[0]:.4f}")
    print(f"    Mass ratio pred/true at t=5: {mass_pred[5] / mass_true[5]:.4f}")
    print(f"    Neg cells at t=0: {neg_frac[0]*100:.2f}%  t=5: {neg_frac[5]*100:.2f}%  "
          f"max: {neg_frac.max()*100:.2f}%")

    # Check the conditioning→forecast transition
    # The last conditioning frame's density vs first forecast frame's pred
    if T_pred < rho_true_all.shape[0]:
        T_cond = rho_true_all.shape[0] - T_pred
        last_cond_true = rho_true_all[T_cond - 1]
        first_forecast_true = rho_true_all[T_cond]
        first_forecast_pred = rho_pred_all[0]

        mass_last_cond = last_cond_true.sum()
        mass_first_true = first_forecast_true.sum()
        mass_first_pred = first_forecast_pred.sum()

        print(f"    Last cond mass:    {mass_last_cond:.1f}")
        print(f"    First true mass:   {mass_first_true:.1f}")
        print(f"    First pred mass:   {mass_first_pred:.1f}")
        print(f"    Jump pred-cond:    {mass_first_pred - mass_last_cond:+.1f} "
              f"({(mass_first_pred - mass_last_cond) / mass_last_cond * 100:+.2f}%)")

        # sqrt-space analysis at transition 
        sqrt_last_cond = np.sqrt(np.maximum(last_cond_true, 0))
        lat_last_cond = (sqrt_last_cond.reshape(1, -1) - X_mean) @ U  # [1, d]
        
        sqrt_first_pred = np.sqrt(np.maximum(first_forecast_pred, 0))
        lat_first_pred = (sqrt_first_pred.reshape(1, -1) - X_mean) @ U

        sqrt_first_true = np.sqrt(np.maximum(first_forecast_true, 0))
        lat_first_true = (sqrt_first_true.reshape(1, -1) - X_mean) @ U

        # Reconstruction error analysis at t=0
        # The prediction at t=0 comes from MVAR forecasting in latent space
        # then lifting back. Check what happens at lifting.
        # What does MVAR predict in latent? Load from mvar output
        print(f"    Latent |true-pred| at t=0: {np.linalg.norm(lat_first_true - lat_first_pred):.4f}")
        print(f"    Latent |true-pred| at t=0 mode-0: {abs(lat_first_true[0,0] - lat_first_pred[0,0]):.4f}")

# Now do the detailed mass timeseries for CUR3 best run
print(f"\n{'─' * 72}")
print(f"  CUR3 Mass Timeseries (test_002 = best run, R²=0.986)")
print(f"{'─' * 72}")

run_dir = f'oscar_output/{name}/test/test_002'
true_d = np.load(f'{run_dir}/density_true.npz')
pred_d = np.load(f'{run_dir}/density_pred_mvar.npz')
rho_true_all = true_d['rho']
rho_pred_all = pred_d['rho']
T_pred = rho_pred_all.shape[0]
rho_true = rho_true_all[-T_pred:]

mass_true = rho_true.sum(axis=(1, 2))
mass_pred = rho_pred_all.sum(axis=(1, 2))
neg_frac = np.array([(rho_pred_all[t] < 0).sum() / rho_pred_all[t].size for t in range(T_pred)])

# Print every 10th timestep
print(f"\n  {'t':>5s}  {'M_true':>10s}  {'M_pred':>10s}  {'ratio':>8s}  {'M_err%':>8s}  {'neg%':>6s}")
print(f"  {'─' * 55}")
for t in list(range(0, min(21, T_pred))) + list(range(30, T_pred, 30)):
    if t >= T_pred:
        break
    ratio = mass_pred[t] / mass_true[t]
    err_pct = (mass_pred[t] - mass_true[t]) / mass_true[t] * 100
    print(f"  {t:5d}  {mass_true[t]:10.1f}  {mass_pred[t]:10.1f}  {ratio:8.4f}  {err_pct:+7.2f}%  {neg_frac[t]*100:5.2f}%")

print(f"\n  Max neg frac: {neg_frac.max()*100:.2f}% at t={neg_frac.argmax()}")
print(f"  Min mass ratio: {(mass_pred/mass_true).min():.4f} at t={(mass_pred/mass_true).argmin()}")
print(f"  Max mass ratio: {(mass_pred/mass_true).max():.4f} at t={(mass_pred/mass_true).argmax()}")
