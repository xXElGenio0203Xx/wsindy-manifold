#!/usr/bin/env python3
"""Verify local alignment pipeline outputs."""

import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("oscar_output/local_align_verify")

# 1. POD basis
pod = np.load(OUT / "mvar/pod_basis.npz", allow_pickle=True)
print("=== POD basis ===")
for k in pod.files:
    v = pod[k]
    print(f"  {k}: {v.shape if hasattr(v, 'shape') else v}")

# 2. Shift alignment data
sa = np.load(OUT / "mvar/shift_align.npz", allow_pickle=True)
print("\n=== Shift alignment data ===")
for k in sa.files:
    v = sa[k]
    print(f"  {k}: {v.shape if hasattr(v, 'shape') else v}")

# 3. MVAR model
mvar = np.load(OUT / "mvar/mvar_model.npz", allow_pickle=True)
print("\n=== MVAR model ===")
for k in mvar.files:
    v = mvar[k]
    print(f"  {k}: {v.shape if hasattr(v, 'shape') else v}")

# 4. Test results
df = pd.read_csv(OUT / "test/test_results.csv")
print("\n=== Test results ===")
cols = ["test_idx", "r2_reconstructed", "r2_latent", "r2_pod", "negativity_frac"]
cols = [c for c in cols if c in df.columns]
print(df[cols].to_string(index=False))

# 5. Check test predictions
for i in range(3):
    pred_path = OUT / f"test/test_{i:03d}/density_pred.npz"
    true_path = OUT / f"test/test_{i:03d}/density_true.npz"
    if not pred_path.exists():
        print(f"\ntest_{i:03d}: NOT FOUND")
        continue
    pred = np.load(pred_path, allow_pickle=True)
    true_data = np.load(true_path, allow_pickle=True)
    rho_pred = pred["rho"]
    rho_true = true_data["rho"]
    fs_idx = int(pred["forecast_start_idx"])
    pred_mass = rho_pred.sum(axis=(1, 2))
    true_mass = rho_true.sum(axis=(1, 2))
    print(f"\ntest_{i:03d}: pred shape={rho_pred.shape}  true shape={rho_true.shape}")
    print(f"  forecast_start_idx = {fs_idx}")
    print(f"  pred mass range: [{pred_mass.min():.1f}, {pred_mass.max():.1f}]")
    print(f"  true mass range: [{true_mass.min():.1f}, {true_mass.max():.1f}]")
    
    # Manual R² on forecast region
    # True density uses original dt; pred uses subsampled dt
    # Subsample true to match pred timesteps
    rom_sub = 3
    true_sub = rho_true[::rom_sub]  # subsample to ROM timebase
    fp = rho_pred[fs_idx:]
    # T_train in ROM steps
    T_train_rom = fs_idx
    ft = true_sub[T_train_rom:T_train_rom + len(fp)]
    print(f"  Forecast comparison: pred {fp.shape} vs true {ft.shape}")
    ss_res = np.sum((ft.flatten() - fp.flatten()) ** 2)
    ss_tot = np.sum((ft.flatten() - ft.flatten().mean()) ** 2)
    r2_manual = 1.0 - ss_res / ss_tot
    print(f"  Manual R² (forecast-only): {r2_manual:.4f}")
    
    # Negative pixel check
    neg_frac = (fp < 0).sum() / fp.size
    print(f"  Negative pixels in forecast: {neg_frac:.2%}")

# 6. Time-resolved R²
r2t_path = OUT / "test/test_000/r2_vs_time.csv"
if r2t_path.exists():
    r2t = pd.read_csv(r2t_path)
    print(f"\n=== R² vs time (test_000): {len(r2t)} rows ===")
    print(f"  t range: [{r2t['time'].iloc[0]:.2f}, {r2t['time'].iloc[-1]:.2f}]s")
    if "r2_snapshot" in r2t.columns:
        print(f"  R² start: {r2t['r2_snapshot'].iloc[0]:.4f}")
        print(f"  R² end:   {r2t['r2_snapshot'].iloc[-1]:.4f}")
    else:
        print(f"  Columns: {list(r2t.columns)}")
        print(r2t.head())

print("\n=== PIPELINE VERIFICATION COMPLETE ===")
