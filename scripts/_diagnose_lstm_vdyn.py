#!/usr/bin/env python3
"""Diagnose LSTM failure in VDYN suite."""
import pandas as pd, os, numpy as np

exps = ['VDYN1_gentle','VDYN2_hypervelocity','VDYN3_hypernoisy',
        'VDYN4_blackhole','VDYN5_supernova','VDYN6_baseline','VDYN7_pure_vicsek']
ROOT = "oscar_output"

print("=" * 90)
print("PART 1: LSTM TRAINING LOG ANALYSIS")
print("=" * 90)
for e in exps:
    p = os.path.join(ROOT, f"{e}_varspeed", "LSTM", "training_log.csv")
    if not os.path.isfile(p):
        continue
    df = pd.read_csv(p)
    cols = df.columns.tolist()
    print(f"\n--- {e} ({len(df)} epochs) ---")
    print(f"  Columns: {cols}")
    if 'train_loss' in cols and 'val_loss' in cols:
        final = df.iloc[-1]
        best_idx = df['val_loss'].idxmin()
        best = df.iloc[best_idx]
        print(f"  Best val_loss: {best['val_loss']:.6f} at epoch {int(best.get('epoch', best_idx))}")
        print(f"  Final: train={final['train_loss']:.6f} val={final['val_loss']:.6f}")
        ratio = final['val_loss'] / max(final['train_loss'], 1e-10)
        print(f"  Overfit ratio (val/train): {ratio:.2f}")
        # Check if early-stopped
        print(f"  Stopped at epoch {len(df)} (max 200 => early_stop={len(df) < 200})")
    elif 'loss' in cols:
        print(f"  Single loss column — final: {df.iloc[-1]['loss']:.6f}")

print("\n\n" + "=" * 90)
print("PART 2: PER-TEST R2 DISTRIBUTION (reconstructed)")
print("=" * 90)
for e in exps:
    for method in ["MVAR", "LSTM"]:
        p = os.path.join(ROOT, f"{e}_varspeed", method, "test_results.csv")
        if not os.path.isfile(p):
            continue
        df = pd.read_csv(p)
        r2 = df['r2_reconstructed']
        neg_frac = df['negativity_frac']
        r2_1step = df['r2_1step']
        n_negative_r2 = (r2 < 0).sum()
        n_catastrophic = (r2 < -1).sum()
        print(f"  {e:30s} {method:5s}  "
              f"median={r2.median():.3f}  mean={r2.mean():.3f}  "
              f"min={r2.min():.2f}  max={r2.max():.2f}  "
              f"neg_r2={n_negative_r2}/20  cata_r2={n_catastrophic}/20  "
              f"1step_mean={r2_1step.mean():.3f}  neg%_mean={neg_frac.mean():.1f}")

print("\n\n" + "=" * 90)
print("PART 3: COMPARE WITH DYN (fixed speed) RESULTS IF AVAILABLE")
print("=" * 90)
dyn_exps = ['DYN1_gentle_v2', 'DYN2_hypervelocity_v2', 'DYN3_hypernoisy_v2',
            'DYN4_blackhole_v2', 'DYN5_supernova', 'DYN6_varspeed_v2', 'DYN7_pure_vicsek']
found = 0
for e in dyn_exps:
    for method in ["MVAR", "LSTM"]:
        p = os.path.join(ROOT, e, method, "test_results.csv")
        if os.path.isfile(p):
            df = pd.read_csv(p)
            r2 = df['r2_reconstructed']
            r2_1s = df['r2_1step']
            neg = df['negativity_frac']
            print(f"  {e:30s} {method:5s}  "
                  f"mean_r2={r2.mean():.3f}  1step={r2_1s.mean():.3f}  neg%={neg.mean():.1f}")
            found += 1
if found == 0:
    print("  No DYN fixed-speed results found locally.")

print("\n\n" + "=" * 90)
print("PART 4: FORECAST HORIZON ANALYSIS")
print("=" * 90)
for e in exps[:2]:  # Just check first 2
    cfg_path = os.path.join(ROOT, f"{e}_varspeed", "config_used.yaml")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            lines = f.readlines()
        # Extract key params
        for line in lines:
            for key in ['T:', 'dt:', 'forecast_start:', 'lag:', 'subsample:']:
                if key in line and not line.strip().startswith('#'):
                    print(f"  {e}: {line.strip()}")

# Compute forecast steps
print(f"\n  Forecast calculation:")
print(f"    Test sim: T=50s, dt=0.04, subsample=3 => {int(50/0.04/3)} POD time steps")
print(f"    Training: T=20s => {int(20/0.04/3)} train steps")
print(f"    forecast_start=0.60 => conditioning from step 0 to {int(50/0.04/3 * 0.60)}")
print(f"    Forecast horizon: {int(50/0.04/3) - int(50/0.04/3 * 0.60)} autoregressive steps")
print(f"    With lag=5, each step uses 5 previous to predict next")

print("\n\n" + "=" * 90)
print("PART 5: R2_LATENT vs R2_RECON DISCONNECT")
print("=" * 90)
print("  If r2_latent > r2_recon: error amplified by POD reconstruction")
print("  If r2_latent < r2_recon: POD smoothing helps despite latent errors")
print()
for e in exps:
    for method in ["MVAR", "LSTM"]:
        p = os.path.join(ROOT, f"{e}_varspeed", method, "test_results.csv")
        if not os.path.isfile(p):
            continue
        df = pd.read_csv(p)
        r2r = df['r2_reconstructed'].mean()
        r2l = df['r2_latent'].mean()
        diff = r2r - r2l
        direction = "latent_better" if r2l > r2r else "recon_better"
        print(f"  {e:30s} {method:5s}  latent={r2l:.3f}  recon={r2r:.3f}  diff={diff:+.3f}  ({direction})")
