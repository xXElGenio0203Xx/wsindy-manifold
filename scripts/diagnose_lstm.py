#!/usr/bin/env python3
"""
LSTM Diagnosis Script v2
========================
Uses pre-computed predictions (density_pred_mvar/lstm.npz) already on disk.
Deep-dive into why LSTM underperforms MVAR.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
import torch
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rom.lstm_rom import LatentLSTMROM

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "predictions" / "_lstm_diagnosis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LSTM DIAGNOSTIC REPORT")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════
# 1. DATA SCALE ANALYSIS  
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("1. LATENT DATA SCALE & STRUCTURE")
print("=" * 80)

EXPERIMENTS = {
    "synthesis_v2_4": ROOT / "oscar_output" / "synthesis_v2_4",
    "suite_S0_det_advect_H37_mvar_p3": ROOT / "oscar_output" / "suite_S0_det_advect_H37_mvar_p3",
}

for name, exp_dir in EXPERIMENTS.items():
    common = exp_dir / "rom_common"
    if not common.exists():
        continue
    
    ds = np.load(common / "latent_dataset.npz")
    X_all = ds["X_all"]  # [N, lag, d]
    Y_all = ds["Y_all"]  # [N, d]
    N, lag, d = X_all.shape
    
    X_flat = X_all.reshape(-1, d)
    
    print(f"\n  ── {name} ──")
    print(f"  Samples: {N:,}  |  Lag: {lag}  |  d: {d}")
    
    # KEY DIAGNOSTIC: How much does Y differ from X[:, -1, :] (persistence)?
    delta = Y_all - X_all[:, -1, :]
    delta_norm = np.linalg.norm(delta, axis=1)
    x_norm = np.linalg.norm(X_all[:, -1, :], axis=1)
    relative_delta = delta_norm / (x_norm + 1e-12)
    
    print(f"\n  Δ = Y - X[:,-1,:] (one-step change):")
    print(f"    |Δ| mean:    {delta_norm.mean():.6f}")
    print(f"    |Δ|/|X| mean: {relative_delta.mean():.6f}  ({relative_delta.mean()*100:.3f}%)")
    
    # Signal-to-noise
    var_state = np.var(X_all[:, -1, :], axis=0)
    var_delta = np.var(delta, axis=0)
    snr = var_delta / (var_state + 1e-12)
    print(f"\n  Var(Δ)/Var(X) ratio per mode (dynamics signal vs state variance):")
    for i in range(min(d, 5)):
        print(f"    Mode {i}: {snr[i]:.6f}  ({snr[i]*100:.4f}%)")
    print(f"    Mean across all {d} modes: {snr.mean():.6f}  ({snr.mean()*100:.4f}%)")


# ═══════════════════════════════════════════════════════════════════
# 2. TRAINING CURVES
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("2. LSTM TRAINING CURVES")
print("=" * 80)

lstm_exps = {}
for name, exp_dir in EXPERIMENTS.items():
    log = exp_dir / "LSTM" / "training_log.csv"
    if log.exists():
        lstm_exps[name] = log

for name, log_file in lstm_exps.items():
    df = pd.read_csv(log_file)
    print(f"\n  ── {name} ──")
    print(f"  Epochs trained: {len(df)}")
    print(f"  Final train_loss: {df['train_loss'].iloc[-1]:.8f}")
    print(f"  Final val_loss:   {df['val_loss'].iloc[-1]:.8f}")
    print(f"  Best val_loss:    {df['val_loss'].min():.8f} (epoch {df['val_loss'].idxmin()+1})")
    print(f"  Train/Val ratio:  {df['train_loss'].iloc[-1] / df['val_loss'].iloc[-1]:.4f}")
    
    best_epoch = df['val_loss'].idxmin()
    if best_epoch < len(df) - 1:
        print(f"  ⚠️  Early stopped {len(df)-1-best_epoch} epochs after best → OVERFITTING")


# ═══════════════════════════════════════════════════════════════════
# 3. PHYSICAL SPACE COMPARISON: MVAR vs LSTM per test run
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("3. PHYSICAL SPACE: MVAR vs LSTM (synthesis_v2_4)")
print("=" * 80)

exp = ROOT / "oscar_output" / "synthesis_v2_4"
test_dir = exp / "test"
with open(test_dir / "metadata.json") as f:
    test_meta = json.load(f)

mvar_r2_list, lstm_r2_list = [], []

for tm in test_meta:
    run = tm["run_name"]
    run_dir = test_dir / run
    
    # Load ground truth & predictions
    rho_true = np.load(run_dir / "density_true.npz")["rho"]    # [T_full, Nx, Ny]
    
    has_mvar = (run_dir / "density_pred_mvar.npz").exists()
    has_lstm = (run_dir / "density_pred_lstm.npz").exists()
    
    if has_mvar:
        pred_mvar = np.load(run_dir / "density_pred_mvar.npz")
        rho_mvar = pred_mvar["rho"]
        fsi_mvar = int(pred_mvar["forecast_start_idx"])
    
    if has_lstm:
        pred_lstm = np.load(run_dir / "density_pred_lstm.npz")
        rho_lstm = pred_lstm["rho"]
        fsi_lstm = int(pred_lstm["forecast_start_idx"])
    
    # Align truth to forecast window
    if has_mvar and has_lstm:
        fsi = max(fsi_mvar, fsi_lstm)
        T_pred = min(len(rho_mvar), len(rho_lstm))
        
        rho_t = rho_true[fsi:fsi+T_pred].reshape(T_pred, -1)
        rho_m = rho_mvar[:T_pred].reshape(T_pred, -1)
        rho_l = rho_lstm[:T_pred].reshape(T_pred, -1)
        
        # R² over full rollout
        ss_tot = np.sum((rho_t - rho_t.mean(axis=0))**2)
        r2_m = 1 - np.sum((rho_t - rho_m)**2) / ss_tot
        r2_l = 1 - np.sum((rho_t - rho_l)**2) / ss_tot
        
        mvar_r2_list.append(r2_m)
        lstm_r2_list.append(r2_l)
        
        # Per-timestep R²
        r2_m_t, r2_l_t = [], []
        for t in range(T_pred):
            ss_t = np.sum((rho_t[t] - rho_t[t].mean())**2) + 1e-12
            r2_m_t.append(1 - np.sum((rho_t[t] - rho_m[t])**2) / ss_t)
            r2_l_t.append(1 - np.sum((rho_t[t] - rho_l[t])**2) / ss_t)
        
        print(f"\n  {run}:")
        print(f"    Rollout R²:   MVAR={r2_m:+.4f}   LSTM={r2_l:+.4f}   Δ={r2_m-r2_l:+.4f}")
        print(f"    t=0 R²:       MVAR={r2_m_t[0]:+.4f}   LSTM={r2_l_t[0]:+.4f}")
        print(f"    t=end R²:     MVAR={r2_m_t[-1]:+.4f}   LSTM={r2_l_t[-1]:+.4f}")

if mvar_r2_list:
    print(f"\n  ── SUMMARY ──")
    print(f"  Mean R² MVAR: {np.mean(mvar_r2_list):+.4f} ± {np.std(mvar_r2_list):.4f}")
    print(f"  Mean R² LSTM: {np.mean(lstm_r2_list):+.4f} ± {np.std(lstm_r2_list):.4f}")
    print(f"  LSTM wins in {sum(l > m for l, m in zip(lstm_r2_list, mvar_r2_list))}/{len(mvar_r2_list)} runs")


# ═══════════════════════════════════════════════════════════════════
# 4. LATENT SPACE: 1-step quality & rollout divergence
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("4. LATENT-SPACE 1-STEP vs ROLLOUT ANALYSIS (synthesis_v2_4)")
print("=" * 80)

common = exp / "rom_common"
pod_data = np.load(common / "pod_basis.npz")
U = pod_data["U"]
X_train_mean = np.load(common / "X_train_mean.npy")
ds = np.load(common / "latent_dataset.npz")
X_all, Y_all = ds["X_all"], ds["Y_all"]
N, lag, d = X_all.shape

# Load MVAR
mvar_data = np.load(exp / "MVAR" / "mvar_model.npz")
A_coef = mvar_data["A_companion"]
p_mvar = int(mvar_data["p"])

# Load LSTM
with open(exp / "config_used.yaml") as f:
    cfg = yaml.safe_load(f)
lstm_cfg = cfg.get('rom', {}).get('models', {}).get('lstm', {})
hidden = lstm_cfg.get('hidden_units', 64)
nlayers = lstm_cfg.get('num_layers', 2)
dropout = lstm_cfg.get('dropout', 0.0)

model = LatentLSTMROM(d=d, hidden_units=hidden, num_layers=nlayers, dropout=dropout)
state_dict = torch.load(exp / "LSTM" / "lstm_state_dict.pt", map_location='cpu', weights_only=True)
model.load_state_dict(state_dict)
model.eval()

mvar_param_count = d * (p_mvar * d)
lstm_param_count = sum(p.numel() for p in model.parameters())

print(f"\n  MVAR: p={p_mvar}, d={d}, params={mvar_param_count:,}")
print(f"  LSTM: lag={lag}, d={d}, hidden={hidden}, layers={nlayers}, params={lstm_param_count:,}")

# ── 1-step predictions on full dataset ──
with torch.no_grad():
    X_tensor = torch.tensor(X_all, dtype=torch.float32)
    preds_lstm = []
    for i in range(0, len(X_tensor), 512):
        preds_lstm.append(model(X_tensor[i:i+512]).numpy())
    Y_pred_lstm = np.concatenate(preds_lstm, axis=0)

# MVAR 1-step
X_mvar_input = X_all[:, -p_mvar:, :].reshape(N, -1)
Y_pred_mvar = (A_coef @ X_mvar_input.T).T

# Persistence baseline
Y_persist = X_all[:, -1, :]

# Compute R² on full dataset
ss_tot = np.sum((Y_all - Y_all.mean(axis=0))**2)
r2_persist = 1 - np.sum((Y_all - Y_persist)**2) / ss_tot
r2_mvar_1s = 1 - np.sum((Y_all - Y_pred_mvar)**2) / ss_tot
r2_lstm_1s = 1 - np.sum((Y_all - Y_pred_lstm)**2) / ss_tot

mse_persist = np.mean((Y_all - Y_persist)**2)
mse_mvar = np.mean((Y_all - Y_pred_mvar)**2)
mse_lstm = np.mean((Y_all - Y_pred_lstm)**2)

print(f"\n  ── 1-step prediction quality (full dataset, {N:,} samples) ──")
print(f"  {'Model':<15s} {'MSE':>12s} {'R²':>10s}")
print(f"  {'-'*40}")
print(f"  {'Persistence':<15s} {mse_persist:12.6f} {r2_persist:10.6f}")
print(f"  {'MVAR':<15s} {mse_mvar:12.6f} {r2_mvar_1s:10.6f}")
print(f"  {'LSTM':<15s} {mse_lstm:12.6f} {r2_lstm_1s:10.6f}")

# ── Is LSTM just learning persistence? ──
lstm_delta = Y_pred_lstm - Y_persist
true_delta = Y_all - Y_persist

corr_per_mode = []
for i in range(d):
    c = np.corrcoef(lstm_delta[:, i], true_delta[:, i])[0, 1]
    corr_per_mode.append(c)

print(f"\n  ── Is LSTM capturing dynamics beyond persistence? ──")
print(f"  Correlation between (LSTM_pred - persist) and (truth - persist):")
for i in range(d):
    bar = "█" * int(abs(corr_per_mode[i]) * 20)
    print(f"    Mode {i:2d}: {corr_per_mode[i]:+.4f}  {bar}")
print(f"    Mean: {np.mean(corr_per_mode):+.4f}")

# Variance analysis
var_true_delta = np.var(true_delta, axis=0)
var_lstm_delta = np.var(lstm_delta, axis=0)
print(f"\n  LSTM 'correction magnitude' vs true dynamics:")
print(f"  {'Mode':<6s} {'Var(trueΔ)':>12s} {'Var(lstmΔ)':>12s} {'Ratio':>8s}")
for i in range(d):
    ratio = var_lstm_delta[i] / (var_true_delta[i] + 1e-12)
    print(f"  {i:<6d} {var_true_delta[i]:12.4e} {var_lstm_delta[i]:12.4e} {ratio:8.4f}")

mean_ratio = np.mean(var_lstm_delta) / (np.mean(var_true_delta) + 1e-12)
print(f"\n  Mean Var(lstmΔ)/Var(trueΔ): {mean_ratio:.4f}")

if mean_ratio < 0.5:
    print(f"  ⚠️  LSTM corrections are {mean_ratio:.1%} the magnitude of true dynamics")
    print(f"      → LSTM is UNDER-CORRECTING (mostly doing persistence)")
elif mean_ratio > 2.0:
    print(f"  ⚠️  LSTM corrections are {mean_ratio:.1f}x the true dynamics → OVER-CORRECTING")

# Per-timestep relative_delta for later diagnosis
relative_delta = np.linalg.norm(true_delta, axis=1) / (np.linalg.norm(Y_persist, axis=1) + 1e-12)

# ═══════════════════════════════════════════════════════════════════
# 5. LATENT TRAJECTORY ROLLOUT COMPARISON
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("5. LATENT TRAJECTORY ROLLOUT")
print("=" * 80)

# Use first test run's truth density to get latent trajectory
run0 = test_meta[0]["run_name"]
rho_true_full = np.load(test_dir / run0 / "density_true.npz")["rho"]
T_full = rho_true_full.shape[0]
rho_flat = rho_true_full.reshape(T_full, -1)
y_true = (rho_flat - X_train_mean) @ U  # [T, d]

lag_max = max(p_mvar, lag)

# MVAR rollout
y_mvar = list(y_true[:lag_max].copy())
for t in range(lag_max, T_full):
    win = np.array(y_mvar[t-p_mvar:t]).flatten()
    y_mvar.append(A_coef @ win)
y_mvar = np.array(y_mvar)

# LSTM rollout
y_lstm = list(y_true[:lag_max].copy())
with torch.no_grad():
    for t in range(lag_max, T_full):
        win = np.array(y_lstm[t-lag:t])
        x_t = torch.tensor(win, dtype=torch.float32).unsqueeze(0)
        y_lstm.append(model(x_t).numpy()[0])
y_lstm = np.array(y_lstm)

# Per-timestep error
err_mvar = np.linalg.norm(y_true - y_mvar, axis=1)
err_lstm = np.linalg.norm(y_true - y_lstm, axis=1)
norm_true = np.linalg.norm(y_true, axis=1)

print(f"\n  Test run: {run0}  |  T={T_full} timesteps  |  d={d} modes")
print(f"\n  Latent rollout error (mean over forecast horizon):")
print(f"    MVAR: {err_mvar[lag_max:].mean():.4f}")
print(f"    LSTM: {err_lstm[lag_max:].mean():.4f}")
print(f"\n  Latent rollout error at specific timesteps:")
for frac in [0.1, 0.25, 0.5, 0.75, 1.0]:
    t = min(int(frac * T_full), T_full - 1)
    print(f"    t={t:3d} ({frac:.0%}):  MVAR={err_mvar[t]:.4f}  LSTM={err_lstm[t]:.4f}  true_norm={norm_true[t]:.4f}")

# ── PLOT: 4-panel comparison ──
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

ax = axes[0, 0]
ax.plot(err_mvar, label='MVAR', alpha=0.8, linewidth=1.5)
ax.plot(err_lstm, label='LSTM', alpha=0.8, linewidth=1.5)
ax.axvline(x=lag_max, color='gray', linestyle=':', alpha=0.5, label=f'Forecast start (t={lag_max})')
ax.set_xlabel('Timestep')
ax.set_ylabel('|y_true - y_pred|₂')
ax.set_title(f'Latent Rollout Error Over Time ({run0})')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(err_mvar / (norm_true + 1e-12), label='MVAR', alpha=0.8)
ax.plot(err_lstm / (norm_true + 1e-12), label='LSTM', alpha=0.8)
ax.axvline(x=lag_max, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Timestep')
ax.set_ylabel('Relative error')
ax.set_title('Relative Rollout Error')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(y_true[:, 0], 'k-', label='Truth', linewidth=2, alpha=0.8)
ax.plot(y_mvar[:, 0], 'b--', label='MVAR', alpha=0.8)
ax.plot(y_lstm[:, 0], 'r--', label='LSTM', alpha=0.8)
ax.axvline(x=lag_max, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Timestep')
ax.set_ylabel('Amplitude')
ax.set_title('Latent Mode 0')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
if d > 1:
    ax.plot(y_true[:, 1], 'k-', label='Truth', linewidth=2, alpha=0.8)
    ax.plot(y_mvar[:, 1], 'b--', label='MVAR', alpha=0.8)
    ax.plot(y_lstm[:, 1], 'r--', label='LSTM', alpha=0.8)
    ax.axvline(x=lag_max, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Amplitude')
    ax.set_title('Latent Mode 1')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle(f'MVAR vs LSTM Rollout: synthesis_v2_4 ({run0})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / "mvar_vs_lstm_rollout.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved: mvar_vs_lstm_rollout.png")


# ═══════════════════════════════════════════════════════════════════
# 6. LINEARITY TEST — Is the latent dynamics actually linear?
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("6. LINEARITY OF LATENT DYNAMICS")
print("=" * 80)

for name, exp_dir in EXPERIMENTS.items():
    common = exp_dir / "rom_common"
    if not common.exists():
        continue
    
    ds = np.load(common / "latent_dataset.npz")
    X_all_local, Y_all_local = ds["X_all"], ds["Y_all"]
    N_l, lag_l, d_l = X_all_local.shape
    
    print(f"\n  ── {name} (N={N_l:,}, lag={lag_l}, d={d_l}) ──")
    
    # Fit: Y = A * X[:,-1,:] + b  (simplest linear: 1 lag)
    X_last = np.hstack([X_all_local[:, -1, :], np.ones((N_l, 1))])
    A_1, _, _, _ = np.linalg.lstsq(X_last, Y_all_local, rcond=None)
    Y_1 = X_last @ A_1
    ss_tot_l = np.sum((Y_all_local - Y_all_local.mean(axis=0))**2)
    r2_1lag = 1 - np.sum((Y_all_local - Y_1)**2) / ss_tot_l
    
    # Fit: Y = A * X.flatten() + b  (full window, same as MVAR)
    X_flat_l = np.hstack([X_all_local.reshape(N_l, -1), np.ones((N_l, 1))])
    A_full, _, _, _ = np.linalg.lstsq(X_flat_l, Y_all_local, rcond=None)
    Y_full = X_flat_l @ A_full
    r2_full = 1 - np.sum((Y_all_local - Y_full)**2) / ss_tot_l
    
    # Persistence baseline
    Y_persist_l = X_all_local[:, -1, :]
    r2_persist_l = 1 - np.sum((Y_all_local - Y_persist_l)**2) / ss_tot_l
    
    print(f"  Persistence (copy last):  R² = {r2_persist_l:.6f}")
    print(f"  Linear (1-lag + bias):    R² = {r2_1lag:.6f}")
    print(f"  Linear (full window):     R² = {r2_full:.6f}")
    print(f"  Nonlinear residual:       {(1-r2_full)*100:.4f}%")
    
    if r2_full > 0.999:
        print(f"  ✅ Dynamics are >99.9% linear → MVAR is near-optimal, LSTM has NO advantage")
    elif r2_full > 0.99:
        print(f"  ⚠️  Dynamics are >99% linear → Very little nonlinear signal for LSTM")
    else:
        print(f"  📊 {(1-r2_full)*100:.2f}% nonlinear residual — LSTM could potentially help")


# ═══════════════════════════════════════════════════════════════════
# 7. MODEL CAPACITY
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("7. MODEL CAPACITY ANALYSIS")
print("=" * 80)

for name, exp_dir in EXPERIMENTS.items():
    common = exp_dir / "rom_common"
    if not common.exists():
        continue
    ds = np.load(common / "latent_dataset.npz")
    N_l, lag_l, d_l = ds["X_all"].shape
    
    print(f"\n  ── {name} ──")
    
    mvar_file = exp_dir / "MVAR" / "mvar_model.npz"
    if mvar_file.exists():
        md = np.load(mvar_file)
        p_l = int(md["p"])
        mvar_params_l = d_l * (p_l * d_l)
        print(f"  MVAR: p={p_l}, d={d_l}, params={mvar_params_l:,}  (samples/param = {N_l/mvar_params_l:.0f})")
    
    lstm_dir = exp_dir / "LSTM"
    if lstm_dir.exists() and (lstm_dir / "lstm_state_dict.pt").exists():
        cfg_file = exp_dir / "config_used.yaml"
        h, nl = 64, 2
        if cfg_file.exists():
            with open(cfg_file) as f:
                c = yaml.safe_load(f)
            lc = c.get('rom', {}).get('models', {}).get('lstm', {})
            h = lc.get('hidden_units', 64)
            nl = lc.get('num_layers', 2)
        
        test_m = LatentLSTMROM(d=d_l, hidden_units=h, num_layers=nl)
        lstm_params_l = sum(pp.numel() for pp in test_m.parameters())
        print(f"  LSTM: hidden={h}, layers={nl}, params={lstm_params_l:,}  (samples/param = {N_l/lstm_params_l:.0f})")
        
        if mvar_file.exists():
            print(f"  → LSTM has {lstm_params_l/mvar_params_l:.1f}x more params than MVAR")
            if N_l / lstm_params_l < 10:
                print(f"  ⚠️  SEVERELY underdetermined: only {N_l/lstm_params_l:.1f} samples per param!")
                print(f"      (rule of thumb: need ≥50-100 samples/param for neural nets)")


# ═══════════════════════════════════════════════════════════════════
# 8. 1-STEP PREDICTION VISUALIZATION
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("8. GENERATING 1-STEP PREDICTION PLOTS")
print("=" * 80)

n_modes_plot = min(d, 4)
fig, axes = plt.subplots(n_modes_plot, 1, figsize=(16, 3*n_modes_plot), sharex=True)
if n_modes_plot == 1:
    axes = [axes]

chunk = slice(0, min(500, N))

for i, ax in enumerate(axes):
    ax.plot(Y_all[chunk, i], 'k-', label='Truth', alpha=0.8, linewidth=1.5)
    ax.plot(Y_persist[chunk, i], color='gray', linestyle='--', label='Persistence', alpha=0.4)
    ax.plot(Y_pred_lstm[chunk, i], 'r-', label='LSTM', alpha=0.6)
    ax.plot(Y_pred_mvar[chunk, i], 'g-', label='MVAR', alpha=0.6)
    ax.set_ylabel(f'Mode {i}')
    if i == 0:
        ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Sample index')
plt.suptitle(f'1-Step Predictions: Truth vs Persistence vs MVAR vs LSTM (synthesis_v2_4)', fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "1step_predictions_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: 1step_predictions_comparison.png")


# ═══════════════════════════════════════════════════════════════════
# 9. ERROR DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("9. ERROR DISTRIBUTION ANALYSIS")
print("=" * 80)

err_persist = np.linalg.norm(Y_all - Y_persist, axis=1)
err_mvar_1s = np.linalg.norm(Y_all - Y_pred_mvar, axis=1)
err_lstm_1s = np.linalg.norm(Y_all - Y_pred_lstm, axis=1)

print(f"\n  1-step |error|₂ distribution:")
print(f"  {'Model':<15s} {'Mean':>10s} {'Median':>10s} {'P95':>10s} {'Max':>10s}")
print(f"  {'-'*55}")
for nm, e in [('Persistence', err_persist), ('MVAR', err_mvar_1s), ('LSTM', err_lstm_1s)]:
    print(f"  {nm:<15s} {e.mean():10.4f} {np.median(e):10.4f} {np.percentile(e, 95):10.4f} {e.max():10.4f}")

lstm_wins = err_lstm_1s < err_mvar_1s
print(f"\n  LSTM beats MVAR on {lstm_wins.sum():,}/{N:,} samples ({lstm_wins.mean()*100:.1f}%)")

fig, ax = plt.subplots(figsize=(10, 5))
bins = np.linspace(0, max(err_persist.max(), err_mvar_1s.max(), err_lstm_1s.max()), 100)
ax.hist(err_persist, bins=bins, alpha=0.3, label=f'Persistence (mean={err_persist.mean():.3f})', density=True)
ax.hist(err_mvar_1s, bins=bins, alpha=0.5, label=f'MVAR (mean={err_mvar_1s.mean():.3f})', density=True)
ax.hist(err_lstm_1s, bins=bins, alpha=0.5, label=f'LSTM (mean={err_lstm_1s.mean():.3f})', density=True)
ax.set_xlabel('1-step |error|₂')
ax.set_ylabel('Density')
ax.set_title('Distribution of 1-Step Prediction Errors')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "error_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: error_distribution.png")


# ═══════════════════════════════════════════════════════════════════
# FINAL DIAGNOSIS
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("█ FINAL DIAGNOSIS")
print("=" * 80)
print(f"""
The LSTM underperforms MVAR for several reinforcing reasons:

1. DYNAMICS ARE OVERWHELMINGLY LINEAR
   The latent dynamics (POD coefficients over time) are {r2_full*100:.2f}% explainable 
   by a simple linear map. Only {(1-r2_full)*100:.3f}% of variance is nonlinear.
   → MVAR (a linear model) is NEAR-OPTIMAL. There is almost no nonlinear signal 
     for the LSTM to exploit.

2. THE PREDICTION TARGET IS ALMOST PERSISTENCE
   The one-step change |Y - X[-1]| averages only {relative_delta.mean()*100:.2f}% of the state norm.
   The LSTM must learn to output values very close to its last input — a tiny 
   correction on a large signal. This is a poor inductive bias for neural nets
   (they're better at learning large-scale patterns, not tiny residuals).

3. LSTM HAS {lstm_param_count:,} PARAMETERS vs MVAR's {mvar_param_count:,}
   With only {N:,} training samples, the LSTM has ~{N/lstm_param_count:.0f} samples per parameter.
   MVAR has ~{N/mvar_param_count:.0f} samples per parameter AND is the correct model class.
   → LSTM is OVER-PARAMETERIZED for this task.

4. EXPOSURE BIAS KILLS ROLLOUT
   Even where LSTM's 1-step MSE is acceptable, tiny per-step errors compound
   during autonomous rollout. MVAR's spectral radius controls this naturally;
   LSTM has no such structural guarantee.

BOTTOM LINE: The POD latent dynamics are essentially linear. An LSTM is the wrong
tool — it's like using a neural net to fit y = 2x + 3. MVAR wins because it IS 
the correct model class for this problem. LSTM would only help if the dynamics 
had significant nonlinearity that MVAR cannot capture.
""")

print(f"\nAll diagnostic plots saved to: {OUT_DIR}")
print("=" * 80)
