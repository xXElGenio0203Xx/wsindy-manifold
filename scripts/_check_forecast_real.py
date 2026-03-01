#!/usr/bin/env python3
"""Check if MVAR forecast is real or just reconstruction."""
import numpy as np

base = 'oscar_output/AL5_align_sqrtSimplex_energy099_p5_H500_eta02_v2_speed2/test/test_008'

pred = np.load(f'{base}/density_pred_mvar.npz')
truth = np.load(f'{base}/density_true.npz')

rho_pred = pred['rho']
rho_true = truth['rho']
fsi = int(pred['forecast_start_idx'])

# Check: is forecast meaningfully different from truth?
dt_true = float(truth['times'][1] - truth['times'][0])
dt_pred = float(pred['times'][1] - pred['times'][0])
rom_sub = max(1, round(dt_pred / dt_true))
truth_sub = rho_true[::rom_sub]

fc_pred = rho_pred[fsi:]
fc_true = truth_sub[fsi:fsi+len(fc_pred)]
L = min(len(fc_pred), len(fc_true))

# Frame-by-frame difference at different time horizons  
for frac in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
    idx = min(int(frac * L), L-1)
    diff = np.abs(fc_pred[idx] - fc_true[idx])
    rel_diff = diff / (np.abs(fc_true[idx]) + 1e-10)
    md = diff.mean()
    mx = diff.max()
    mr = rel_diff.mean()
    print(f"t_frac={frac*100:.0f}% (idx={idx}): mean_diff={md:.6f}, max_diff={mx:.4f}, mean_rel={mr:.4f}")

print()
print(f"Total forecast frames: {L}")
print(f"forecast_start_idx: {fsi}")

# Are ALL forecast frames identical to truth? (would mean no actual forecasting)
identical_frames = 0
for i in range(L):
    if np.allclose(fc_pred[i], fc_true[i], atol=1e-6):
        identical_frames += 1
pct = identical_frames/L*100
print(f"Frames identical to truth: {identical_frames}/{L} ({pct:.1f}%)")

# Is the forecast just a constant (repeating the IC)?
ic_frame = fc_pred[0]
constant_frames = 0
for i in range(L):
    if np.allclose(fc_pred[i], ic_frame, atol=1e-6):
        constant_frames += 1
pct2 = constant_frames/L*100
print(f"Frames identical to IC: {constant_frames}/{L} ({pct2:.1f}%)")

# Check the reported R² is computed on shift-aligned vs un-aligned
# The R² reported in metrics is -62, but manual on saved npz gives 0.99 
# because saved npz was un-shifted for visualization
# The evaluator computes R² in ALIGNED space (before un-shifting),
# which is the correct thing to do for metric computation
import json
with open(f'{base}/metrics_summary.json') as f:
    m = json.load(f)
print()
print(f"Reported r2_recon: {m['r2_recon']:.4f}")
print(f"shift_align: {m.get('shift_align')}")
print()
print("KEY INSIGHT: The reported R² is computed in SHIFT-ALIGNED space")
print("(before un-shifting). The saved density_pred.npz has been UN-SHIFTED")
print("for visualization. So manual R² on saved npz will NOT match reported R²")
print("because the density transform (sqrt + simplex) and shift alignment")
print("create differences when working in un-aligned physical space.")
