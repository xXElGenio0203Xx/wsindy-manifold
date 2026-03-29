"""Cross-modal analysis: why CS01 has BIC=23 vs others BIC=2.
Computes lagged cross-correlation matrices to show inter-mode coupling persistence."""
import numpy as np
import os
import yaml

experiments = [
    ('DO_CS01_swarm_C01_l05', 23),  # (name, BIC)
    ('DO_DR01_dring_C01_l01', 2),
    ('DO_SM01_mill_C05_l01', 3),
    ('DO_EC01_esccol_C2_l3', 1),
    ('DO_DM01_dmill_C09_l05', 3),
]

for name, bic in experiments:
    base = f'oscar_output/{name}'

    # Load POD basis and mean
    pod = np.load(f'{base}/rom_common/pod_basis.npz')
    U = pod['U']
    X_mean = np.load(f'{base}/rom_common/X_train_mean.npy')
    d = U.shape[1]

    # Load config for subsample
    with open(f'{base}/config_used.yaml') as f:
        cfg = yaml.safe_load(f)
    subsample = cfg.get('density', {}).get('subsample', 3)

    # Project training trajectories to latent space
    train_dir = f'{base}/train'
    traj_dirs = sorted([dd for dd in os.listdir(train_dir) if dd.startswith('train_')])
    n_use = min(len(traj_dirs), 20)

    latent_trajs = []
    for i in range(n_use):
        rho = np.load(f'{train_dir}/{traj_dirs[i]}/density.npz')['rho']
        T_snap = rho.shape[0]
        rho_flat = rho.reshape(T_snap, -1)
        y = (rho_flat - X_mean) @ U  # (T, d)
        # Subsample to match what VAR uses
        y_sub = y[::subsample]
        latent_trajs.append(y_sub)

    T = latent_trajs[0].shape[0]
    print(f"\n{'='*70}")
    print(f"  {name}  (BIC = {bic})")
    print(f"  T_sub={T}, d={d}, n_trajs={n_use}")
    print(f"{'='*70}")

    # Concatenate all trajectories
    Y = np.stack(latent_trajs)  # (n_use, T, d)

    # 1) Compute mean squared cross-correlation at each lag
    max_lag = min(T - 1, 50)

    # For each lag, compute the d×d correlation matrix, excluding diagonal
    # This shows how much modes predict OTHER modes at different lags
    cross_corr_strength = np.zeros(max_lag + 1)
    diag_corr_strength = np.zeros(max_lag + 1)

    for lag in range(0, max_lag + 1):
        # For each trajectory, compute cross-correlation matrix at this lag
        R_sum = np.zeros((d, d))
        for traj in latent_trajs:
            for m1 in range(d):
                for m2 in range(d):
                    x1 = traj[lag:, m1] - traj[lag:, m1].mean()
                    x2 = traj[:T-lag, m2] - traj[:T-lag, m2].mean()
                    s1 = np.sqrt((x1**2).sum())
                    s2 = np.sqrt((x2**2).sum())
                    if s1 > 1e-15 and s2 > 1e-15:
                        R_sum[m1, m2] += np.abs((x1 * x2).sum() / (s1 * s2))
        R_avg = R_sum / n_use

        # Cross-correlation: off-diagonal mean
        mask = ~np.eye(d, dtype=bool)
        cross_corr_strength[lag] = R_avg[mask].mean()
        # Auto-correlation: diagonal mean
        diag_corr_strength[lag] = np.diag(R_avg).mean()

    # 2) How much ADDITIONAL cross-correlation exists beyond lag 5?
    cross_at_5 = cross_corr_strength[5]
    cross_beyond_5 = cross_corr_strength[6:31].mean()
    auto_at_5 = diag_corr_strength[5]
    auto_beyond_5 = diag_corr_strength[6:31].mean()

    print(f"\n  Lag  |  Auto-corr (diag)  |  Cross-corr (off-diag)")
    print(f"  -----|--------------------|-----------------------")
    for lag in [0, 1, 2, 3, 5, 8, 10, 15, 20, 25, 30]:
        if lag <= max_lag:
            print(f"   {lag:2d}  |      {diag_corr_strength[lag]:.4f}       |       {cross_corr_strength[lag]:.4f}")

    # 3) Key metric: "cross-modal persistence"
    # How slowly does the cross-correlation decay?
    cross_half = np.where(cross_corr_strength < cross_corr_strength[1] * 0.5)[0]
    cross_half_lag = cross_half[0] if len(cross_half) > 0 else max_lag

    print(f"\n  Cross-corr half-life: lag = {cross_half_lag}")
    print(f"  Cross-corr at lag=5: {cross_at_5:.4f}")
    print(f"  Cross-corr mean lag 6-30: {cross_beyond_5:.4f}")
    print(f"  Ratio (persistence): {cross_beyond_5/max(cross_at_5,1e-10):.3f}")
    
    # 4) "Innovation" test: how much does lag>5 help predict next step?
    # Compute 1-step residual for VAR(5) and VAR(1) using simple least squares
    # on a single concatenated trajectory
    concat = np.concatenate(latent_trajs[:10], axis=0)  # (10*T, d)
    N = concat.shape[0]
    
    # VAR(1): predict y_t = A1 * y_{t-1}
    X1 = concat[:-1]
    Y1 = concat[1:]
    A1, _, _, _ = np.linalg.lstsq(X1, Y1, rcond=None)
    resid1 = Y1 - X1 @ A1
    var1_var = np.var(resid1)
    
    # VAR(5): predict y_t = A1*y_{t-1} + ... + A5*y_{t-5}
    X5 = np.hstack([concat[5-j-1:N-j-1] for j in range(5)])  # (N-5, 5*d)
    Y5 = concat[5:]
    A5, _, _, _ = np.linalg.lstsq(X5, Y5, rcond=None)
    resid5 = Y5 - X5 @ A5
    var5_var = np.var(resid5)
    
    # VAR(20): predict with 20 lags
    w = 20
    if N > w + 10:
        Xw = np.hstack([concat[w-j-1:N-j-1] for j in range(w)])
        Yw = concat[w:]
        Aw, _, _, _ = np.linalg.lstsq(Xw, Yw, rcond=None)
        residw = Yw - Xw @ Aw
        varw_var = np.var(residw)
    else:
        varw_var = float('nan')
    
    print(f"\n  Residual variance (prediction error):")
    print(f"    VAR(1):  {var1_var:.6f}")
    print(f"    VAR(5):  {var5_var:.6f}  ({(1-var5_var/var1_var)*100:.1f}% reduction from VAR(1))")
    print(f"    VAR(20): {varw_var:.6f}  ({(1-varw_var/var1_var)*100:.1f}% reduction from VAR(1))")
    print(f"    VAR(5→20) improvement: {(1-varw_var/var5_var)*100:.1f}%")

print("\n\nDone.")
