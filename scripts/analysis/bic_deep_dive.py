"""Compute the log-determinant VAR residual curve that BIC actually uses.
This shows why CS01 is special at the CROSS-MODAL level."""
import numpy as np
import os
import yaml

experiments = [
    ('DO_CS01_swarm_C01_l05', 23),
    ('DO_DR01_dring_C01_l01', 2),
    ('DO_EC01_esccol_C2_l3', 1),
    ('DO_DM01_dmill_C09_l05', 3),
]

for name, bic_reported in experiments:
    base = f'oscar_output/{name}'
    pod = np.load(f'{base}/rom_common/pod_basis.npz')
    U = pod['U']
    X_mean = np.load(f'{base}/rom_common/X_train_mean.npy')
    d = U.shape[1]

    with open(f'{base}/config_used.yaml') as f:
        cfg = yaml.safe_load(f)
    subsample = cfg.get('density', {}).get('subsample', 3)

    train_dir = f'{base}/train'
    traj_dirs = sorted([dd for dd in os.listdir(train_dir) if dd.startswith('train_')])
    n_use = min(len(traj_dirs), 50)

    latent_trajs = []
    for i in range(n_use):
        rho = np.load(f'{train_dir}/{traj_dirs[i]}/density.npz')['rho']
        y = (rho.reshape(rho.shape[0], -1) - X_mean) @ U
        latent_trajs.append(y[::subsample])

    T = latent_trajs[0].shape[0]
    concat = np.concatenate(latent_trajs, axis=0)  # (n_use*T, d)
    N = concat.shape[0]

    print(f"\n{'='*70}")
    print(f"  {name}  (BIC reported = {bic_reported})")
    print(f"  N={N} (={n_use}x{T}), d={d}")
    print(f"{'='*70}")

    # For each lag order w, fit VAR(w) and compute log(det(Sigma_resid))
    # BIC = N*log(det(Sigma)) + d^2*w*log(N)
    # We just compare log(det(Sigma)) across lags
    lags_to_test = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30]
    
    print(f"\n  {'lag':>4s}  {'log(det(Sig))':>14s}  {'trace(Sig)':>12s}  {'BIC proxy':>12s}")
    print(f"  {'----':>4s}  {'-'*14:>14s}  {'-'*12:>12s}  {'-'*12:>12s}")

    for w in lags_to_test:
        if w >= T - 1:
            continue

        # Build VAR(w) design matrix
        Xw = np.hstack([concat[w-j-1:N-j-1] for j in range(w)])  # (N-w, d*w)
        Yw = concat[w:]  # (N-w, d)
        n_obs = Yw.shape[0]

        # Fit by OLS
        A, _, _, _ = np.linalg.lstsq(Xw, Yw, rcond=None)
        resid = Yw - Xw @ A
        Sigma = (resid.T @ resid) / n_obs  # (d, d) covariance

        log_det = np.linalg.slogdet(Sigma)[1]
        trace = np.trace(Sigma)
        k_params = d * d * w
        bic_proxy = n_obs * log_det + k_params * np.log(n_obs)

        print(f"  {w:4d}  {log_det:14.4f}  {trace:12.6f}  {bic_proxy:12.1f}")

    # Also compute the spectral (eigenvalue) structure of the lag-1 coefficients
    # This reveals whether dynamics are rotational (complex eigenvalues) or diffusive
    w = 1
    Xw = concat[:-1]
    Yw = concat[1:]
    A1, _, _, _ = np.linalg.lstsq(Xw, Yw, rcond=None)
    eigs = np.linalg.eigvals(A1)
    
    # Sort by magnitude
    mag = np.abs(eigs)
    idx = np.argsort(-mag)
    eigs = eigs[idx]
    
    print(f"\n  Top eigenvalues of VAR(1) coefficient A1:")
    print(f"  {'#':>3s}  {'|lambda|':>10s}  {'Re':>10s}  {'Im':>10s}  {'Freq (cyc/step)':>16s}")
    for i in range(min(10, d)):
        e = eigs[i]
        freq = np.abs(np.angle(e)) / (2 * np.pi)
        print(f"  {i+1:3d}  {np.abs(e):10.4f}  {e.real:10.4f}  {e.imag:10.4f}  {freq:16.4f}")

    # Count how many eigenvalues are > 0.9 magnitude (slow modes)
    n_slow = np.sum(mag > 0.9)
    n_complex = np.sum(np.abs(eigs.imag) > 0.01)
    print(f"\n  Slow modes (|lambda|>0.9): {n_slow}/{d}")
    print(f"  Complex pairs: {n_complex}/{d}")
    
    # Dominant frequency
    dom_freq = np.abs(np.angle(eigs[0])) / (2 * np.pi)
    print(f"  Dominant frequency: {dom_freq:.4f} cycles/step")
    if dom_freq > 0.01:
        print(f"  Dominant period: {1/dom_freq:.1f} steps = {1/dom_freq * 0.12:.1f}s")

print("\n\nDone.")
