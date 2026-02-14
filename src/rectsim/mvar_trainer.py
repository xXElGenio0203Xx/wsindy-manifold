"""
MVAR Trainer Module
===================

Trains Multivariate AutoRegressive (MVAR) models on POD latent space.
Supports:
- Ridge regression with configurable regularization
- Optional eigenvalue stability enforcement
- Companion matrix form
- k-step rollout loss training (multi-step gradient descent)
"""

import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge


def train_mvar_model(pod_data, rom_config):
    """
    Train MVAR model on POD latent space.
    
    Parameters
    ----------
    pod_data : dict
        POD data from build_pod_basis, must contain:
        - X_latent: latent training data
        - M: number of training runs
        - T_rom: timesteps per run
        - R_POD: number of POD modes
    rom_config : dict
        ROM configuration with keys:
        - mvar_lag: lag order p (default: 5)
        - ridge_alpha: regularization parameter (default: 1e-6)
        - eigenvalue_threshold: optional stability threshold
    
    Returns
    -------
    dict
        MVAR model dictionary with keys:
        - model: trained Ridge model
        - P_LAG: lag order
        - RIDGE_ALPHA: regularization parameter
        - r2_train: training R²
        - train_rmse: training RMSE
        - A_matrices: coefficient matrices (p, d, d)
        - rho_before: spectral radius before scaling
        - rho_after: spectral radius after scaling
    """
    
    X_latent = pod_data['X_latent']
    M = pod_data['M']
    T_rom = pod_data['T_rom']
    R_POD = pod_data['R_POD']
    
    # Support both old and new config structures
    # New: rom.models.mvar.lag and rom.models.mvar.ridge_alpha
    # Old: rom.mvar_lag and rom.ridge_alpha (backward compatible)
    if 'models' in rom_config and 'mvar' in rom_config['models']:
        mvar_config = rom_config['models']['mvar']
        P_LAG = mvar_config.get('lag', 5)
        RIDGE_ALPHA = mvar_config.get('ridge_alpha', 1e-6)
    else:
        # Backward compatibility with old config structure
        P_LAG = rom_config.get('mvar_lag', 5)
        RIDGE_ALPHA = rom_config.get('ridge_alpha', 1e-6)
    
    print(f"\nTraining global MVAR (p={P_LAG}, α={RIDGE_ALPHA})...")
    
    # Reshape latent data for MVAR
    X_latent_runs = X_latent.reshape(M, T_rom, R_POD)
    
    # Build training matrices
    X_train_list = []
    Y_train_list = []
    
    for m in range(M):
        X_m = X_latent_runs[m]  # Shape: (T_rom, R_POD)
        
        for t in range(P_LAG, T_rom):
            # Feature vector: [x(t-p), ..., x(t-1)]
            x_hist = X_m[t-P_LAG:t].flatten()  # Shape: (P_LAG * R_POD,)
            y_target = X_m[t]  # Shape: (R_POD,)
            
            X_train_list.append(x_hist)
            Y_train_list.append(y_target)
    
    X_train = np.array(X_train_list)
    Y_train = np.array(Y_train_list)
    
    print(f"✓ MVAR training data: X{X_train.shape}, Y{Y_train.shape}")
    
    # Train Ridge regression
    mvar_model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    mvar_model.fit(X_train, Y_train)
    
    # Training R²
    Y_train_pred = mvar_model.predict(X_train)
    ss_res = np.sum((Y_train - Y_train_pred)**2)
    ss_tot = np.sum((Y_train - Y_train.mean(axis=0))**2)
    r2_train = 1 - ss_res / ss_tot
    
    # Training RMSE
    train_rmse = np.sqrt(np.mean((Y_train - Y_train_pred)**2))
    
    print(f"✓ Training R² = {r2_train:.4f}")
    print(f"✓ Training RMSE = {train_rmse:.6f}")
    
    # Always compute spectral radius of companion matrix
    eigenvalue_threshold = rom_config.get('eigenvalue_threshold', None)
    # Also check in mvar config block
    if eigenvalue_threshold is None and 'models' in rom_config and 'mvar' in rom_config['models']:
        eigenvalue_threshold = rom_config['models']['mvar'].get('eigenvalue_threshold', None)
    
    scale_factor = 1.0
    rho_before = 0.0
    rho_after = 0.0
    
    # Build companion matrix and compute spectral radius (always)
    A_coef_check = mvar_model.coef_  # Shape: (R_POD, P_LAG * R_POD)
    companion_dim_check = P_LAG * R_POD
    C_check = np.zeros((companion_dim_check, companion_dim_check))
    C_check[:R_POD, :] = A_coef_check
    for k in range(P_LAG - 1):
        C_check[(k+1)*R_POD:(k+2)*R_POD, k*R_POD:(k+1)*R_POD] = np.eye(R_POD)
    eigenvalues_check = np.linalg.eigvals(C_check)
    moduli_check = np.abs(eigenvalues_check)
    rho_before = float(np.max(moduli_check))
    n_unstable_check = int(np.sum(moduli_check > 1.0))
    rho_after = rho_before  # Will be updated if scaling is applied
    print(f"\nSpectral radius of companion matrix ({companion_dim_check}×{companion_dim_check}):")
    print(f"   ρ = {rho_before:.6f}")
    print(f"   Unstable eigenvalues (|λ|>1): {n_unstable_check}/{companion_dim_check}")
    
    if eigenvalue_threshold is not None:
        # Reuse companion matrix already built above
        C = C_check.copy()
        
        if rho_before > eigenvalue_threshold:
            # ----------------------------------------------------------
            # Uniform coefficient scaling to enforce stability
            # ----------------------------------------------------------
            # Scale only the A coefficient rows of the companion matrix
            # (not the identity sub-diagonal blocks) iteratively until
            # the spectral radius is at or below the threshold.
            #
            # NOTE: Schur-based selective projection was tried (V2.1)
            # but FAILED — modifying diagonal entries of the upper-
            # triangular Schur factor corrupts off-diagonal coupling,
            # producing artifacts that amplify dynamics (R²=-35,594).
            # Uniform scaling is blunt but mathematically safe.
            # ----------------------------------------------------------
            for iteration in range(50):
                current_rho = np.max(np.abs(np.linalg.eigvals(C)))
                if current_rho <= eigenvalue_threshold:
                    break
                scale = eigenvalue_threshold / current_rho
                # Scale only the A coefficient block (first R_POD rows)
                C[:R_POD, :] *= scale
                mvar_model.coef_ *= scale
            
            # Scale intercept proportionally
            if mvar_model.intercept_ is not None:
                total_scale = eigenvalue_threshold / rho_before
                mvar_model.intercept_ *= total_scale
            
            rho_after = np.max(np.abs(np.linalg.eigvals(C)))
            
            # Log the eigenvalue spectrum after scaling
            evals_after = np.abs(np.linalg.eigvals(C))
            evals_sorted = np.sort(evals_after)[::-1]
            print(f"   ⚠️  Uniform scaling: ρ {rho_before:.4f} → {rho_after:.4f} ({iteration+1} iterations)")
            print(f"   Eigenvalue spectrum: [{', '.join(f'{e:.4f}' for e in evals_sorted[:8])}{'...' if len(evals_sorted) > 8 else ''}]")
        else:
            print(f"   ✓ Model is stable (ρ={rho_before:.4f} ≤ {eigenvalue_threshold})")
            rho_after = rho_before
    
    # Reshape MVAR coefficients to match stable pipeline format
    # Stable pipeline stores A_matrices as (p, d, d) and uses different structure
    A_matrices = mvar_model.coef_.reshape(R_POD, P_LAG, R_POD).transpose(1, 0, 2)  # (p, d, d)
    
    return {
        'model': mvar_model,
        'P_LAG': P_LAG,
        'RIDGE_ALPHA': RIDGE_ALPHA,
        'r2_train': r2_train,
        'train_rmse': train_rmse,
        'A_matrices': A_matrices,
        'rho_before': float(rho_before),
        'rho_after': float(rho_after),
        'R_POD': R_POD
    }


def save_mvar_model(mvar_data, mvar_dir):
    """
    Save MVAR model to disk in standard format.
    
    Parameters
    ----------
    mvar_data : dict
        MVAR model dictionary from train_mvar_model
    mvar_dir : Path
        Directory to save MVAR model
    """
    mvar_dir = Path(mvar_dir)
    mvar_dir.mkdir(parents=True, exist_ok=True)
    
    # Save MVAR model with stable pipeline keys
    np.savez_compressed(
        mvar_dir / "mvar_model.npz",
        A_matrices=mvar_data['A_matrices'],
        A_companion=mvar_data['model'].coef_,  # Store flat version as companion
        p=mvar_data['P_LAG'],  # Changed from p_lag to p
        r=mvar_data['R_POD'],  # Changed from R_POD to r
        alpha=mvar_data['RIDGE_ALPHA'],  # Changed from ridge_alpha to alpha
        train_r2=mvar_data['r2_train'],
        train_rmse=mvar_data['train_rmse'],
        rho_before=mvar_data['rho_before'],
        rho_after=mvar_data['rho_after']
    )
    
    print(f"✓ MVAR model saved to {mvar_dir}/mvar_model.npz")


# =========================================================================
# k-step MVAR Training (multi-step rollout loss via gradient descent)
# =========================================================================

def train_mvar_kstep(pod_data, rom_config, y_trajs=None):
    """
    Train MVAR model with k-step rollout loss using gradient descent.
    
    Instead of the standard 1-step Ridge regression, this minimizes:
    
        L = (1/K) Σ_{j=1}^{K} || y(t+j) - ŷ(t+j) ||²
    
    where ŷ(t+j) is obtained by autoregressively rolling out the VAR(p)
    model j steps from the true history y(t-p+1)...y(t).
    
    This requires gradient descent because the k-step loss involves
    chaining the linear predictor, making it non-trivial for k>1.
    
    Parameters
    ----------
    pod_data : dict
        POD data from build_pod_basis (same as train_mvar_model)
    rom_config : dict
        ROM configuration; the mvar block should contain:
        - lag: int (default 5)
        - ridge_alpha: float (default 1e-6)
        - kstep_k: int — number of rollout steps for loss (default 5)
        - kstep_lr: float — learning rate (default 1e-3)
        - kstep_epochs: int — max epochs (default 500)
        - kstep_patience: int — early stopping patience (default 50)
        - kstep_weights: str — 'uniform' or 'decay' (default 'uniform')
    y_trajs : list of np.ndarray, optional
        List of latent trajectories [T_rom, d]. If None, reconstructed
        from pod_data['X_latent'].
    
    Returns
    -------
    dict
        Same structure as train_mvar_model, compatible with save_mvar_model.
    """
    import torch
    import torch.nn as nn
    
    X_latent = pod_data['X_latent']
    M = pod_data['M']
    T_rom = pod_data['T_rom']
    R_POD = pod_data['R_POD']
    
    # Parse config
    if 'models' in rom_config and 'mvar' in rom_config['models']:
        mvar_config = rom_config['models']['mvar']
    else:
        mvar_config = rom_config
    
    P_LAG = mvar_config.get('lag', 5)
    RIDGE_ALPHA = mvar_config.get('ridge_alpha', 1e-6)
    K_STEPS = mvar_config.get('kstep_k', 5)
    LR = mvar_config.get('kstep_lr', 1e-3)
    MAX_EPOCHS = mvar_config.get('kstep_epochs', 500)
    PATIENCE = mvar_config.get('kstep_patience', 50)
    WEIGHT_SCHEME = mvar_config.get('kstep_weights', 'uniform')
    
    print(f"\nTraining k-step MVAR (p={P_LAG}, k={K_STEPS}, α={RIDGE_ALPHA})...")
    print(f"   Optimizer: Adam, lr={LR}, max_epochs={MAX_EPOCHS}, patience={PATIENCE}")
    print(f"   Weight scheme: {WEIGHT_SCHEME}")
    
    # ---- Step 1: Initialize with Ridge (warm start) ----
    print(f"\n   Phase 1: Warm-start with 1-step Ridge regression...")
    
    # Build trajectories if not provided
    if y_trajs is None:
        X_latent_runs = X_latent.reshape(M, T_rom, R_POD)
        y_trajs = [X_latent_runs[m] for m in range(M)]
    
    # Build 1-step dataset for warm start
    X_train_list, Y_train_list = [], []
    for y_m in y_trajs:
        T_m = len(y_m)
        for t in range(P_LAG, T_m):
            X_train_list.append(y_m[t-P_LAG:t].flatten())
            Y_train_list.append(y_m[t])
    X_train = np.array(X_train_list)
    Y_train = np.array(Y_train_list)
    
    ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    ridge.fit(X_train, Y_train)
    
    # Compute 1-step baseline R²
    Y_pred_1s = ridge.predict(X_train)
    ss_res_1s = np.sum((Y_train - Y_pred_1s)**2)
    ss_tot_1s = np.sum((Y_train - Y_train.mean(axis=0))**2)
    r2_1step = 1 - ss_res_1s / ss_tot_1s
    print(f"   1-step Ridge R² = {r2_1step:.4f}")
    
    # ---- Step 2: Build k-step training windows ----
    # Each sample: history [p, d], future_targets [K, d]
    windows_hist = []  # [N, p, d]
    windows_future = []  # [N, K, d]
    
    for y_m in y_trajs:
        T_m = len(y_m)
        for t in range(P_LAG, T_m - K_STEPS):
            windows_hist.append(y_m[t-P_LAG:t])        # [p, d]
            windows_future.append(y_m[t:t+K_STEPS])    # [K, d]
    
    windows_hist = np.array(windows_hist)    # [N, p, d]
    windows_future = np.array(windows_future)  # [N, K, d]
    N_samples = len(windows_hist)
    
    print(f"   k-step dataset: {N_samples} windows (p={P_LAG}, K={K_STEPS})")
    
    # ---- Step 3: Define differentiable VAR model ----
    device = torch.device('cpu')  # MVAR is tiny, CPU is fine
    
    # VAR(p): y(t+1) = A @ [y(t-p+1); ...; y(t)] + b
    # A shape: [d, p*d], b shape: [d]
    A = torch.tensor(ridge.coef_, dtype=torch.float64, device=device, requires_grad=True)
    b = torch.tensor(ridge.intercept_, dtype=torch.float64, device=device, requires_grad=True)
    
    # Step weights for the k-step loss
    if WEIGHT_SCHEME == 'decay':
        step_weights = torch.tensor([1.0 / (j+1) for j in range(K_STEPS)],
                                     dtype=torch.float64, device=device)
    else:  # uniform
        step_weights = torch.ones(K_STEPS, dtype=torch.float64, device=device)
    step_weights = step_weights / step_weights.sum()
    
    # Convert data to tensors
    hist_t = torch.tensor(windows_hist, dtype=torch.float64, device=device)    # [N, p, d]
    future_t = torch.tensor(windows_future, dtype=torch.float64, device=device)  # [N, K, d]
    
    # Train/val split (80/20)
    n_val = max(1, int(0.2 * N_samples))
    n_train = N_samples - n_val
    perm = torch.randperm(N_samples)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    
    # L2 penalty coefficient (matches Ridge alpha)
    l2_coef = RIDGE_ALPHA / n_train
    
    # ---- Step 4: Optimize with Adam ----
    print(f"\n   Phase 2: k-step optimization (Adam)...")
    optimizer = torch.optim.Adam([A, b], lr=LR)
    
    best_val_loss = float('inf')
    best_A = A.clone().detach()
    best_b = b.clone().detach()
    epochs_no_improve = 0
    
    def kstep_loss(A, b, hist, future, weights):
        """Compute k-step rollout loss for a batch."""
        bsz = hist.shape[0]
        d = A.shape[0]
        p = hist.shape[1]
        K = future.shape[1]
        
        total_loss = torch.tensor(0.0, dtype=torch.float64, device=device)
        current = hist.clone()  # [bsz, p, d]
        
        for j in range(K):
            # Flatten history: [bsz, p*d]
            x_flat = current.reshape(bsz, -1)
            # Predict: y_next = x_flat @ A^T + b
            y_next = x_flat @ A.t() + b.unsqueeze(0)  # [bsz, d]
            # Loss at step j
            step_loss = ((y_next - future[:, j, :]) ** 2).mean()
            total_loss = total_loss + weights[j] * step_loss
            # Slide window
            current = torch.cat([current[:, 1:, :], y_next.unsqueeze(1)], dim=1)
        
        # L2 penalty on A
        l2_pen = l2_coef * (A ** 2).sum()
        
        return total_loss + l2_pen
    
    BATCH_SIZE = min(512, n_train)
    
    for epoch in range(MAX_EPOCHS):
        # Mini-batch training
        shuf = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0
        
        for start in range(0, n_train, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_train)
            idx = train_idx[shuf[start:end]]
            
            optimizer.zero_grad()
            loss = kstep_loss(A, b, hist_t[idx], future_t[idx], step_weights)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        epoch_loss /= n_batches
        
        # Validation
        with torch.no_grad():
            val_loss = kstep_loss(A, b, hist_t[val_idx], future_t[val_idx], step_weights).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_A = A.clone().detach()
            best_b = b.clone().detach()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epoch % 50 == 0 or epochs_no_improve == 0:
            print(f"   Epoch {epoch:4d}: train={epoch_loss:.6f}, val={val_loss:.6f}"
                  f"{'  *' if epochs_no_improve == 0 else ''}")
        
        if epochs_no_improve >= PATIENCE:
            print(f"   Early stopping at epoch {epoch} (patience={PATIENCE})")
            break
    
    print(f"   Best val loss: {best_val_loss:.6f}")
    
    # ---- Step 5: Package as sklearn-compatible model ----
    # Create a Ridge object and overwrite its coefficients
    final_model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    final_model.fit(X_train[:2], Y_train[:2])  # Dummy fit to initialize
    final_model.coef_ = best_A.numpy()
    final_model.intercept_ = best_b.numpy()
    
    # Compute final training R² (1-step, for comparison)
    Y_pred_final = final_model.predict(X_train)
    ss_res_f = np.sum((Y_train - Y_pred_final)**2)
    r2_train = 1 - ss_res_f / ss_tot_1s
    train_rmse = np.sqrt(np.mean((Y_train - Y_pred_final)**2))
    
    print(f"\n   Final 1-step R² = {r2_train:.4f} (was {r2_1step:.4f} from Ridge)")
    
    # Spectral radius
    A_coef = final_model.coef_
    companion_dim = P_LAG * R_POD
    C = np.zeros((companion_dim, companion_dim))
    C[:R_POD, :] = A_coef
    for k in range(P_LAG - 1):
        C[(k+1)*R_POD:(k+2)*R_POD, k*R_POD:(k+1)*R_POD] = np.eye(R_POD)
    eigenvalues = np.linalg.eigvals(C)
    rho = float(np.max(np.abs(eigenvalues)))
    
    print(f"   Spectral radius: ρ = {rho:.6f}")
    
    A_matrices = final_model.coef_.reshape(R_POD, P_LAG, R_POD).transpose(1, 0, 2)
    
    return {
        'model': final_model,
        'P_LAG': P_LAG,
        'RIDGE_ALPHA': RIDGE_ALPHA,
        'r2_train': r2_train,
        'train_rmse': train_rmse,
        'A_matrices': A_matrices,
        'rho_before': rho,
        'rho_after': rho,
        'R_POD': R_POD,
        'kstep_k': K_STEPS,
        'kstep_val_loss': best_val_loss,
    }
