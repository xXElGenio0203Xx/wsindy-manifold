"""
MVAR Trainer Module
===================

Trains Multivariate AutoRegressive (MVAR) models on POD latent space.
Supports:
- Ridge regression with configurable regularization
- Optional eigenvalue stability enforcement
- Companion matrix form
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
    
    # Optional: Eigenvalue stability check via FULL companion matrix
    eigenvalue_threshold = rom_config.get('eigenvalue_threshold', None)
    # Also check in mvar config block
    if eigenvalue_threshold is None and 'models' in rom_config and 'mvar' in rom_config['models']:
        eigenvalue_threshold = rom_config['models']['mvar'].get('eigenvalue_threshold', None)
    
    scale_factor = 1.0
    rho_before = 0.0
    rho_after = 0.0
    
    if eigenvalue_threshold is not None:
        # Build FULL companion matrix for VAR(p):
        #   C = [A_1  A_2  ...  A_p]
        #       [ I    0   ...   0 ]
        #       [ 0    I   ...   0 ]
        #       [ ...            0 ]
        # where A_k are the (R_POD × R_POD) coefficient matrices
        A_coef = mvar_model.coef_  # Shape: (R_POD, P_LAG * R_POD)
        
        companion_dim = P_LAG * R_POD
        C = np.zeros((companion_dim, companion_dim))
        
        # First block row: the A matrices
        C[:R_POD, :] = A_coef  # [A_1, A_2, ..., A_p]
        
        # Identity blocks on the sub-diagonal
        for k in range(P_LAG - 1):
            C[(k+1)*R_POD:(k+2)*R_POD, k*R_POD:(k+1)*R_POD] = np.eye(R_POD)
        
        eigenvalues = np.linalg.eigvals(C)
        moduli = np.abs(eigenvalues)
        rho_before = np.max(moduli)
        n_unstable = np.sum(moduli > 1.0)
        
        print(f"\nStability check (FULL companion matrix {companion_dim}×{companion_dim}):")
        print(f"   Spectral radius ρ = {rho_before:.6f}")
        print(f"   Unstable eigenvalues (|λ|>1): {n_unstable}/{companion_dim}")
        
        if rho_before > eigenvalue_threshold:
            # Scale ALL coefficients uniformly to bring spectral radius to threshold
            # For companion matrix: scaling coefs by s scales eigenvalues by s
            # (identity blocks stay, so this is approximate — iterate to converge)
            for iteration in range(10):
                scale = eigenvalue_threshold / np.max(np.abs(np.linalg.eigvals(C)))
                C[:R_POD, :] *= scale
                mvar_model.coef_ *= scale
                if mvar_model.intercept_ is not None:
                    mvar_model.intercept_ *= scale
                current_rho = np.max(np.abs(np.linalg.eigvals(C)))
                if current_rho <= eigenvalue_threshold:
                    break
            
            rho_after = np.max(np.abs(np.linalg.eigvals(C)))
            print(f"   ⚠️  Scaled coefficients: ρ {rho_before:.4f} → {rho_after:.4f} (threshold={eigenvalue_threshold})")
            print(f"   Iterations: {iteration + 1}")
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
