"""
POD Builder Module
==================

Constructs Proper Orthogonal Decomposition (POD) basis from training data.
Supports:
- Fixed mode count or energy threshold
- Subsampling in time
- Mean-centering
"""

import numpy as np
from pathlib import Path


def build_pod_basis(train_dir, n_train, rom_config, density_key='rho'):
    """
    Build POD basis from training density data.
    
    Parameters
    ----------
    train_dir : Path
        Directory containing training runs
    n_train : int
        Number of training runs
    rom_config : dict
        ROM configuration with keys:
        - subsample: temporal subsampling factor (default: 1)
        - fixed_modes: fixed number of modes (optional, takes priority)
        - fixed_d: fallback name for fixed_modes
        - pod_energy: energy threshold (default: 0.995)
    density_key : str
        Key for density data in npz files (default: 'rho')
    
    Returns
    -------
    dict
        POD data dictionary with keys:
        - U_r: POD basis (n_spatial, n_modes)
        - S: all singular values
        - X_mean: mean field
        - X_all: all training data (centered)
        - R_POD: number of modes
        - energy_captured: actual energy captured
        - cumulative_energy: cumulative energy ratios
        - total_energy: total energy
        - M: number of training runs
        - T_rom: timesteps per run
    """
    
    ROM_SUBSAMPLE = rom_config.get('subsample', rom_config.get('rom_subsample', 1))
    
    print(f"\nLoading training density data (subsample={ROM_SUBSAMPLE})...")
    
    # Load all training density data
    X_list = []
    for i in range(n_train):
        run_dir = train_dir / f"train_{i:03d}"
        data = np.load(run_dir / "density.npz")
        density = data[density_key]
        
        # Subsample in time if requested
        if ROM_SUBSAMPLE > 1:
            density = density[::ROM_SUBSAMPLE]
        
        # Flatten each timestep
        T_sub = density.shape[0]
        X_run = density.reshape(T_sub, -1)
        X_list.append(X_run)
    
    # Stack all data
    X_all = np.vstack(X_list)
    M = n_train
    T_rom = X_list[0].shape[0]
    
    print(f"✓ Loaded data shape: {X_all.shape}")
    print(f"   {M} runs × {T_rom} timesteps × {X_all.shape[1]} spatial dims")
    
    # Compute POD
    print("\nComputing global POD...")
    X_mean = X_all.mean(axis=0)
    X_centered = X_all - X_mean
    
    U, S, Vt = np.linalg.svd(X_centered.T, full_matrices=False)
    
    # Determine number of modes
    # Priority: fixed_modes/fixed_d (if specified) > pod_energy (threshold)
    FIXED_D = rom_config.get('fixed_modes', None)  # Check 'fixed_modes' first (standard name)
    if FIXED_D is None:
        FIXED_D = rom_config.get('fixed_d', None)  # Fall back to 'fixed_d' for backward compatibility
    
    TARGET_ENERGY = rom_config.get('pod_energy', 0.995)
    
    total_energy = np.sum(S**2)
    cumulative_energy = np.cumsum(S**2) / total_energy
    
    if FIXED_D is not None:
        # Use fixed dimension (PRIORITY: explicit mode count overrides energy threshold)
        R_POD = min(FIXED_D, len(S))
        energy_captured = cumulative_energy[R_POD - 1]
        print(f"✓ Using FIXED d={R_POD} modes (energy={energy_captured:.4f}, hard cap from config)")
    else:
        # Use energy threshold
        R_POD = np.searchsorted(cumulative_energy, TARGET_ENERGY) + 1
        energy_captured = cumulative_energy[R_POD - 1]
        print(f"✓ R_POD = {R_POD} modes (energy={energy_captured:.4f}, threshold={TARGET_ENERGY})")
    
    U_r = U[:, :R_POD]
    
    # Project to latent space
    X_latent = X_centered @ U_r
    print(f"✓ Latent training data shape: ({M*T_rom}, {R_POD})")
    
    return {
        'U_r': U_r,
        'S': S,
        'X_mean': X_mean,
        'X_centered': X_centered,
        'X_latent': X_latent,
        'R_POD': R_POD,
        'energy_captured': energy_captured,
        'cumulative_energy': cumulative_energy,
        'total_energy': total_energy,
        'M': M,
        'T_rom': T_rom
    }


def save_pod_basis(pod_data, mvar_dir):
    """
    Save POD basis to disk in standard format.
    
    Parameters
    ----------
    pod_data : dict
        POD data dictionary from build_pod_basis
    mvar_dir : Path
        Directory to save POD basis
    """
    mvar_dir = Path(mvar_dir)
    mvar_dir.mkdir(parents=True, exist_ok=True)
    
    # Save mean separately (stable pipeline format)
    np.save(mvar_dir / "X_train_mean.npy", pod_data['X_mean'])
    
    # Save POD basis with stable pipeline keys
    np.savez_compressed(
        mvar_dir / "pod_basis.npz",
        U=pod_data['U_r'],
        singular_values=pod_data['S'][:pod_data['R_POD']],
        all_singular_values=pod_data['S'],
        total_energy=pod_data['total_energy'],
        explained_energy=pod_data['cumulative_energy'][pod_data['R_POD']-1] * pod_data['total_energy'],
        energy_ratio=pod_data['energy_captured'],
        cumulative_ratio=pod_data['cumulative_energy']
    )
    
    print(f"✓ POD basis saved to {mvar_dir}/pod_basis.npz")
