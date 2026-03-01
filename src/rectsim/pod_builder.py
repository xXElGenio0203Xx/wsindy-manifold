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

from rectsim.shift_align import align_training_data


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
        - pod_energy / energy_threshold: energy threshold (default: 0.995)
        - density_transform: optional transform before POD
          'raw' (default), 'log', 'sqrt', 'meansub'
        - density_transform_eps: epsilon for log/sqrt (default: 1e-8)
        Priority: fixed_modes/fixed_d > energy_threshold
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
        - density_transform: which transform was applied
        - density_transform_eps: epsilon used
    """
    
    ROM_SUBSAMPLE = rom_config.get('subsample', rom_config.get('rom_subsample', 1))
    
    print(f"\nLoading training density data (subsample={ROM_SUBSAMPLE})...")
    
    # Load all training density data
    X_list = []
    density_shape_2d = None  # (Ny, Nx) — captured from first run
    for i in range(n_train):
        run_dir = train_dir / f"train_{i:03d}"
        data = np.load(run_dir / "density.npz")
        density = data[density_key]
        
        # Subsample in time if requested
        if ROM_SUBSAMPLE > 1:
            density = density[::ROM_SUBSAMPLE]
        
        # Capture spatial dimensions from first run
        if density_shape_2d is None:
            density_shape_2d = density.shape[1:]  # (Ny, Nx)
        
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
    
    # ------ Optional shift alignment (remove translational motion) ------
    shift_align = rom_config.get('shift_align', False)
    shift_align_ref = rom_config.get('shift_align_ref', 'mean')
    shift_align_data = None
    unaligned_svd = None  # Will hold SVD of raw data when alignment is used
    
    if shift_align:
        # Compute unaligned POD spectra BEFORE alignment (for comparison plots)
        save_unaligned = rom_config.get('save_unaligned_pod', True)
        if save_unaligned:
            print("\nComputing unaligned POD spectrum (for comparison)...")
            X_raw_mean = X_all.mean(axis=0)
            X_raw_centered = X_all - X_raw_mean
            _, S_raw, _ = np.linalg.svd(X_raw_centered.T, full_matrices=False)
            raw_total = np.sum(S_raw**2)
            raw_cum = np.cumsum(S_raw**2) / raw_total
            unaligned_svd = {
                'S': S_raw,
                'total_energy': raw_total,
                'cumulative_energy': raw_cum,
            }
            print(f"  ✓ Unaligned spectrum: {len(S_raw)} singular values")
            del X_raw_centered  # Free memory
        
        Ny, Nx = density_shape_2d
        print(f"\nApplying shift alignment (ref={shift_align_ref})...")
        densities_2d = X_all.reshape(-1, Ny, Nx)
        sa_result = align_training_data(densities_2d, M, T_rom, ref_method=shift_align_ref)
        X_all = sa_result['aligned'].reshape(-1, Ny * Nx)
        shift_align_data = {
            'ref': sa_result['ref'],
            'shifts': sa_result['shifts'],
            'ref_method': sa_result['ref_method'],
            'density_shape_2d': density_shape_2d,
        }
        print(f"  ✓ Shift alignment complete")
    
    # Apply density transform if requested
    density_transform = rom_config.get('density_transform', 'raw')
    density_transform_eps = rom_config.get('density_transform_eps', 1e-8)
    
    if density_transform == 'log':
        print(f"\nApplying log transform: log(rho + {density_transform_eps})")
        X_all = np.log(X_all + density_transform_eps)
    elif density_transform == 'sqrt':
        print(f"\nApplying sqrt transform: sqrt(rho + {density_transform_eps})")
        X_all = np.sqrt(X_all + density_transform_eps)
    elif density_transform == 'meansub':
        # Per-snapshot mean subtraction (remove spatially-uniform component)
        # Note: this is BEFORE POD mean-centering (which is across all snapshots)
        snapshot_means = X_all.mean(axis=1, keepdims=True)
        X_all = X_all - snapshot_means
        print(f"\nApplying meansub transform: rho - mean(rho) per snapshot")
    elif density_transform in ('raw', 'none'):
        print(f"\nNo density transform (raw)")
    else:
        raise ValueError(f"Unknown density_transform: '{density_transform}'. Use 'raw', 'log', 'sqrt', or 'meansub'.")
    
    # Compute POD
    print("\nComputing global POD...")
    X_mean = X_all.mean(axis=0)
    X_centered = X_all - X_mean
    
    U, S, Vt = np.linalg.svd(X_centered.T, full_matrices=False)
    
    # Determine number of modes
    # Priority: fixed_modes/fixed_d (if specified) > pod_energy/energy_threshold (threshold)
    FIXED_D = rom_config.get('fixed_modes', None)  # Check 'fixed_modes' first (standard name)
    if FIXED_D is None:
        FIXED_D = rom_config.get('fixed_d', None)  # Fall back to 'fixed_d' for backward compatibility
    
    TARGET_ENERGY = rom_config.get('pod_energy', rom_config.get('energy_threshold', 0.995))
    
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
        'T_rom': T_rom,
        'density_transform': density_transform,
        'density_transform_eps': density_transform_eps,
        'shift_align': shift_align,
        'shift_align_data': shift_align_data,
        'unaligned_svd': unaligned_svd,
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
    
    # Save unaligned POD spectrum if computed (for eigenvalue decay comparison)
    unaligned_svd = pod_data.get('unaligned_svd', None)
    if unaligned_svd is not None:
        np.savez_compressed(
            mvar_dir / "pod_basis_unaligned.npz",
            all_singular_values=unaligned_svd['S'],
            total_energy=unaligned_svd['total_energy'],
            cumulative_ratio=unaligned_svd['cumulative_energy'],
        )
        print(f"✓ Unaligned POD spectrum saved to {mvar_dir}/pod_basis_unaligned.npz")
    
    # Save shift alignment data if present
    sa_data = pod_data.get('shift_align_data', None)
    if sa_data is not None:
        np.savez_compressed(
            mvar_dir / "shift_align.npz",
            ref=sa_data['ref'],
            shifts=sa_data['shifts'],
            ref_method=str(sa_data['ref_method']),
            density_shape_2d=np.array(sa_data['density_shape_2d']),
        )
        print(f"✓ Shift alignment data saved to {mvar_dir}/shift_align.npz")
