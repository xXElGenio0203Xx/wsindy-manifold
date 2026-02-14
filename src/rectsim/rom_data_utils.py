"""
ROM Data Utilities for MVAR and LSTM Training

Provides utilities for building windowed datasets from POD latent trajectories
that can be shared between MVAR and LSTM models.
"""

import numpy as np
from typing import List, Tuple


def build_latent_dataset(y_trajs: List[np.ndarray], lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build windowed dataset from POD latent trajectories for ROM training.
    
    This function creates a shared dataset that can be used by BOTH MVAR and LSTM
    models. Each sample consists of a window of past latent states (X) and the
    next latent state to predict (Y).
    
    Parameters
    ----------
    y_trajs : list of np.ndarray
        List of latent trajectories from POD, one per training simulation.
        Each array has shape [K_c, d] where:
            K_c = number of time steps for trajectory c
            d = latent dimension (number of POD modes)
    
    lag : int
        Window length (number of past time steps to use as input).
        Must match rom.models.*.lag from the config.
        For MVAR: this is the order p
        For LSTM: this is the sequence length
    
    Returns
    -------
    X_all : np.ndarray
        Input windows, shape [N_samples, lag, d]
        Each row is a sequence: [y(t_{k-lag}), ..., y(t_{k-1})]
    
    Y_all : np.ndarray
        Target states, shape [N_samples, d]
        Each row is the next state: y(t_k)
    
    Examples
    --------
    >>> # Example with 3 trajectories
    >>> traj1 = np.random.randn(100, 10)  # 100 timesteps, 10 modes
    >>> traj2 = np.random.randn(100, 10)
    >>> traj3 = np.random.randn(100, 10)
    >>> y_trajs = [traj1, traj2, traj3]
    >>> lag = 5
    >>> X_all, Y_all = build_latent_dataset(y_trajs, lag)
    >>> print(X_all.shape)  # (285, 5, 10) = 3 * (100 - 5) samples
    >>> print(Y_all.shape)  # (285, 10)
    
    Notes
    -----
    - N_samples = sum_c (K_c - lag) over all training trajectories
    - For MVAR: X_all will be reshaped to [N_samples, lag*d] before regression
    - For LSTM: X_all is used directly as sequence input
    - The dataset is identical for both models to ensure fair comparison
    """
    
    # Validate inputs
    if not y_trajs:
        raise ValueError("y_trajs must be a non-empty list of trajectories")
    
    if lag <= 0:
        raise ValueError(f"lag must be positive, got {lag}")
    
    # Check that all trajectories have the same latent dimension
    d_values = [y.shape[1] for y in y_trajs]
    if len(set(d_values)) > 1:
        raise ValueError(f"All trajectories must have same latent dimension, got {set(d_values)}")
    
    d = d_values[0]
    
    # Validate that trajectories are long enough
    min_length = min(y.shape[0] for y in y_trajs)
    if min_length <= lag:
        raise ValueError(
            f"All trajectories must have length > lag. "
            f"Got min_length={min_length}, lag={lag}"
        )
    
    # Initialize lists to collect windows and targets
    X_list = []
    Y_list = []
    
    # Loop over all trajectories
    for traj_idx, Y_c in enumerate(y_trajs):
        K_c, d_c = Y_c.shape
        
        # For each valid time index k, build a (window, target) pair
        for k in range(lag, K_c):
            # Window: from k-lag to k-1 (inclusive), shape [lag, d]
            X_window = Y_c[k - lag : k, :]  # [lag, d]
            
            # Target: next state at time k, shape [d]
            Y_target = Y_c[k, :]  # [d]
            
            # Append to lists
            X_list.append(X_window)
            Y_list.append(Y_target)
    
    # Stack into arrays
    X_all = np.stack(X_list, axis=0)  # [N_samples, lag, d]
    Y_all = np.stack(Y_list, axis=0)  # [N_samples, d]
    
    # Validate output shapes
    N_samples = X_all.shape[0]
    expected_samples = sum(y.shape[0] - lag for y in y_trajs)
    
    assert X_all.shape == (N_samples, lag, d), \
        f"Expected X_all.shape = ({N_samples}, {lag}, {d}), got {X_all.shape}"
    assert Y_all.shape == (N_samples, d), \
        f"Expected Y_all.shape = ({N_samples}, {d}), got {Y_all.shape}"
    assert N_samples == expected_samples, \
        f"Expected {expected_samples} samples, got {N_samples}"
    
    return X_all, Y_all


def build_multistep_latent_dataset(
    y_trajs: List[np.ndarray], lag: int, k_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build windowed dataset with k-step-ahead targets for multi-step rollout loss.
    
    Each sample has a lag-length input window and k future targets (1-step, 2-step,
    ... k-step ahead). Only windows where all k future steps are available within
    the same trajectory are included.
    
    Parameters
    ----------
    y_trajs : list of np.ndarray
        List of latent trajectories, each shape [K_c, d].
    lag : int
        Window length.
    k_steps : int
        Number of future steps to include as targets.
    
    Returns
    -------
    X_all : np.ndarray, shape [N_samples, lag, d]
        Input windows.
    Y_multi : np.ndarray, shape [N_samples, k_steps, d]
        Multi-step targets: Y_multi[i, j, :] = y(t + j + 1) for sample i.
    """
    if k_steps < 1:
        raise ValueError(f"k_steps must be >= 1, got {k_steps}")
    
    d = y_trajs[0].shape[1]
    X_list = []
    Y_list = []
    
    for Y_c in y_trajs:
        K_c = Y_c.shape[0]
        # Need lag input steps + k_steps future steps
        for t in range(lag, K_c - k_steps + 1):
            X_list.append(Y_c[t - lag : t, :])                # [lag, d]
            Y_list.append(Y_c[t : t + k_steps, :])            # [k_steps, d]
    
    X_all = np.stack(X_list, axis=0)      # [N, lag, d]
    Y_multi = np.stack(Y_list, axis=0)    # [N, k_steps, d]
    
    return X_all, Y_multi


def print_dataset_info(X_all: np.ndarray, Y_all: np.ndarray, lag: int, verbose: bool = True):
    """
    Print information about the windowed dataset.
    
    Parameters
    ----------
    X_all : np.ndarray
        Input windows, shape [N_samples, lag, d]
    Y_all : np.ndarray
        Target states, shape [N_samples, d]
    lag : int
        Window length
    verbose : bool, optional
        If True, print detailed statistics. Default: True
    """
    
    N_samples, lag_check, d = X_all.shape
    
    if verbose:
        print(f"\n{'='*80}")
        print("Windowed Latent Dataset")
        print(f"{'='*80}")
        print(f"  Input shape (X):  {X_all.shape}  [N_samples, lag, d]")
        print(f"  Output shape (Y): {Y_all.shape}  [N_samples, d]")
        print(f"  Number of samples: {N_samples:,}")
        print(f"  Window length (lag): {lag}")
        print(f"  Latent dimension (d): {d}")
        print(f"  Total parameters (MVAR): {lag * d * d:,}  (lag × d × d)")
        print(f"\n  X statistics:")
        print(f"    mean: {np.mean(X_all):.6f}")
        print(f"    std:  {np.std(X_all):.6f}")
        print(f"    min:  {np.min(X_all):.6f}")
        print(f"    max:  {np.max(X_all):.6f}")
        print(f"\n  Y statistics:")
        print(f"    mean: {np.mean(Y_all):.6f}")
        print(f"    std:  {np.std(Y_all):.6f}")
        print(f"    min:  {np.min(Y_all):.6f}")
        print(f"    max:  {np.max(Y_all):.6f}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    """Test the dataset builder with dummy data."""
    
    print("\nTesting build_latent_dataset with dummy data...\n")
    
    # Test 1: Small example
    print("Test 1: Small example (3 trajectories, lag=5)")
    np.random.seed(42)
    traj1 = np.random.randn(100, 10)  # 100 timesteps, 10 modes
    traj2 = np.random.randn(100, 10)
    traj3 = np.random.randn(100, 10)
    y_trajs = [traj1, traj2, traj3]
    lag = 5
    
    X_all, Y_all = build_latent_dataset(y_trajs, lag)
    print_dataset_info(X_all, Y_all, lag)
    
    # Verify expected number of samples
    expected_samples = 3 * (100 - 5)  # 3 trajectories × 95 samples each
    assert X_all.shape[0] == expected_samples, f"Expected {expected_samples} samples"
    print(f"✓ Verified: {expected_samples} samples generated\n")
    
    # Test 2: Varying trajectory lengths
    print("Test 2: Varying trajectory lengths")
    traj1 = np.random.randn(80, 15)   # 80 timesteps, 15 modes
    traj2 = np.random.randn(120, 15)  # 120 timesteps
    traj3 = np.random.randn(100, 15)  # 100 timesteps
    y_trajs = [traj1, traj2, traj3]
    lag = 10
    
    X_all, Y_all = build_latent_dataset(y_trajs, lag)
    expected_samples = (80 - 10) + (120 - 10) + (100 - 10)  # 70 + 110 + 90 = 270
    assert X_all.shape[0] == expected_samples, f"Expected {expected_samples} samples"
    print(f"  Shape: X{X_all.shape}, Y{Y_all.shape}")
    print(f"✓ Verified: {expected_samples} samples (70 + 110 + 90)\n")
    
    # Test 3: MVAR reshape demonstration
    print("Test 3: MVAR reshape (flatten lag dimension)")
    X_all, Y_all = build_latent_dataset([traj1, traj2, traj3], lag)
    X_mvar = X_all.reshape(X_all.shape[0], -1)  # [N_samples, lag*d]
    print(f"  Original X shape: {X_all.shape}  [N_samples, lag, d]")
    print(f"  MVAR X shape:     {X_mvar.shape}  [N_samples, lag*d]")
    print(f"  Y shape:          {Y_all.shape}  [N_samples, d]")
    print(f"✓ MVAR would fit: Y ~ X_mvar @ A^T + b\n")
    
    # Test 4: Error handling
    print("Test 4: Error handling")
    try:
        # Lag too large
        build_latent_dataset([np.random.randn(5, 10)], lag=10)
        print("✗ Should have raised ValueError for lag > trajectory length")
    except ValueError as e:
        print(f"✓ Caught expected error: {e}\n")
    
    try:
        # Inconsistent dimensions
        build_latent_dataset([np.random.randn(100, 10), np.random.randn(100, 15)], lag=5)
        print("✗ Should have raised ValueError for inconsistent dimensions")
    except ValueError as e:
        print(f"✓ Caught expected error: {e}\n")
    
    print("="*80)
    print("✅ All tests passed!")
    print("="*80)
