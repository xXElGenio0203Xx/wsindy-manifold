"""EF-ROM data handling and global POD basis computation.

This module implements the data pipeline for the Empirical Flow Reduced-Order Model
(EF-ROM) approach, inspired by Alvarez et al. "Next Generation Equation-Free
Multiscale Modelling of Crowd Dynamics via Machine Learning".

Key components:
- Load density movies from ensemble simulations
- Compute global POD (PCA) basis across multiple runs
- Project density fields to latent coordinates
- Reconstruct density fields from latent coordinates
- Prepare train/test datasets for MVAR training

The workflow is:
1. Generate ensemble with varied ICs (see ENSEMBLE_GUIDE.md)
2. Load density movies: load_density_runs()
3. Build snapshot matrix: build_snapshot_matrix()
4. Compute global POD: compute_pod()
5. Project to latent space: project_onto_pod()
6. Prepare MVAR dataset: build_latent_dataset()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class PODBasis:
    """Global POD basis for density field compression.

    Attributes
    ----------
    modes : np.ndarray, shape (d, r)
        Orthonormal POD modes in vectorized grid space. Columns are modes.
    singular_values : np.ndarray, shape (r,)
        Singular values corresponding to each mode.
    mean : np.ndarray, shape (d,)
        Spatial mean used for centering before projection.
    grid_shape : tuple of (int, int)
        Grid dimensions (nx, ny) of the original density field.
    energy : np.ndarray, shape (r,)
        Cumulative energy fraction captured by modes 0 through k.

    Notes
    -----
    The POD modes are computed via SVD of the centered snapshot matrix:
    X = U Σ V^T, where U contains the spatial modes (eigenvectors of X X^T).
    
    Energy is defined as the cumulative sum of squared singular values:
    energy[k] = sum(S[0:k+1]**2) / sum(S**2)
    """

    modes: np.ndarray
    singular_values: np.ndarray
    mean: np.ndarray
    grid_shape: Tuple[int, int]
    energy: np.ndarray

    def __post_init__(self):
        """Validate dimensions."""
        d, r = self.modes.shape
        assert self.singular_values.shape == (r,), "singular_values must have shape (r,)"
        assert self.mean.shape == (d,), "mean must have shape (d,)"
        assert self.energy.shape == (r,), "energy must have shape (r,)"
        assert len(self.grid_shape) == 2, "grid_shape must be (nx, ny)"
        nx, ny = self.grid_shape
        assert d == nx * ny, f"modes.shape[0]={d} must equal nx*ny={nx*ny}"

    @property
    def n_modes(self) -> int:
        """Number of POD modes retained."""
        return self.modes.shape[1]

    @property
    def spatial_dim(self) -> int:
        """Spatial dimension (d = nx * ny)."""
        return self.modes.shape[0]


def load_density_runs(
    root: Path,
    pattern: str = "**/density.npz",
    max_runs: int | None = None,
    t_slice: slice | None = None,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """Load density movies from ensemble simulation runs.

    Scans the directory tree under `root` for files matching `pattern`,
    loads the density field from each, and returns a list of arrays and
    metadata.

    Parameters
    ----------
    root : Path
        Root directory containing simulation outputs
        (e.g., simulations/<model_id>/).
    pattern : str, optional
        Glob pattern for finding density files. Default is "**/density.npz".
    max_runs : int or None, optional
        Maximum number of runs to load. If None, load all found runs.
    t_slice : slice or None, optional
        Time slice to extract from each density movie. If None, use all
        time steps. Example: slice(0, -10) excludes last 10 frames.

    Returns
    -------
    densities : list of np.ndarray
        Each element has shape (T_i, nx, ny) for run i.
    metadata : list of dict
        Metadata for each run with keys:
        - 'run_path': Path to the run directory
        - 'run_id': Run identifier (directory name)
        - 'T': Number of time steps
        - 'grid_shape': (nx, ny)
        - 'run_index': Index in the list

    Examples
    --------
    >>> from pathlib import Path
    >>> root = Path("simulations/social_force_N200_T100")
    >>> densities, meta = load_density_runs(root, max_runs=5)
    >>> len(densities)
    5
    >>> densities[0].shape
    (1001, 128, 128)

    Notes
    -----
    Assumes each density.npz file contains a 'rho' field with shape
    (T, nx, ny) or (T, nx, ny, 1). The singleton channel dimension is
    squeezed if present.
    """
    root = Path(root)
    density_files = sorted(root.rglob(pattern))

    if max_runs is not None:
        density_files = density_files[:max_runs]

    densities = []
    metadata = []

    for idx, density_path in enumerate(density_files):
        # Load density field
        data = np.load(density_path)
        rho = data["rho"]

        # Ensure shape is (T, nx, ny)
        if rho.ndim == 4 and rho.shape[-1] == 1:
            rho = rho.squeeze(-1)
        elif rho.ndim != 3:
            raise ValueError(
                f"Expected density shape (T, nx, ny) or (T, nx, ny, 1), "
                f"got {rho.shape} in {density_path}"
            )

        # Apply time slice if provided
        if t_slice is not None:
            rho = rho[t_slice]

        T, nx, ny = rho.shape

        # Extract run information
        run_dir = density_path.parent
        run_id = run_dir.name

        densities.append(rho)
        metadata.append({
            "run_path": run_dir,
            "run_id": run_id,
            "T": T,
            "grid_shape": (nx, ny),
            "run_index": idx,
        })

    return densities, metadata


def build_snapshot_matrix(
    densities: List[np.ndarray],
    center: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Build a global snapshot matrix from multiple density runs.

    Flattens and concatenates density fields from multiple runs into a single
    snapshot matrix X of shape (d, M), where:
    - d = nx * ny (spatial degrees of freedom)
    - M = total number of time snapshots across all runs

    Parameters
    ----------
    densities : list of np.ndarray
        Each array has shape (T_i, nx, ny).
    center : bool, optional
        If True, subtract the global spatial mean across all snapshots.
        Default is True.

    Returns
    -------
    X : np.ndarray, shape (d, M)
        Snapshot matrix. If center=True, mean has been removed.
    mean : np.ndarray, shape (d,)
        Spatial mean across all snapshots. Zero if center=False.
    grid_shape : tuple of (int, int)
        Grid dimensions (nx, ny) inferred from first density array.

    Examples
    --------
    >>> densities = [np.random.rand(100, 32, 32) for _ in range(5)]
    >>> X, mean, grid_shape = build_snapshot_matrix(densities)
    >>> X.shape
    (1024, 500)
    >>> mean.shape
    (1024,)
    >>> grid_shape
    (32, 32)

    Notes
    -----
    All density arrays must have the same spatial grid shape (nx, ny).
    They can have different numbers of time steps T_i.
    """
    if not densities:
        raise ValueError("densities list is empty")

    # Infer grid shape from first run
    T0, nx, ny = densities[0].shape
    grid_shape = (nx, ny)
    d = nx * ny

    # Validate that all runs have the same grid shape
    for i, rho in enumerate(densities):
        if rho.shape[1:] != grid_shape:
            raise ValueError(
                f"Run {i} has grid shape {rho.shape[1:]}, "
                f"expected {grid_shape}"
            )

    # Flatten each run and concatenate
    # Each rho has shape (T_i, nx, ny), flatten to (T_i, d)
    flattened = [rho.reshape(rho.shape[0], -1) for rho in densities]

    # Concatenate along time axis: (sum(T_i), d)
    X_time = np.vstack(flattened)  # shape (M, d)

    # Transpose to (d, M) for standard snapshot matrix format
    X = X_time.T  # shape (d, M)

    # Compute and remove mean if requested
    if center:
        mean = X.mean(axis=1)
        X = X - mean[:, np.newaxis]
    else:
        mean = np.zeros(d)

    return X, mean, grid_shape


def compute_pod(
    X: np.ndarray,
    mean: np.ndarray,
    grid_shape: Tuple[int, int],
    energy_tol: float = 0.995,
    r_max: int | None = None,
) -> PODBasis:
    """Compute a global POD basis from a snapshot matrix.

    Performs singular value decomposition (SVD) on the centered snapshot
    matrix and selects modes based on cumulative energy threshold.

    Parameters
    ----------
    X : np.ndarray, shape (d, M)
        Snapshot matrix with mean already removed (centered).
    mean : np.ndarray, shape (d,)
        Spatial mean that was subtracted from X.
    grid_shape : tuple of (int, int)
        Grid dimensions (nx, ny).
    energy_tol : float, optional
        Keep the minimal number of modes such that cumulative energy
        fraction >= energy_tol. Default is 0.995 (99.5%).
    r_max : int or None, optional
        Optional hard cap on the number of modes. If specified, the
        number of modes is min(r_energy, r_max).

    Returns
    -------
    PODBasis
        POD basis with modes, singular values, mean, grid shape, and
        cumulative energy fractions.

    Examples
    --------
    >>> X = np.random.randn(1024, 500)
    >>> mean = np.zeros(1024)
    >>> pod = compute_pod(X, mean, (32, 32), energy_tol=0.99)
    >>> pod.n_modes
    42
    >>> pod.energy[-1] >= 0.99
    True

    Notes
    -----
    Energy is defined as the squared singular values:
    E_k = sum(S[0:k]**2) / sum(S**2)

    The POD modes (columns of U) are the left singular vectors of X,
    which are the eigenvectors of the spatial covariance matrix X X^T.
    """
    d, M = X.shape

    # Perform SVD: X = U Σ V^T
    # U: (d, k), S: (k,), Vt: (k, M) where k = min(d, M)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Compute cumulative energy fractions
    energy_vals = S ** 2
    total_energy = energy_vals.sum()
    cum_energy = np.cumsum(energy_vals) / total_energy

    # Determine number of modes to keep based on energy threshold
    r_energy = int(np.searchsorted(cum_energy, energy_tol) + 1)

    # Apply hard cap if provided
    if r_max is not None:
        r = min(r_energy, r_max)
    else:
        r = r_energy

    # Ensure we keep at least 1 mode
    r = max(1, min(r, len(S)))

    # Extract selected modes
    modes = U[:, :r]
    singular_values = S[:r]
    energy = cum_energy[:r]

    return PODBasis(
        modes=modes,
        singular_values=singular_values,
        mean=mean,
        grid_shape=grid_shape,
        energy=energy,
    )


def project_onto_pod(
    rho: np.ndarray,
    pod: PODBasis,
    center: bool = True,
) -> np.ndarray:
    """Project a density movie onto a POD basis.

    Maps density fields from spatial representation (T, nx, ny) to
    latent POD coordinates (T, r).

    Parameters
    ----------
    rho : np.ndarray, shape (T, nx, ny)
        Density field time series.
    pod : PODBasis
        POD basis object.
    center : bool, optional
        If True, subtract pod.mean before projection. Default is True.

    Returns
    -------
    y : np.ndarray, shape (T, r)
        Latent trajectory in POD coordinates, where r is the number of modes.

    Examples
    --------
    >>> rho = np.random.rand(100, 32, 32)
    >>> pod = PODBasis(...)  # from compute_pod
    >>> y = project_onto_pod(rho, pod)
    >>> y.shape
    (100, 50)

    Notes
    -----
    The projection is: y_t = modes.T @ (rho_t - mean)
    where rho_t is flattened to shape (d,).
    """
    T, nx, ny = rho.shape

    # Validate grid shape
    if (nx, ny) != pod.grid_shape:
        raise ValueError(
            f"Density grid shape {(nx, ny)} does not match "
            f"POD grid shape {pod.grid_shape}"
        )

    # Flatten: (T, nx, ny) -> (T, d)
    rho_flat = rho.reshape(T, -1)

    # Center if requested
    if center:
        rho_centered = rho_flat - pod.mean[np.newaxis, :]
    else:
        rho_centered = rho_flat

    # Project: y = X @ modes, where X is (T, d) and modes is (d, r)
    # Result is (T, r)
    y = rho_centered @ pod.modes

    return y


def reconstruct_from_pod(
    y: np.ndarray,
    pod: PODBasis,
    add_mean: bool = True,
) -> np.ndarray:
    """Reconstruct density fields from latent POD coefficients.

    Maps latent coordinates (T, r) back to spatial density fields (T, nx, ny).

    Parameters
    ----------
    y : np.ndarray, shape (T, r)
        Latent trajectory in POD coordinates.
    pod : PODBasis
        POD basis object.
    add_mean : bool, optional
        If True, add pod.mean back into the reconstruction. Default is True.

    Returns
    -------
    rho_rec : np.ndarray, shape (T, nx, ny)
        Reconstructed density movie.

    Examples
    --------
    >>> y = np.random.rand(100, 50)
    >>> pod = PODBasis(...)  # from compute_pod
    >>> rho_rec = reconstruct_from_pod(y, pod)
    >>> rho_rec.shape
    (100, 32, 32)

    Notes
    -----
    The reconstruction is: rho_t = modes @ y_t + mean
    The result is reshaped from (d,) to (nx, ny).
    """
    T, r = y.shape

    # Validate mode count
    if r != pod.n_modes:
        raise ValueError(
            f"Latent dimension {r} does not match "
            f"POD mode count {pod.n_modes}"
        )

    # Reconstruct: X = y @ modes.T, where y is (T, r) and modes.T is (r, d)
    # Result is (T, d)
    rho_flat = y @ pod.modes.T

    # Add mean if requested
    if add_mean:
        rho_flat = rho_flat + pod.mean[np.newaxis, :]

    # Reshape to (T, nx, ny)
    nx, ny = pod.grid_shape
    rho_rec = rho_flat.reshape(T, nx, ny)

    return rho_rec


def build_latent_dataset(
    densities: List[np.ndarray],
    pod: PODBasis,
    train_frac: float = 0.9,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Build train/test latent datasets from ensemble projections.

    Projects each density run onto the POD basis and splits into train/test
    segments in time. Useful for training a single global MVAR model across
    multiple simulation runs.

    Parameters
    ----------
    densities : list of np.ndarray
        Each array has shape (T_i, nx, ny).
    pod : PODBasis
        POD basis for projection.
    train_frac : float, optional
        Fraction of each run to use for training (in time). Default is 0.9.

    Returns
    -------
    data : dict
        Dictionary with keys 'train' and 'test', each mapping to:
        - 'y' : np.ndarray, shape (T_total, r)
            Latent coordinates concatenated across runs.
        - 'run_ids' : np.ndarray[int], shape (T_total,)
            Run index for each time point.
        - 't_idx' : np.ndarray[int], shape (T_total,)
            Original time index within each run.

    Examples
    --------
    >>> densities = [np.random.rand(100, 32, 32) for _ in range(5)]
    >>> pod = PODBasis(...)
    >>> data = build_latent_dataset(densities, pod, train_frac=0.8)
    >>> data['train']['y'].shape
    (400, 50)
    >>> data['test']['y'].shape
    (100, 50)

    Notes
    -----
    For each run k with T_k time steps:
    - Train: time indices 0 to T_train_k = int(train_frac * T_k)
    - Test: time indices T_train_k to T_k

    This enables training a global MVAR on combined data from all runs
    while reserving test segments for validation.
    """
    train_ys = []
    train_run_ids = []
    train_t_idxs = []

    test_ys = []
    test_run_ids = []
    test_t_idxs = []

    for run_idx, rho in enumerate(densities):
        T = rho.shape[0]

        # Project to latent space
        y = project_onto_pod(rho, pod, center=True)

        # Determine split point
        T_train = int(train_frac * T)
        T_train = max(1, min(T_train, T - 1))  # Ensure both splits are non-empty

        # Split into train and test
        y_train = y[:T_train]
        y_test = y[T_train:]

        # Track metadata
        train_ys.append(y_train)
        train_run_ids.append(np.full(T_train, run_idx, dtype=int))
        train_t_idxs.append(np.arange(T_train, dtype=int))

        test_ys.append(y_test)
        test_run_ids.append(np.full(T - T_train, run_idx, dtype=int))
        test_t_idxs.append(np.arange(T_train, T, dtype=int))

    # Concatenate across runs
    data = {
        "train": {
            "y": np.vstack(train_ys),
            "run_ids": np.concatenate(train_run_ids),
            "t_idx": np.concatenate(train_t_idxs),
        },
        "test": {
            "y": np.vstack(test_ys),
            "run_ids": np.concatenate(test_run_ids),
            "t_idx": np.concatenate(test_t_idxs),
        },
    }

    return data
