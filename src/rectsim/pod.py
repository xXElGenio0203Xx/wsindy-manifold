"""
POD/SVD Restriction-Lifting for Density Fields (Alvarez et al., 2025)

Implements proper orthogonal decomposition (POD) with temporal centering for
mass-normalized density snapshots on a fixed grid. Provides restriction (R)
and lifting (L) operators for low-dimensional latent representation.

Key Features:
- Temporal centering: X_bar = X - x_bar·1^T
- Energy-based dimension selection: choose d via cumulative energy ≥ τ
- Mass preservation: ∫ρ = 1 maintained through restriction/lifting
- Orthonormal basis: U_d^T @ U_d = I_d
- Both standard and temporal covariance SVD routes

Reference: Alvarez et al. (2025), Algorithm for crowd density POD
"""

import numpy as np
from typing import Union, Tuple, Optional, Dict, List
import warnings


class PODProjector:
    """
    POD-based dimensionality reduction for mass-normalized density fields.
    
    Algorithm:
    1. Assemble data matrix X (n_c × n_t) from flattened density snapshots
    2. Compute temporal mean: x_bar = (1/n_t) X 1_t
    3. Center: X_bar = X - x_bar·1^T
    4. SVD: X_bar = U Σ V^T (economy mode)
    5. Choose dimension d via energy threshold τ
    6. Store U_d (first d left singular vectors) and x_bar
    7. Restriction: y = U_d^T (x - x_bar)
    8. Lifting: x = U_d y + x_bar
    
    Parameters
    ----------
    energy_threshold : float, default=0.99
        Target cumulative energy for dimension selection: d = min{k: E(k) ≥ τ}
    fixed_d : int or None, default=None
        If provided, override energy threshold and use this dimension
    tol_mass : float, default=1e-12
        Tolerance for mass preservation checks
    use_weighted_mass : bool, default=False
        If True, use δx·δy-weighted mass checks; if False, use unweighted sum
    delta_x : float or None, default=None
        Grid spacing in x (optional, for mass checks)
    delta_y : float or None, default=None
        Grid spacing in y (optional, for mass checks)
    randomized : bool, default=False
        If True, use randomized SVD for large matrices (faster but approximate)
    random_state : int, default=0
        Random seed for reproducibility (used if randomized=True)
    temporal_cov : bool, default=False
        If True, use temporal covariance route (efficient when n_c >> n_t)
    """
    
    def __init__(
        self,
        energy_threshold: float = 0.99,
        fixed_d: Optional[int] = None,
        tol_mass: float = 1e-12,
        use_weighted_mass: bool = False,
        delta_x: Optional[float] = None,
        delta_y: Optional[float] = None,
        randomized: bool = False,
        random_state: int = 0,
        temporal_cov: bool = False,
    ):
        self.energy_threshold = energy_threshold
        self.fixed_d = fixed_d
        self.tol_mass = tol_mass
        self.use_weighted_mass = use_weighted_mass
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.randomized = randomized
        self.random_state = random_state
        self.temporal_cov = temporal_cov
        
        # Attributes set during fit
        self.n_x: Optional[int] = None
        self.n_y: Optional[int] = None
        self.n_c: Optional[int] = None
        self.n_t: Optional[int] = None
        self.x_bar: Optional[np.ndarray] = None  # (n_c,)
        self.U_d: Optional[np.ndarray] = None    # (n_c, d)
        self.s: Optional[np.ndarray] = None      # (r,) singular values
        self.d: Optional[int] = None             # chosen latent dimension
        self.energy_curve: Optional[np.ndarray] = None  # cumulative energy
        self._fitted: bool = False
    
    def fit(
        self,
        rho_list: Union[np.ndarray, List[np.ndarray]],
        delta_x: Optional[float] = None,
        delta_y: Optional[float] = None,
    ) -> "PODProjector":
        """
        Fit POD projector to mass-normalized density snapshots.
        
        Parameters
        ----------
        rho_list : array-like of shape (n_t, n_x, n_y) or list of (n_x, n_y) arrays
            Mass-normalized density snapshots: sum(rho_k) * δx * δy = 1 for all k
        delta_x : float, optional
            Grid spacing in x (overrides constructor value)
        delta_y : float, optional
            Grid spacing in y (overrides constructor value)
        
        Returns
        -------
        self : PODProjector
            Fitted projector with U_d, x_bar, s, d, and energy_curve set
        """
        # Handle delta values
        if delta_x is not None:
            self.delta_x = delta_x
        if delta_y is not None:
            self.delta_y = delta_y
        
        # Convert to array and extract shapes
        if isinstance(rho_list, list):
            rho_array = np.array(rho_list, dtype=np.float64)
        else:
            rho_array = np.asarray(rho_list, dtype=np.float64)
        
        if rho_array.ndim != 3:
            raise ValueError(f"Expected 3D array (n_t, n_x, n_y), got shape {rho_array.shape}")
        
        self.n_t, self.n_x, self.n_y = rho_array.shape
        self.n_c = self.n_x * self.n_y
        
        # Step 1: Assemble data matrix X (n_c × n_t)
        X = rho_array.reshape(self.n_t, self.n_c).T  # (n_c, n_t)
        
        # Assert mass normalization for each snapshot
        if self.delta_x is not None and self.delta_y is not None:
            area = self.delta_x * self.delta_y
            masses = np.sum(X, axis=0) * area
            mass_errors = np.abs(masses - 1.0)
            max_mass_error = np.max(mass_errors)
            
            if max_mass_error > self.tol_mass:
                warnings.warn(
                    f"Input snapshots not mass-normalized: max error = {max_mass_error:.2e} > {self.tol_mass:.2e}"
                )
        
        # Step 2: Compute temporal mean and center
        one_t = np.ones((self.n_t, 1))
        self.x_bar = (X @ one_t / self.n_t).ravel()  # (n_c,)
        X_bar = X - self.x_bar[:, None]  # (n_c, n_t)
        
        # Step 3: SVD (choose route based on temporal_cov flag)
        if self.temporal_cov and self.n_c > 10 * self.n_t:
            # Temporal covariance route (efficient when n_c >> n_t)
            self._fit_temporal_cov(X_bar)
        else:
            # Direct SVD route
            self._fit_direct_svd(X_bar)
        
        # Step 4: Choose latent dimension d
        total_energy = np.sum(self.s**2)
        self.energy_curve = np.cumsum(self.s**2) / total_energy
        
        if self.fixed_d is not None:
            self.d = min(self.fixed_d, len(self.s))
        else:
            # Find smallest d where E(d) ≥ τ
            self.d = int(np.searchsorted(self.energy_curve, self.energy_threshold) + 1)
            self.d = min(self.d, len(self.s))
        
        # Step 5: Extract U_d (first d columns)
        self.U_d = self.U[:, :self.d]
        
        self._fitted = True
        return self
    
    def _fit_direct_svd(self, X_bar: np.ndarray):
        """Standard SVD route: X_bar = U Σ V^T"""
        if self.randomized:
            from sklearn.utils.extmath import randomized_svd
            self.U, self.s, Vt = randomized_svd(
                X_bar,
                n_components=min(X_bar.shape) - 1,
                random_state=self.random_state,
            )
        else:
            self.U, self.s, Vt = np.linalg.svd(X_bar, full_matrices=False)
    
    def _fit_temporal_cov(self, X_bar: np.ndarray):
        """
        Temporal covariance route (efficient when n_c >> n_t):
        Ct = X_bar^T @ X_bar = V Λ V^T
        Then U = X_bar @ V @ Λ^{-1/2}
        """
        Ct = X_bar.T @ X_bar  # (n_t, n_t)
        
        # Eigendecompose temporal covariance
        eigvals, V = np.linalg.eigh(Ct)
        
        # Sort in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        V = V[:, idx]
        
        # Keep only positive eigenvalues
        tol = max(X_bar.shape) * np.finfo(float).eps * eigvals[0]
        mask = eigvals > tol
        eigvals = eigvals[mask]
        V = V[:, mask]
        
        # Singular values and left singular vectors
        self.s = np.sqrt(eigvals)
        self.U = X_bar @ V @ np.diag(1.0 / self.s)  # (n_c, r)
    
    def transform(self, rho_or_X: np.ndarray) -> np.ndarray:
        """
        Restriction: R(X) = U_d^T (X - x_bar·1^T)
        
        Parameters
        ----------
        rho_or_X : ndarray
            Either:
            - Batch of density snapshots: (n_t, n_x, n_y)
            - Data matrix: (n_c, n_t)
        
        Returns
        -------
        Y_d : ndarray of shape (d, n_t)
            Latent coordinates
        """
        self._check_fitted()
        
        # Convert to data matrix if needed
        if rho_or_X.ndim == 3:
            n_t, n_x, n_y = rho_or_X.shape
            if n_x != self.n_x or n_y != self.n_y:
                raise ValueError(
                    f"Shape mismatch: expected (*, {self.n_x}, {self.n_y}), got {rho_or_X.shape}"
                )
            X = rho_or_X.reshape(n_t, self.n_c).T  # (n_c, n_t)
        elif rho_or_X.ndim == 2:
            if rho_or_X.shape[0] != self.n_c:
                raise ValueError(
                    f"Shape mismatch: expected ({self.n_c}, *), got {rho_or_X.shape}"
                )
            X = rho_or_X
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {rho_or_X.shape}")
        
        # Center and project
        X_bar = X - self.x_bar[:, None]
        Y_d = self.U_d.T @ X_bar  # (d, n_t)
        
        return Y_d
    
    def inverse_transform(self, Y_d: np.ndarray) -> np.ndarray:
        """
        Lifting: L(Y_d) = U_d Y_d + x_bar·1^T
        
        Parameters
        ----------
        Y_d : ndarray of shape (d, n_t)
            Latent coordinates
        
        Returns
        -------
        X_hat : ndarray of shape (n_c, n_t)
            Reconstructed data matrix (flattened density snapshots)
        """
        self._check_fitted()
        
        if Y_d.shape[0] != self.d:
            raise ValueError(f"Expected first dimension {self.d}, got {Y_d.shape[0]}")
        
        X_hat = self.U_d @ Y_d + self.x_bar[:, None]  # (n_c, n_t)
        
        return X_hat
    
    def restrict_one(self, rho_single: np.ndarray) -> np.ndarray:
        """
        Restriction for a single snapshot: R(x) = U_d^T (x - x_bar)
        
        Parameters
        ----------
        rho_single : ndarray of shape (n_x, n_y)
            Single density snapshot
        
        Returns
        -------
        y : ndarray of shape (d,)
            Latent coordinates
        """
        self._check_fitted()
        
        if rho_single.shape != (self.n_x, self.n_y):
            raise ValueError(
                f"Shape mismatch: expected ({self.n_x}, {self.n_y}), got {rho_single.shape}"
            )
        
        x = rho_single.ravel().astype(np.float64)
        y = self.U_d.T @ (x - self.x_bar)
        
        return y
    
    def lift_one(self, y: np.ndarray) -> np.ndarray:
        """
        Lifting for a single latent vector: L(y) = U_d y + x_bar
        
        Parameters
        ----------
        y : ndarray of shape (d,)
            Latent coordinates
        
        Returns
        -------
        x : ndarray of shape (n_c,)
            Reconstructed flattened density snapshot
        """
        self._check_fitted()
        
        if y.shape[0] != self.d:
            raise ValueError(f"Expected dimension {self.d}, got {y.shape[0]}")
        
        x = self.U_d @ y + self.x_bar
        
        return x
    
    def mass_check(self, X_hat: np.ndarray) -> float:
        """
        Compute maximum mass error across columns.
        
        Parameters
        ----------
        X_hat : ndarray of shape (n_c, n_t)
            Reconstructed data matrix
        
        Returns
        -------
        max_mass_error : float
            Maximum absolute mass error: max_k |∫ρ_k - 1|
        """
        self._check_fitted()
        
        if self.use_weighted_mass and self.delta_x is not None and self.delta_y is not None:
            # Weighted: mass = δx·δy·sum(ρ)
            weight = self.delta_x * self.delta_y
            masses = weight * np.sum(X_hat, axis=0)
        else:
            # Unweighted: mass = sum(ρ)
            masses = np.sum(X_hat, axis=0)
        
        mass_error = np.max(np.abs(masses - 1.0))
        
        return mass_error
    
    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Compute relative L2 reconstruction error per column.
        
        Parameters
        ----------
        X : ndarray of shape (n_c, n_t)
            Original data matrix
        
        Returns
        -------
        e2_rec : ndarray of shape (n_t,)
            Relative reconstruction errors: ||x_k - x̂_k||_2 / ||x_k||_2
        """
        self._check_fitted()
        
        if X.shape[0] != self.n_c:
            raise ValueError(f"Expected first dimension {self.n_c}, got {X.shape[0]}")
        
        # Reconstruct
        X_bar = X - self.x_bar[:, None]
        Y_d = self.U_d.T @ X_bar
        X_hat = self.U_d @ Y_d + self.x_bar[:, None]
        
        # Compute relative errors
        diff = X - X_hat
        norms_diff = np.linalg.norm(diff, axis=0)
        norms_orig = np.linalg.norm(X, axis=0)
        
        # Avoid division by zero
        mask = norms_orig > 0
        e2_rec = np.zeros(X.shape[1])
        e2_rec[mask] = norms_diff[mask] / norms_orig[mask]
        
        return e2_rec
    
    def orthonormality_check(self) -> float:
        """
        Check orthonormality: ||U_d^T @ U_d - I_d||_max
        
        Returns
        -------
        max_error : float
            Maximum absolute deviation from identity
        """
        self._check_fitted()
        
        I_d = np.eye(self.d)
        UtU = self.U_d.T @ self.U_d
        
        max_error = np.max(np.abs(UtU - I_d))
        
        return max_error
    
    def get_metadata(self) -> Dict:
        """
        Get metadata about the fitted POD projector.
        
        Returns
        -------
        metadata : dict
            Dictionary with keys: n_x, n_y, n_c, n_t, d, energy_threshold,
            energy_achieved, singular_values, etc.
        """
        self._check_fitted()
        
        metadata = {
            "n_x": self.n_x,
            "n_y": self.n_y,
            "n_c": self.n_c,
            "n_t": self.n_t,
            "d": self.d,
            "energy_threshold": self.energy_threshold,
            "energy_achieved": self.energy_curve[self.d - 1] if self.d > 0 else 0.0,
            "singular_values": self.s.tolist(),
            "delta_x": self.delta_x,
            "delta_y": self.delta_y,
            "use_weighted_mass": self.use_weighted_mass,
            "tol_mass": self.tol_mass,
        }
        
        return metadata
    
    def save(self, filepath: str):
        """
        Save POD projector to disk.
        
        Parameters
        ----------
        filepath : str
            Path to save file (will use .npz format)
        """
        self._check_fitted()
        
        np.savez(
            filepath,
            n_x=self.n_x,
            n_y=self.n_y,
            n_c=self.n_c,
            n_t=self.n_t,
            d=self.d,
            x_bar=self.x_bar,
            U_d=self.U_d,
            s=self.s,
            energy_curve=self.energy_curve,
            energy_threshold=self.energy_threshold,
            delta_x=self.delta_x,
            delta_y=self.delta_y,
            use_weighted_mass=self.use_weighted_mass,
            tol_mass=self.tol_mass,
        )
    
    @classmethod
    def load(cls, filepath: str) -> "PODProjector":
        """
        Load POD projector from disk.
        
        Parameters
        ----------
        filepath : str
            Path to saved file (.npz format)
        
        Returns
        -------
        projector : PODProjector
            Loaded projector
        """
        data = np.load(filepath, allow_pickle=False)
        
        projector = cls(
            energy_threshold=float(data["energy_threshold"]),
            tol_mass=float(data["tol_mass"]),
            use_weighted_mass=bool(data["use_weighted_mass"]),
            delta_x=float(data["delta_x"]) if data["delta_x"] is not None else None,
            delta_y=float(data["delta_y"]) if data["delta_y"] is not None else None,
        )
        
        projector.n_x = int(data["n_x"])
        projector.n_y = int(data["n_y"])
        projector.n_c = int(data["n_c"])
        projector.n_t = int(data["n_t"])
        projector.d = int(data["d"])
        projector.x_bar = data["x_bar"]
        projector.U_d = data["U_d"]
        projector.s = data["s"]
        projector.energy_curve = data["energy_curve"]
        projector._fitted = True
        
        return projector
    
    def _check_fitted(self):
        """Raise error if projector not fitted"""
        if not self._fitted:
            raise RuntimeError("PODProjector must be fitted before use. Call fit() first.")
