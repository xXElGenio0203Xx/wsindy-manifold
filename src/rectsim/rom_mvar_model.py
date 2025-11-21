"""ROM/MVAR model class for encoding, forecasting, and decoding density fields.

This module provides a clean API for loading and using trained POD+MVAR models
to forecast density evolution from latent representations.

Author: Maria
Date: November 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import numpy as np


@dataclass
class PODMVARModel:
    """POD+MVAR reduced-order model for density forecasting.
    
    Attributes
    ----------
    pod_modes : np.ndarray
        POD basis vectors, shape (n_spatial, latent_dim).
        n_spatial = Nx * Ny for 2D density fields.
    mean_mode : np.ndarray
        Mean density field (flattened), shape (n_spatial,).
    A0 : np.ndarray
        MVAR constant term, shape (latent_dim,).
    A_coeffs : np.ndarray
        MVAR coefficient matrices, shape (mvar_order, latent_dim, latent_dim).
        A_coeffs[k] is the coefficient matrix for lag k+1.
    latent_dim : int
        Number of POD modes retained.
    mvar_order : int
        Number of lags in MVAR model.
    grid_shape : tuple[int, int]
        Spatial grid dimensions (Ny, Nx).
    dt : float
        Time step used during training.
    meta : dict
        Additional metadata from training.
    """
    
    pod_modes: np.ndarray
    mean_mode: np.ndarray
    A0: np.ndarray
    A_coeffs: np.ndarray
    latent_dim: int
    mvar_order: int
    grid_shape: tuple[int, int]
    dt: float
    meta: Dict[str, Any]
    
    @classmethod
    def load(cls, rom_dir: Path) -> "PODMVARModel":
        """Load POD basis and MVAR coefficients from directory.
        
        Expected files in rom_dir:
        - model/pod_basis.npz: Contains 'pod_modes', 'mean_mode', 'nx', 'ny', 'latent_dim'
          (legacy format from save_pod_model in rom_mvar.py)
        - model/mvar_params.npz: Contains 'A0', 'A_coeffs', 'mvar_order'
        - model/train_summary.json: Contains 'dt' and other metadata
        
        Parameters
        ----------
        rom_dir : Path
            Directory containing trained ROM model files.
            
        Returns
        -------
        PODMVARModel
            Loaded model ready for encoding/forecasting/decoding.
            
        Raises
        ------
        FileNotFoundError
            If required model files are missing.
        ValueError
            If model files have inconsistent shapes.
        """
        import json
        
        rom_dir = Path(rom_dir)
        model_dir = rom_dir / "model"
        
        # Load POD basis
        pod_path = model_dir / "pod_basis.npz"
        if not pod_path.exists():
            raise FileNotFoundError(f"POD basis not found at {pod_path}")
        
        pod_data = np.load(pod_path)
        # Legacy format uses 'pod_modes' and 'mean_mode'
        pod_modes = pod_data["pod_modes"]  # (latent_dim, n_spatial)
        mean_mode = pod_data["mean_mode"]  # (n_spatial,)
        Nx = int(pod_data["nx"])
        Ny = int(pod_data["ny"])
        latent_dim = int(pod_data["latent_dim"])
        
        # Load MVAR coefficients
        mvar_path = model_dir / "mvar_params.npz"
        if not mvar_path.exists():
            raise FileNotFoundError(f"MVAR params not found at {mvar_path}")
        
        mvar_data = np.load(mvar_path)
        A0 = mvar_data["A0"]  # (latent_dim,)
        A_coeffs = mvar_data["A_coeffs"]  # (order, latent_dim, latent_dim) from save_mvar_model
        mvar_order = int(mvar_data["order"])  # key is 'order' not 'mvar_order'
        
        # Load metadata
        meta_path = model_dir / "train_summary.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            dt = float(meta.get("dt", 0.01))
        else:
            meta = {}
            dt = 0.01
            print(f"Warning: {meta_path} not found, using default dt={dt}")
        
        # Validate shapes
        n_spatial = Nx * Ny
        # pod_modes shape is (latent_dim, n_spatial) from save_pod_model
        if pod_modes.shape != (latent_dim, n_spatial):
            raise ValueError(
                f"POD modes shape {pod_modes.shape} inconsistent with "
                f"expected ({latent_dim}, {n_spatial})"
            )
        if mean_mode.shape != (n_spatial,):
            raise ValueError(f"Mean mode shape {mean_mode.shape} != ({n_spatial},)")
        if A0.shape != (latent_dim,):
            raise ValueError(f"A0 shape {A0.shape} != ({latent_dim},)")
        if A_coeffs.shape != (mvar_order, latent_dim, latent_dim):
            raise ValueError(
                f"A_coeffs shape {A_coeffs.shape} != "
                f"({mvar_order}, {latent_dim}, {latent_dim})"
            )
        
        return cls(
            pod_modes=pod_modes,
            mean_mode=mean_mode,
            A0=A0,
            A_coeffs=A_coeffs,
            latent_dim=latent_dim,
            mvar_order=mvar_order,
            grid_shape=(Ny, Nx),
            dt=dt,
            meta=meta,
        )
    
    def encode(self, density_movie: np.ndarray) -> np.ndarray:
        """Project density movie to latent space using POD.
        
        Parameters
        ----------
        density_movie : np.ndarray
            Density fields, shape (T, Ny, Nx).
            
        Returns
        -------
        latent_movie : np.ndarray
            Latent coefficients, shape (T, latent_dim).
            
        Notes
        -----
        Projects (density - mean) onto POD modes:
        latent[t] = U^T @ (density[t].flatten() - mean)
        """
        T = density_movie.shape[0]
        Ny, Nx = self.grid_shape
        
        if density_movie.shape[1:] != (Ny, Nx):
            raise ValueError(
                f"Density movie grid shape {density_movie.shape[1:]} "
                f"does not match model grid {(Ny, Nx)}"
            )
        
        # Flatten spatial dimensions
        density_flat = density_movie.reshape(T, -1)  # (T, n_spatial)
        
        # Center and project
        # pod_modes shape is (latent_dim, n_spatial), need transpose
        centered = density_flat - self.mean_mode[None, :]  # (T, n_spatial)
        latent = centered @ self.pod_modes.T  # (T, n_spatial) @ (n_spatial, latent_dim) → (T, latent_dim)
        
        return latent
    
    def forecast(self, y_init: np.ndarray, n_steps: int) -> np.ndarray:
        """Forecast latent coefficients forward using MVAR model.
        
        The MVAR model has the form:
            y_t = A0 + Σ_{k=1}^p A_k y_{t-k} + noise
        
        Parameters
        ----------
        y_init : np.ndarray
            Initial latent states for seeding the forecast, shape (p, latent_dim),
            ordered as [y_{t-p+1}, ..., y_{t-1}, y_t] (oldest to newest).
        n_steps : int
            Number of time steps to forecast forward.
            
        Returns
        -------
        latent_preds : np.ndarray
            Forecasted latent coefficients, shape (n_steps, latent_dim).
            
        Notes
        -----
        This implements autoregressive forecasting where each predicted step
        becomes part of the history for subsequent predictions.
        """
        if y_init.shape != (self.mvar_order, self.latent_dim):
            raise ValueError(
                f"y_init shape {y_init.shape} != "
                f"expected ({self.mvar_order}, {self.latent_dim})"
            )
        
        # Initialize history buffer with seed states
        history = y_init.copy()  # (p, d)
        predictions = np.zeros((n_steps, self.latent_dim))
        
        for t in range(n_steps):
            # Compute next step: y_t = A0 + Σ A_k y_{t-k}
            y_next = self.A0.copy()
            
            # Add contribution from each lag (history[-1] is most recent)
            for k in range(self.mvar_order):
                lag_idx = -(k + 1)  # -1, -2, ..., -p
                y_next += self.A_coeffs[k] @ history[lag_idx]
            
            predictions[t] = y_next
            
            # Update history: shift and append new prediction
            history = np.vstack([history[1:], y_next[None, :]])
        
        return predictions
    
    def decode(self, latent_movie: np.ndarray) -> np.ndarray:
        """Reconstruct density fields from latent coefficients.
        
        Parameters
        ----------
        latent_movie : np.ndarray
            Latent coefficients, shape (T, latent_dim).
            
        Returns
        -------
        density_movie : np.ndarray
            Reconstructed density fields, shape (T, Ny, Nx).
            
        Notes
        -----
        Reconstructs density as:
        density[t] = mean + U @ latent[t]
        """
        if latent_movie.shape[1] != self.latent_dim:
            raise ValueError(
                f"Latent dimension {latent_movie.shape[1]} != "
                f"expected {self.latent_dim}"
            )
        
        T = latent_movie.shape[0]
        Ny, Nx = self.grid_shape
        
        # Reconstruct: density_flat = mean + pod_modes^T @ latent^T
        # pod_modes is (latent_dim, n_spatial), so pod_modes.T is (n_spatial, latent_dim)
        # latent_movie is (T, latent_dim)
        # Result: (T, latent_dim) @ (latent_dim, n_spatial) = (T, n_spatial)
        density_flat = self.mean_mode[None, :] + latent_movie @ self.pod_modes  # (T, n_spatial)
        
        # Reshape to 2D grid
        density_movie = density_flat.reshape(T, Ny, Nx)
        
        return density_movie
    
    def predict_from_density(
        self, 
        density_init: np.ndarray, 
        n_steps: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """End-to-end prediction: encode initial states, forecast, and decode.
        
        Parameters
        ----------
        density_init : np.ndarray
            Initial density fields for seeding, shape (p, Ny, Nx),
            where p = mvar_order.
        n_steps : int
            Number of time steps to forecast.
            
        Returns
        -------
        density_pred : np.ndarray
            Predicted density fields, shape (n_steps, Ny, Nx).
        latent_pred : np.ndarray
            Predicted latent coefficients, shape (n_steps, latent_dim).
        """
        # Encode initial densities
        latent_init = self.encode(density_init)  # (p, latent_dim)
        
        # Forecast in latent space
        latent_pred = self.forecast(latent_init, n_steps)
        
        # Decode predictions
        density_pred = self.decode(latent_pred)
        
        return density_pred, latent_pred
