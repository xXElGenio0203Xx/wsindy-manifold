"""
Linear multivariate autoregressive (VAR) models for latent dynamics.

Enhanced implementation with:
- Automatic lag selection via AIC/BIC
- Forecast horizon testing
- Diagnostic plotting
- Stochastic noise modeling
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import matplotlib.pyplot as plt

Array = np.ndarray


def build_lagged(Y: Array, w: int) -> Tuple[Array, Array]:
    """Construct lagged design and target matrices for VAR fitting."""

    if Y.ndim != 2:
        raise ValueError("Y must have shape (T, d)")
    if w <= 0:
        raise ValueError("w must be a positive integer")

    T, d = Y.shape
    if T <= w:
        raise ValueError("Need at least w+1 samples to build a VAR model")

    targets = Y[w:]
    rows = T - w
    ones = np.ones((rows, 1), dtype=float)
    lagged_blocks = [Y[w - lag : T - lag] for lag in range(1, w + 1)]
    design = np.hstack([block for block in lagged_blocks])
    Z = np.hstack([ones, design])
    return Z, targets


def fit_mvar(Y: Array, w: int, ridge_lambda: float = 0.0) -> Dict[str, Array]:
    """Fit a multivariate autoregression of order ``w`` to latent data."""

    Z, targets = build_lagged(Y, w=w)
    ZTZ = Z.T @ Z
    ZTy = Z.T @ targets

    if ridge_lambda < 0:
        raise ValueError("ridge_lambda must be non-negative")

    if ridge_lambda > 0:
        reg = ridge_lambda * np.eye(ZTZ.shape[0])
        reg[0, 0] = 0.0  # Do not regularise the intercept term.
        ZTZ = ZTZ + reg

    try:
        theta = np.linalg.solve(ZTZ, ZTy)
    except np.linalg.LinAlgError:
        theta = np.linalg.pinv(ZTZ) @ ZTy

    d = targets.shape[1]
    A0 = theta[0]
    coeffs = theta[1:]
    A = np.empty((w, d, d), dtype=float)
    for lag in range(w):
        block = coeffs[lag * d : (lag + 1) * d]
        A[lag] = block.T
    return {"A0": A0, "A": A, "w": int(w), "ridge_lambda": float(ridge_lambda)}


def forecast_step(yhist: Array, model: Dict[str, Array]) -> Array:
    """One-step VAR forecast using the most recent ``w`` latent states."""

    A0 = model["A0"]
    A = model["A"]
    w = model["w"]
    if yhist.shape[0] != w:
        raise ValueError(f"Expected history with length {w}, got {yhist.shape[0]}")

    y_next = A0.copy()
    for j in range(w):
        y_next += A[j] @ yhist[-(j + 1)]
    return y_next


def rollout(Y0: Array, steps: int, model: Dict[str, Array]) -> Array:
    """Roll out a VAR model for ``steps`` future points given the last ``w`` states."""

    if steps < 0:
        raise ValueError("steps must be non-negative")

    history = np.array(Y0, dtype=float)
    if history.ndim != 2:
        raise ValueError("Y0 must have shape (w, d)")
    w = model["w"]
    if history.shape[0] != w:
        raise ValueError(f"Expected seed with shape ({w}, d)")

    d = history.shape[1]
    outputs = np.empty((steps, d), dtype=float)
    for i in range(steps):
        y_next = forecast_step(history, model)
        outputs[i] = y_next
        if w > 1:
            history[:-1] = history[1:]
        history[-1] = y_next
    return outputs


def save_mvar(path: str | Path, model: Dict[str, Array]) -> Path:
    """Persist an MVAR model to disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        A0=model["A0"],
        A=model["A"],
        w=np.array(model["w"], dtype=int),
        ridge_lambda=np.array(model.get("ridge_lambda", 0.0), dtype=float),
    )
    return path


def load_mvar(path: str | Path) -> Dict[str, Array]:
    """Load an MVAR model previously saved with :func:`save_mvar`."""

    with np.load(path, allow_pickle=False) as data:
        A0 = data["A0"]
        A = data["A"]
        w = int(data["w"])  # scalar array
        ridge_lambda = float(data["ridge_lambda"])
    return {"A0": A0, "A": A, "w": w, "ridge_lambda": ridge_lambda}


__all__ = [
    "build_lagged",
    "fit_mvar",
    "forecast_step",
    "rollout",
    "save_mvar",
    "load_mvar",
    "MVARModel",
    "fit_mvar_auto",
    "horizon_test",
    "plot_lag_selection",
    "plot_forecast_trajectories",
    "plot_horizon_test",
]


# ============================================================================
# Enhanced MVAR Implementation with Automatic Lag Selection
# ============================================================================

class MVARModel:
    """
    Enhanced Multivariate Autoregressive model with automatic lag selection.
    
    Mathematical Model:
        y(t_k) = A_0 + sum_{j=1}^w A_j * y(t_{k-j}) + epsilon(t_k)
    
    where:
        - y(t_k) in R^d: latent trajectory
        - A_0 in R^d: bias vector
        - A_j in R^{d x d}: lag matrices
        - w: autoregressive order (default 3, per Alvarez et al. 2025)
        - epsilon ~ N(0, Sigma): residual noise
    
    Attributes:
        max_lag: Maximum lag order to test
        regularization: Ridge penalty for numerical stability
        criterion: Lag selection criterion ("AIC", "BIC", or None)
        noise_type: Noise sampling ("gaussian" or "uniform")
        train_fraction: Fraction of data for training
        A0: Fitted bias vector
        A_list: List of lag matrices
        cov: Residual covariance
        best_lag: Selected lag order
    """
    
    def __init__(
        self,
        max_lag: int = 3,
        regularization: float = 1e-4,
        criterion: str = "AIC",
        noise_type: str = "gaussian",
        train_fraction: float = 0.8,
    ):
        self.max_lag = max_lag
        self.regularization = regularization
        self.criterion = criterion.upper() if criterion else None
        self.noise_type = noise_type.lower()
        self.train_fraction = train_fraction
        
        # Fitted parameters
        self.A0: Optional[Array] = None
        self.A_list: Optional[List[Array]] = None
        self.cov: Optional[Array] = None
        self.best_lag: Optional[int] = None
        self.lag_scores: Dict[int, float] = {}
        self.train_rmse: Optional[float] = None
    
    def fit(self, Y: Array) -> None:
        """
        Fit MVAR model with automatic lag selection.
        
        Args:
            Y: Latent trajectory
                Shape (T, d) for single trajectory
                Shape (C, T, d) for ensemble
        """
        # Flatten ensemble if needed
        if Y.ndim == 3:
            C, T, d = Y.shape
            Y = Y.reshape(-1, d)
        elif Y.ndim != 2:
            raise ValueError(f"Y must have shape (T, d) or (C, T, d), got {Y.shape}")
        
        T, d = Y.shape
        T0 = int(self.train_fraction * T)
        Y_train = Y[:T0]
        
        print(f"MVAR: Training on {T0}/{T} samples")
        print(f"MVAR: Testing lag orders 1 to {self.max_lag}")
        
        # Test each lag
        best_score = np.inf
        best_params = None
        
        for w in range(1, self.max_lag + 1):
            if T0 <= w:
                warnings.warn(f"Insufficient samples for lag {w}")
                continue
            
            # Fit with this lag
            model_dict = fit_mvar(Y_train, w=w, ridge_lambda=self.regularization)
            
            # Compute predictions using the fitted model
            Z, targets = build_lagged(Y_train, w=w)
            n_samples = targets.shape[0]
            
            # Predict: Y_pred = Z @ [A0; A1.T; A2.T; ...]
            # Need to construct parameter matrix properly
            A0 = model_dict["A0"]  # (d,)
            A = model_dict["A"]  # (w, d, d)
            
            # Build prediction manually
            Y_pred = np.tile(A0, (n_samples, 1))
            for j in range(w):
                # Z[:, 1 + j*d:1 + (j+1)*d] @ A[j]
                Y_pred += Z[:, 1 + j*d:1 + (j+1)*d] @ A[j].T
            
            residuals = targets - Y_pred
            
            # Residual covariance
            cov = (residuals.T @ residuals) / (n_samples - w * d - 1)
            
            # Log-likelihood
            sign, logdet = np.linalg.slogdet(cov + 1e-10 * np.eye(d))
            log_lik = -0.5 * n_samples * (d * np.log(2 * np.pi) + logdet + d)
            
            # Criterion
            k = d + w * d * d  # Number of parameters
            if self.criterion == "AIC":
                score = -2 * log_lik + 2 * k
            elif self.criterion == "BIC":
                score = -2 * log_lik + k * np.log(n_samples)
            else:
                score = -log_lik
            
            self.lag_scores[w] = score
            print(f"  Lag {w}: {self.criterion or 'LL'} = {score:.2f}")
            
            if score < best_score:
                best_score = score
                best_params = (w, model_dict, cov)
                self.best_lag = w
        
        if best_params is None:
            raise ValueError("Failed to fit any lag order")
        
        # Store best model
        w, model_dict, cov = best_params
        self.A0 = model_dict["A0"]
        self.A_list = [model_dict["A"][j] for j in range(w)]
        self.cov = cov
        
        print(f"MVAR: Selected lag w* = {self.best_lag}")
        
        # Compute training RMSE
        Z_train, targets_train = build_lagged(Y_train, w=self.best_lag)
        Y_pred_train = self._predict_from_design(Z_train)
        self.train_rmse = np.sqrt(np.mean((targets_train - Y_pred_train) ** 2))
        print(f"MVAR: Training RMSE = {self.train_rmse:.4f}")
    
    def _predict_from_design(self, Z: Array) -> Array:
        """Predict from design matrix (no noise)."""
        A0_col = np.tile(self.A0, (Z.shape[0], 1))
        A_pred = np.zeros_like(A0_col)
        d = self.A0.shape[0]
        for j, A_j in enumerate(self.A_list):
            A_pred += Z[:, 1 + j*d:1 + (j+1)*d] @ A_j.T
        return A0_col + A_pred
    
    def forecast(self, Y_init: Array, steps: int, add_noise: bool = True) -> Array:
        """
        Forecast future states.
        
        Args:
            Y_init: Initial conditions (w, d)
            steps: Number of steps to forecast
            add_noise: Whether to add stochastic noise
            
        Returns:
            Y_forecast: Forecasted trajectory (steps, d)
        """
        if self.best_lag is None:
            raise ValueError("Model must be fitted first")
        
        if Y_init.ndim == 1:
            Y_init = Y_init.reshape(1, -1)
        
        if Y_init.shape[0] < self.best_lag:
            raise ValueError(f"Y_init must have >= {self.best_lag} samples")
        
        Y_init = Y_init[-self.best_lag:]
        d = Y_init.shape[1]
        Y_forecast = np.zeros((steps, d))
        buffer = Y_init.copy()
        
        for t in range(steps):
            y_next = self.A0.copy()
            for j, A_j in enumerate(self.A_list):
                y_next += A_j @ buffer[-(j+1)]
            
            if add_noise:
                y_next += self._sample_noise(d)
            
            Y_forecast[t] = y_next
            buffer = np.vstack([buffer[1:], y_next])
        
        return Y_forecast
    
    def _sample_noise(self, d: int) -> Array:
        """Sample noise from specified distribution."""
        if self.noise_type == "uniform":
            sigma = np.sqrt(np.mean(np.diag(self.cov)))
            eta = np.sqrt(12) * sigma
            return np.random.uniform(-eta/2, eta/2, size=d)
        else:  # gaussian
            return np.random.multivariate_normal(np.zeros(d), self.cov)
    
    def evaluate_forecast(self, Y_true: Array, Y_pred: Array) -> Dict[str, float]:
        """
        Compute forecast accuracy metrics.
        
        Returns:
            metrics: Dict with rmse, mae, corr_mean, corr_per_dim, spectral_ratio
        """
        min_len = min(Y_true.shape[0], Y_pred.shape[0])
        Y_true = Y_true[:min_len]
        Y_pred = Y_pred[:min_len]
        
        rmse = np.sqrt(np.mean((Y_true - Y_pred) ** 2))
        mae = np.mean(np.abs(Y_true - Y_pred))
        
        d = Y_true.shape[1]
        corr_per_dim = np.zeros(d)
        for i in range(d):
            if np.std(Y_true[:, i]) > 1e-10 and np.std(Y_pred[:, i]) > 1e-10:
                corr_per_dim[i] = np.corrcoef(Y_true[:, i], Y_pred[:, i])[0, 1]
        
        corr_mean = np.mean(corr_per_dim)
        
        # Spectral energy
        fft_true = np.fft.rfft(Y_true, axis=0)
        fft_pred = np.fft.rfft(Y_pred, axis=0)
        energy_true = np.sum(np.abs(fft_true) ** 2)
        energy_pred = np.sum(np.abs(fft_pred) ** 2)
        spectral_ratio = energy_pred / (energy_true + 1e-10)
        
        return {
            "rmse": rmse,
            "mae": mae,
            "corr_mean": corr_mean,
            "corr_per_dim": corr_per_dim,
            "spectral_ratio": spectral_ratio,
        }
    
    def to_dict(self) -> Dict:
        """Export model as dictionary (compatible with existing API)."""
        if self.best_lag is None:
            raise ValueError("Model must be fitted first")
        
        A = np.array(self.A_list)  # (w, d, d)
        return {
            "A0": self.A0,
            "A": A,
            "w": self.best_lag,
            "ridge_lambda": self.regularization,
            "cov": self.cov,
        }


def fit_mvar_auto(
    Y: Array,
    max_lag: int = 3,
    criterion: str = "AIC",
    regularization: float = 1e-4,
    **kwargs
) -> MVARModel:
    """
    Fit MVAR model with automatic lag selection.
    
    Args:
        Y: Latent trajectory (T, d) or (C, T, d)
        max_lag: Maximum lag to test (default 3)
        criterion: "AIC", "BIC", or None
        regularization: Ridge penalty
        
    Returns:
        Fitted MVARModel
    """
    model = MVARModel(
        max_lag=max_lag,
        regularization=regularization,
        criterion=criterion,
        **kwargs
    )
    model.fit(Y)
    return model


def horizon_test(
    Y: Array,
    max_lag: int = 3,
    horizon_ratios: List[float] = [0.5, 1.0, 2.0, 3.0],
    n_trials: int = 5,
    train_fraction: float = 0.8,
    **kwargs
) -> Tuple[MVARModel, Dict[float, Dict[str, float]]]:
    """
    Test forecast stability vs prediction horizon.
    
    For each ratio r, trains on T0 and forecasts T1 = r * T0.
    
    Args:
        Y: Full trajectory (T, d)
        max_lag: Maximum lag to test
        horizon_ratios: List of T1/T0 ratios
        n_trials: Trials per ratio
        train_fraction: Training fraction
        
    Returns:
        model: Fitted MVARModel
        results: Dict mapping ratio to metrics
    """
    if Y.ndim == 3:
        Y = Y.reshape(-1, Y.shape[-1])
    
    T, d = Y.shape
    T0 = int(train_fraction * T)
    
    print(f"\nMVAR Horizon Test: T0 = {T0}")
    print(f"Testing ratios T1/T0 = {horizon_ratios}")
    
    # Fit model on T0
    model = fit_mvar_auto(Y[:T0], max_lag=max_lag, **kwargs)
    
    results = {}
    for ratio in horizon_ratios:
        T1 = int(ratio * T0)
        
        if T0 + T1 > T:
            warnings.warn(f"Insufficient data for ratio {ratio}")
            continue
        
        print(f"\n  Ratio {ratio}: T1 = {T1} steps")
        
        rmse_list, mae_list, corr_list = [], [], []
        
        for trial in range(n_trials):
            Y_init = Y[:T0][-model.best_lag:]
            Y_forecast = model.forecast(Y_init, T1, add_noise=True)
            Y_true = Y[T0:T0 + T1]
            
            metrics = model.evaluate_forecast(Y_true, Y_forecast)
            rmse_list.append(metrics["rmse"])
            mae_list.append(metrics["mae"])
            corr_list.append(metrics["corr_mean"])
        
        results[ratio] = {
            "rmse": np.mean(rmse_list),
            "rmse_std": np.std(rmse_list),
            "mae": np.mean(mae_list),
            "mae_std": np.std(mae_list),
            "corr": np.mean(corr_list),
            "corr_std": np.std(corr_list),
        }
        
        print(f"    RMSE: {results[ratio]['rmse']:.4f} ± {results[ratio]['rmse_std']:.4f}")
        print(f"    Corr: {results[ratio]['corr']:.4f} ± {results[ratio]['corr_std']:.4f}")
    
    return model, results


def plot_lag_selection(model: MVARModel, save_path: Optional[str] = None):
    """Plot lag selection criterion vs lag order."""
    if not model.lag_scores:
        raise ValueError("Model must be fitted first")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    lags = sorted(model.lag_scores.keys())
    scores = [model.lag_scores[lag] for lag in lags]
    
    ax.plot(lags, scores, 'o-', linewidth=2, markersize=8)
    ax.axvline(model.best_lag, color='r', linestyle='--',
               label=f'Selected: w* = {model.best_lag}')
    
    ax.set_xlabel('Lag Order w', fontsize=12)
    ax.set_ylabel(f'{model.criterion or "Negative Log-Likelihood"}', fontsize=12)
    ax.set_title('MVAR Lag Selection', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved lag selection plot to {save_path}")
    
    return fig


def plot_forecast_trajectories(
    Y_true: Array,
    Y_forecast: Array,
    dims_to_plot: Optional[List[int]] = None,
    save_path: Optional[str] = None
):
    """Plot forecasted vs true trajectories."""
    d = Y_true.shape[1]
    if dims_to_plot is None:
        dims_to_plot = list(range(min(3, d)))
    
    n_dims = len(dims_to_plot)
    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 3*n_dims))
    if n_dims == 1:
        axes = [axes]
    
    T0 = Y_true.shape[0]
    T1 = Y_forecast.shape[0]
    t_true = np.arange(T0)
    t_forecast = np.arange(T0, T0 + T1)
    
    for idx, dim in enumerate(dims_to_plot):
        ax = axes[idx]
        
        ax.plot(t_true, Y_true[:, dim], 'b-', linewidth=1.5,
               label='Ground Truth', alpha=0.7)
        ax.plot(t_forecast, Y_forecast[:, dim], 'r--', linewidth=1.5,
               label='MVAR Forecast', alpha=0.7)
        ax.axvline(T0, color='k', linestyle=':', alpha=0.5,
                  label='Forecast Start')
        
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel(f'Latent Dim {dim}', fontsize=11)
        ax.set_title(f'Dimension {dim}', fontsize=12)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()
    
    plt.suptitle('MVAR Forecast vs Ground Truth',
                fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved forecast trajectories plot to {save_path}")
    
    return fig


def plot_horizon_test(
    horizon_results: Dict[float, Dict[str, float]],
    save_path: Optional[str] = None
):
    """Plot RMSE and correlation vs T1/T0 ratio."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ratios = sorted(horizon_results.keys())
    rmse_means = [horizon_results[r]["rmse"] for r in ratios]
    rmse_stds = [horizon_results[r]["rmse_std"] for r in ratios]
    corr_means = [horizon_results[r]["corr"] for r in ratios]
    corr_stds = [horizon_results[r]["corr_std"] for r in ratios]
    
    # RMSE vs horizon
    ax1.errorbar(ratios, rmse_means, yerr=rmse_stds,
                marker='o', linewidth=2, markersize=8, capsize=5)
    ax1.set_xlabel('Forecast Horizon T₁/T₀', fontsize=12)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_title('Forecast Error vs Horizon', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Correlation vs horizon
    ax2.errorbar(ratios, corr_means, yerr=corr_stds,
                marker='o', linewidth=2, markersize=8, capsize=5, color='green')
    ax2.set_xlabel('Forecast Horizon T₁/T₀', fontsize=12)
    ax2.set_ylabel('Mean Correlation', fontsize=12)
    ax2.set_title('Forecast Correlation vs Horizon', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.suptitle('MVAR Horizon Test: Generalization Decay',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved horizon test plot to {save_path}")
    
    return fig

