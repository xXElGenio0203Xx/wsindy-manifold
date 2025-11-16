"""
Non-stationarity handling for latent MVAR (Alvarez et al., 2025)

Implements ADF testing, differencing, and detrending for time series to ensure
stationarity before MVAR modeling. Provides inverse transforms for forecasting.

Key features:
- Augmented Dickey-Fuller (ADF) testing with α=0.01 significance
- Per-coordinate transforms: raw, detrend, first difference, seasonal difference
- Per-case processing (no cross-case leakage)
- Inverse transforms for level reconstruction
- Configurable thresholds and policies

Reference: Alvarez et al. (2025), Section on non-stationarity handling
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import warnings

try:
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn(
        "statsmodels not available. Using minimal ADF implementation. "
        "Install statsmodels for better ADF testing: pip install statsmodels"
    )


@dataclass
class CoordDecision:
    """Decision and parameters for a single coordinate's stationarity transform."""
    mode: str  # "raw" | "detrend" | "diff" | "seasonal_diff" | "diff_then_detrend"
    adf_variant: str  # "trend" (ct), "const" (c), "nc" (no constant)
    adf_pvalue: float
    adf_lag: int
    beta0: Optional[float] = None  # Detrend intercept
    beta1: Optional[float] = None  # Detrend slope
    init_level: Optional[float] = None  # Initial level for differenced series
    seasonal_period: Optional[int] = None
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'mode': self.mode,
            'adf_variant': self.adf_variant,
            'adf_pvalue': float(self.adf_pvalue),
            'adf_lag': int(self.adf_lag),
            'beta0': float(self.beta0) if self.beta0 is not None else None,
            'beta1': float(self.beta1) if self.beta1 is not None else None,
            'init_level': float(self.init_level) if self.init_level is not None else None,
            'seasonal_period': self.seasonal_period,
            'notes': self.notes,
            'stationary': self.adf_pvalue < 0.01,  # Using default alpha
            'requires_transform': self.mode != 'raw'
        }


@dataclass
class CaseMeta:
    """Metadata for a single case's stationarity processing."""
    decisions: List[CoordDecision]  # Per-coordinate decisions (length d)
    trim_left: int  # Samples trimmed due to differencing
    K_in: int  # Input length
    K_out: int  # Output length after transform
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'decisions': [dec.to_dict() for dec in self.decisions],
            'trim_left': int(self.trim_left),
            'K_in': int(self.K_in),
            'K_out': int(self.K_out),
            'num_coordinates': len(self.decisions),
            'stationary_count': sum(1 for d in self.decisions if d.adf_pvalue < 0.01),
            'transform_count': sum(1 for d in self.decisions if d.mode != 'raw'),
        }


class NonStationarityProcessor:
    """
    Process time series for stationarity using ADF tests and transforms.
    
    Workflow:
    1. fit(Y_cases): Run ADF tests and decide transforms per coordinate/case
    2. transform(Y_cases): Apply transforms to get stationary series
    3. inverse_levels(Z_forecasts): Convert forecasts back to original scale
    
    Parameters
    ----------
    adf_alpha : float, default=0.01
        Significance level for ADF test (paper uses 0.01)
    adf_max_lags : int or None, default=None
        Maximum lags for ADF test (None = auto via BIC)
    trend_policy : str, default="auto"
        How to handle trends: "auto", "always_detrend", "never_detrend"
    seasonal_period : int or None, default=None
        Period for seasonal differencing (if applicable)
    verbose : bool, default=True
        Print warnings and notes
    
    Examples
    --------
    >>> proc = NonStationarityProcessor(adf_alpha=0.01)
    >>> proc.fit([Y_case_1, Y_case_2])  # Each (d, K_c)
    >>> Z_cases, meta = proc.transform([Y_case_1, Y_case_2])
    >>> # ... fit MVAR on Z_cases ...
    >>> # ... generate forecasts Z_forecasts ...
    >>> Yhat = proc.inverse_levels(Z_forecasts, meta)
    """
    
    def __init__(
        self,
        adf_alpha: float = 0.01,
        adf_max_lags: Optional[int] = None,
        trend_policy: str = "auto",
        seasonal_period: Optional[int] = None,
        verbose: bool = True,
    ):
        self.adf_alpha = adf_alpha
        self.adf_max_lags = adf_max_lags
        self.trend_policy = trend_policy
        self.seasonal_period = seasonal_period
        self.verbose = verbose
        
        self.case_meta: List[CaseMeta] = []
        self.d: Optional[int] = None
        self._fitted: bool = False
    
    # ========== Public API ==========
    
    def fit(self, Y_cases: List[np.ndarray]) -> "NonStationarityProcessor":
        """
        Analyze stationarity and decide transforms for each coordinate/case.
        
        Parameters
        ----------
        Y_cases : list of ndarray
            Each array has shape (d, K_c) where d = latent dimension,
            K_c = number of time steps for that case. Columns are time-ordered.
        
        Returns
        -------
        self : NonStationarityProcessor
            Fitted processor with transform decisions stored
        """
        if not Y_cases:
            raise ValueError("Y_cases cannot be empty")
        
        self.case_meta = []
        self.d = Y_cases[0].shape[0]
        
        for case_idx, Y in enumerate(Y_cases):
            d, K = Y.shape
            if d != self.d:
                raise ValueError(
                    f"Dimension mismatch: case {case_idx} has d={d}, expected {self.d}"
                )
            
            decisions = []
            trim_left = 0
            
            for coord_idx in range(d):
                y = Y[coord_idx, :].astype(np.float64)
                
                # Check for near-constant series
                if np.std(y) < 1e-10:
                    if self.verbose:
                        warnings.warn(
                            f"Case {case_idx}, coord {coord_idx}: near-constant series, "
                            f"using raw levels"
                        )
                    dec = CoordDecision(
                        mode="raw",
                        adf_variant="const",
                        adf_pvalue=1.0,
                        adf_lag=0,
                        notes="near-constant series"
                    )
                else:
                    dec = self._decide_transform_for_coord(y, case_idx, coord_idx)
                
                decisions.append(dec)
                
                # Track maximum trim needed
                if dec.mode in ("diff", "diff_then_detrend"):
                    trim_left = max(trim_left, 1)
                elif dec.mode == "seasonal_diff" and dec.seasonal_period:
                    trim_left = max(trim_left, dec.seasonal_period)
            
            meta = CaseMeta(
                decisions=decisions,
                trim_left=trim_left,
                K_in=K,
                K_out=K - trim_left
            )
            self.case_meta.append(meta)
        
        self._fitted = True
        return self
    
    def transform(
        self, Y_cases: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[CaseMeta]]:
        """
        Apply decided transforms to make series stationary.
        
        Parameters
        ----------
        Y_cases : list of ndarray
            Each array has shape (d, K_c), same as in fit()
        
        Returns
        -------
        Z_cases : list of ndarray
            Transformed series, each shape (d, K_out) where K_out = K_in - trim_left
        case_meta : list of CaseMeta
            Metadata needed for inverse transforms
        """
        self._check_fitted()
        
        Z_cases = []
        for case_idx, Y in enumerate(Y_cases):
            meta = self.case_meta[case_idx]
            d, K = Y.shape
            
            Z_coords = []
            for coord_idx in range(d):
                y = Y[coord_idx, :].astype(np.float64)
                dec = meta.decisions[coord_idx]
                z = self._apply_transform(y, dec)
                Z_coords.append(z)
            
            # Align lengths (differencing reduces length)
            min_len = min(len(z) for z in Z_coords)
            Z = np.vstack([z[-min_len:] for z in Z_coords])
            Z_cases.append(Z)
        
        return Z_cases, self.case_meta
    
    def inverse_levels(
        self,
        Z_forecasts_cases: List[np.ndarray],
        case_meta: Optional[List[CaseMeta]] = None
    ) -> List[np.ndarray]:
        """
        Convert transformed-scale forecasts back to original levels.
        
        Parameters
        ----------
        Z_forecasts_cases : list of ndarray
            Forecasts on transformed scale, each shape (d, T_pred)
        case_meta : list of CaseMeta, optional
            Metadata from transform(). If None, uses self.case_meta
        
        Returns
        -------
        Yhat_cases : list of ndarray
            Forecasts on original level scale, each shape (d, T_pred)
        """
        if case_meta is None:
            case_meta = self.case_meta
        
        Yhat_cases = []
        for case_idx, Zf in enumerate(Z_forecasts_cases):
            meta = case_meta[case_idx]
            d, T_pred = Zf.shape
            
            Yhat = np.zeros((d, T_pred))
            for coord_idx in range(d):
                dec = meta.decisions[coord_idx]
                Yhat[coord_idx, :] = self._inverse_transform(
                    Zf[coord_idx, :], dec, start_level=dec.init_level
                )
            
            Yhat_cases.append(Yhat)
        
        return Yhat_cases
    
    def summary(self) -> str:
        """
        Generate human-readable summary of stationarity analysis.
        
        Returns
        -------
        str
            Multi-line summary with ADF results and transform decisions
        """
        self._check_fitted()
        
        lines = [
            f"NonStationarityProcessor Summary",
            f"{'='*60}",
            f"ADF significance level: α={self.adf_alpha}",
            f"Cases: {len(self.case_meta)}",
            f"Latent dimension: {self.d}",
            ""
        ]
        
        for case_idx, meta in enumerate(self.case_meta):
            lines.append(f"Case {case_idx + 1}:")
            lines.append(f"  K_in={meta.K_in}, K_out={meta.K_out}, trim={meta.trim_left}")
            lines.append(f"  Stationary: {meta.to_dict()['stationary_count']}/{len(meta.decisions)}")
            lines.append(f"  Transforms: {meta.to_dict()['transform_count']}/{len(meta.decisions)}")
            lines.append("")
            
            for coord_idx, dec in enumerate(meta.decisions):
                status = "✓ stationary" if dec.adf_pvalue < self.adf_alpha else "✗ non-stationary"
                lines.append(
                    f"    Coord {coord_idx}: {status} (p={dec.adf_pvalue:.4f}, "
                    f"mode={dec.mode}, variant={dec.adf_variant}, lag={dec.adf_lag})"
                )
                if dec.notes:
                    lines.append(f"      Note: {dec.notes}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export full analysis results as dictionary for JSON serialization.
        
        Returns
        -------
        dict
            Complete results including ADF p-values, decisions, and metadata
        """
        self._check_fitted()
        
        return {
            'adf_alpha': self.adf_alpha,
            'adf_max_lags': self.adf_max_lags,
            'trend_policy': self.trend_policy,
            'seasonal_period': self.seasonal_period,
            'num_cases': len(self.case_meta),
            'latent_dimension': self.d,
            'cases': [meta.to_dict() for meta in self.case_meta],
            'summary': {
                'total_coordinates': self.d * len(self.case_meta),
                'stationary_coordinates': sum(
                    meta.to_dict()['stationary_count'] for meta in self.case_meta
                ),
                'non_stationary_coordinates': sum(
                    self.d - meta.to_dict()['stationary_count'] for meta in self.case_meta
                ),
                'transforms_applied': sum(
                    meta.to_dict()['transform_count'] for meta in self.case_meta
                ),
            }
        }
    
    # ========== Internal Methods ==========
    
    def _decide_transform_for_coord(
        self, y: np.ndarray, case_idx: int, coord_idx: int
    ) -> CoordDecision:
        """
        Decide stationarity transform for a single coordinate using ADF tests.
        
        Algorithm (per paper):
        1. Try ADF with trend (constant + linear trend)
        2. If fails, try ADF with constant only
        3. If still fails, try detrending
        4. If still fails, difference once
        5. Verify differenced series is stationary
        """
        # Step 1: ADF with trend
        p_trend, lag_trend = self._adf(y, variant="ct")
        if p_trend < self.adf_alpha:
            return CoordDecision(
                mode="raw",
                adf_variant="trend",
                adf_pvalue=p_trend,
                adf_lag=lag_trend,
                notes="stationary with trend component"
            )
        
        # Step 2: ADF with constant only
        p_const, lag_const = self._adf(y, variant="c")
        if p_const < self.adf_alpha:
            return CoordDecision(
                mode="raw",
                adf_variant="const",
                adf_pvalue=p_const,
                adf_lag=lag_const,
                notes="stationary around constant mean"
            )
        
        # Step 3: Check if detrending helps (trend-stationary)
        beta0, beta1, resid = self._linear_detrend(y)
        p_resid, lag_resid = self._adf(resid, variant="c")
        if p_resid < self.adf_alpha:
            return CoordDecision(
                mode="detrend",
                adf_variant="const",
                adf_pvalue=p_resid,
                adf_lag=lag_resid,
                beta0=beta0,
                beta1=beta1,
                notes="trend-stationary, detrended"
            )
        
        # Step 4: First difference (unit root)
        if len(y) < 3:
            warnings.warn(
                f"Case {case_idx}, coord {coord_idx}: series too short for differencing"
            )
            return CoordDecision(
                mode="raw",
                adf_variant="const",
                adf_pvalue=p_const,
                adf_lag=lag_const,
                notes="series too short, using raw"
            )
        
        dy = np.diff(y)
        p_diff, lag_diff = self._adf(dy, variant="c")
        
        notes = ""
        if p_diff >= self.adf_alpha:
            notes = "Still non-stationary after 1st difference; may need 2nd diff or seasonal"
            if self.verbose:
                warnings.warn(
                    f"Case {case_idx}, coord {coord_idx}: {notes}"
                )
        
        # Check for over-differencing (lag-1 ACF strongly negative)
        if len(dy) > 1:
            acf1 = np.corrcoef(dy[:-1], dy[1:])[0, 1]
            if acf1 < -0.5:
                notes += "; ACF(1) < -0.5 suggests over-differencing"
                if self.verbose:
                    warnings.warn(
                        f"Case {case_idx}, coord {coord_idx}: strong negative ACF after "
                        f"differencing (ACF1={acf1:.3f}), consider detrending instead"
                    )
        
        return CoordDecision(
            mode="diff",
            adf_variant="const",
            adf_pvalue=p_diff,
            adf_lag=lag_diff,
            init_level=float(y[0]),
            notes=notes
        )
    
    def _apply_transform(self, y: np.ndarray, dec: CoordDecision) -> np.ndarray:
        """Apply the decided transform to a series."""
        if dec.mode == "raw":
            return y
        
        elif dec.mode == "detrend":
            _, _, resid = self._linear_detrend(y, beta0=dec.beta0, beta1=dec.beta1)
            return resid
        
        elif dec.mode == "diff":
            return np.diff(y)
        
        elif dec.mode == "seasonal_diff":
            s = dec.seasonal_period or 1
            return y[s:] - y[:-s]
        
        elif dec.mode == "diff_then_detrend":
            dy = np.diff(y)
            beta0, beta1, resid = self._linear_detrend(dy)
            # Store parameters for inverse
            dec.beta0 = beta0
            dec.beta1 = beta1
            return resid
        
        else:
            raise ValueError(f"Unknown transform mode: {dec.mode}")
    
    def _inverse_transform(
        self, z: np.ndarray, dec: CoordDecision, start_level: Optional[float]
    ) -> np.ndarray:
        """Apply inverse transform to reconstruct original scale."""
        if dec.mode == "raw":
            return z
        
        elif dec.mode == "detrend":
            k = np.arange(len(z))
            trend = (dec.beta0 or 0.0) + (dec.beta1 or 0.0) * k
            return trend + z
        
        elif dec.mode == "diff":
            if start_level is None:
                raise ValueError("init_level required for inverse of differenced series")
            # Cumulative sum to integrate differences
            y = np.empty(len(z))
            level = start_level
            for t in range(len(z)):
                level = level + z[t]
                y[t] = level
            return y
        
        elif dec.mode == "seasonal_diff":
            raise NotImplementedError(
                "Provide initial s levels to invert seasonal differencing"
            )
        
        elif dec.mode == "diff_then_detrend":
            # Add trend back to detrended differences, then integrate
            k = np.arange(len(z))
            dy_hat = (dec.beta0 or 0.0) + (dec.beta1 or 0.0) * k + z
            
            if start_level is None:
                raise ValueError("init_level required for inverse of differenced series")
            
            y = np.empty(len(z))
            level = start_level
            for t in range(len(z)):
                level = level + dy_hat[t]
                y[t] = level
            return y
        
        else:
            raise ValueError(f"Unknown transform mode: {dec.mode}")
    
    # ========== Utilities ==========
    
    def _adf(self, y: np.ndarray, variant: str = "c") -> Tuple[float, int]:
        """
        Run Augmented Dickey-Fuller test.
        
        Parameters
        ----------
        y : ndarray
            Time series to test
        variant : str
            "nc" (no constant), "c" (constant), "ct" (constant + trend)
        
        Returns
        -------
        p_value : float
            ADF test p-value
        used_lag : int
            Number of lags used in test
        """
        if HAS_STATSMODELS:
            try:
                regression = {"nc": "nc", "c": "c", "ct": "ct"}[variant]
                result = adfuller(
                    y,
                    maxlag=self.adf_max_lags,
                    regression=regression,
                    autolag="BIC" if self.adf_max_lags is None else None
                )
                p_value = float(result[1])
                used_lag = int(result[2])
                return p_value, used_lag
            except Exception as e:
                if self.verbose:
                    warnings.warn(f"ADF test failed: {e}, using fallback")
                return self._adf_minimal(y, variant=variant)
        else:
            return self._adf_minimal(y, variant=variant)
    
    def _linear_detrend(
        self,
        y: np.ndarray,
        beta0: Optional[float] = None,
        beta1: Optional[float] = None
    ) -> Tuple[float, float, np.ndarray]:
        """
        Fit or apply linear detrending.
        
        Parameters
        ----------
        y : ndarray
            Time series
        beta0, beta1 : float, optional
            If provided, use these parameters instead of fitting
        
        Returns
        -------
        beta0 : float
            Intercept
        beta1 : float
            Slope
        resid : ndarray
            Detrended residuals
        """
        k = np.arange(len(y))
        
        if beta0 is None or beta1 is None:
            # Fit OLS: y = beta0 + beta1 * k + residual
            X = np.column_stack([np.ones_like(k), k])
            theta, *_ = np.linalg.lstsq(X, y, rcond=None)
            beta0, beta1 = float(theta[0]), float(theta[1])
        
        trend = beta0 + beta1 * k
        resid = y - trend
        
        return beta0, beta1, resid
    
    def _adf_minimal(
        self, y: np.ndarray, variant: str = "c", max_lag: int = 8
    ) -> Tuple[float, int]:
        """
        Minimal ADF implementation (fallback when statsmodels unavailable).
        
        This is a simplified version that may not be as accurate as statsmodels.
        Returns conservative p-value estimates.
        """
        # Use variance ratio as a simple heuristic
        # If variance of differences << variance of levels, likely stationary
        if len(y) < 10:
            return 1.0, 0  # Too short, assume non-stationary
        
        dy = np.diff(y)
        var_levels = np.var(y)
        var_diffs = np.var(dy)
        
        if var_levels < 1e-10:
            return 0.5, 0  # Near constant
        
        ratio = var_diffs / var_levels
        
        # Heuristic: ratio < 0.1 suggests stationarity
        # This is very rough and should be replaced with proper ADF
        if ratio < 0.05:
            p_value = 0.001
        elif ratio < 0.1:
            p_value = 0.05
        elif ratio < 0.2:
            p_value = 0.15
        else:
            p_value = 0.5
        
        return p_value, 0
    
    def _check_fitted(self):
        """Raise error if not fitted."""
        if not self._fitted:
            raise RuntimeError(
                "NonStationarityProcessor not fitted. Call fit() first."
            )
