"""
Runtime Analysis Module
========================

Provides comprehensive runtime and computational time analysis for ROM models,
enabling fair comparison between MVAR and LSTM implementations.

This module tracks:
- Training time (data preparation, model fitting, validation)
- Inference time (per-step prediction, batched forecasting)
- Memory usage (model parameters, cached data)
- Throughput metrics (predictions per second, samples per second)
- Scaling characteristics (complexity analysis)

Designed for inclusion in pipeline_summary.json outputs.
"""

import time
import numpy as np
import psutil
import sys
from pathlib import Path
from typing import Dict, Optional, Callable, List, Tuple, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import json


@dataclass
class TimingResult:
    """Container for timing measurements."""
    total_seconds: float
    mean_seconds: Optional[float] = None
    std_seconds: Optional[float] = None
    min_seconds: Optional[float] = None
    max_seconds: Optional[float] = None
    samples: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with non-None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class MemoryResult:
    """Container for memory measurements."""
    peak_mb: float
    current_mb: float
    model_parameters: int
    parameter_memory_mb: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RuntimeProfile:
    """Complete runtime profile for a ROM model."""
    model_name: str
    training: TimingResult
    inference_single_step: TimingResult
    inference_full_trajectory: TimingResult
    memory: MemoryResult
    throughput: Dict[str, float]
    complexity: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to nested dictionary."""
        return {
            'model_name': self.model_name,
            'training': self.training.to_dict(),
            'inference': {
                'single_step': self.inference_single_step.to_dict(),
                'full_trajectory': self.inference_full_trajectory.to_dict()
            },
            'memory': self.memory.to_dict(),
            'throughput': self.throughput,
            'complexity': self.complexity
        }


class RuntimeAnalyzer:
    """
    Comprehensive runtime analysis for ROM models.
    
    Usage Example
    -------------
    >>> analyzer = RuntimeAnalyzer()
    >>> 
    >>> # Time training
    >>> with analyzer.time_operation('training') as timer:
    >>>     model = train_mvar_model(data)
    >>> 
    >>> # Benchmark inference
    >>> inference_times = analyzer.benchmark_inference(
    >>>     forecast_fn, z0, n_steps=100, n_trials=50
    >>> )
    >>> 
    >>> # Build profile
    >>> profile = analyzer.build_profile(
    >>>     model_name='MVAR',
    >>>     training_time=timer.elapsed,
    >>>     inference_times=inference_times,
    >>>     model_params=compute_mvar_params(model)
    >>> )
    >>> 
    >>> # Save to JSON
    >>> analyzer.save_profile(profile, output_dir / 'runtime_profile.json')
    """
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.process = psutil.Process()
        
    @contextmanager
    def time_operation(self, name: str):
        """
        Context manager for timing code blocks.
        
        Parameters
        ----------
        name : str
            Identifier for this timing measurement
            
        Yields
        ------
        timer : SimpleNamespace
            Object with .elapsed attribute after context exit
            
        Example
        -------
        >>> with analyzer.time_operation('data_loading') as timer:
        >>>     data = load_large_dataset()
        >>> print(f"Loading took {timer.elapsed:.2f}s")
        """
        from types import SimpleNamespace
        timer = SimpleNamespace()
        
        start = time.perf_counter()
        try:
            yield timer
        finally:
            elapsed = time.perf_counter() - start
            timer.elapsed = elapsed
            
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(elapsed)
    
    def benchmark_inference(
        self,
        forecast_fn: Callable,
        z0: np.ndarray,
        n_steps: int = 100,
        n_trials: int = 50,
        warmup_trials: int = 5
    ) -> Dict[str, TimingResult]:
        """
        Benchmark inference performance with multiple trials.
        
        Parameters
        ----------
        forecast_fn : callable
            Function that takes (z0, n_steps) and returns forecast trajectory
        z0 : np.ndarray
            Initial latent state
        n_steps : int, default=100
            Number of forecast steps
        n_trials : int, default=50
            Number of timing trials
        warmup_trials : int, default=5
            Number of warmup trials (excluded from statistics)
            
        Returns
        -------
        timings : dict
            Dictionary with 'single_step' and 'full_trajectory' TimingResult objects
        """
        print(f"\n🔬 Benchmarking inference ({n_trials} trials, {n_steps} steps)...")
        
        # Warmup
        for _ in range(warmup_trials):
            _ = forecast_fn(z0, n_steps)
        
        # Single-step inference timing
        single_step_times = []
        for _ in range(n_trials):
            start = time.perf_counter()
            _ = forecast_fn(z0, 1)
            elapsed = time.perf_counter() - start
            single_step_times.append(elapsed)
        
        # Full trajectory timing
        full_traj_times = []
        for _ in range(n_trials):
            start = time.perf_counter()
            _ = forecast_fn(z0, n_steps)
            elapsed = time.perf_counter() - start
            full_traj_times.append(elapsed)
        
        single_step_times = np.array(single_step_times)
        full_traj_times = np.array(full_traj_times)
        
        return {
            'single_step': TimingResult(
                total_seconds=single_step_times.sum(),
                mean_seconds=float(single_step_times.mean()),
                std_seconds=float(single_step_times.std()),
                min_seconds=float(single_step_times.min()),
                max_seconds=float(single_step_times.max()),
                samples=n_trials
            ),
            'full_trajectory': TimingResult(
                total_seconds=full_traj_times.sum(),
                mean_seconds=float(full_traj_times.mean()),
                std_seconds=float(full_traj_times.std()),
                min_seconds=float(full_traj_times.min()),
                max_seconds=float(full_traj_times.max()),
                samples=n_trials
            )
        }
    
    def measure_memory(
        self,
        model_params: int,
        param_dtype: np.dtype = np.float64
    ) -> MemoryResult:
        """
        Measure current memory usage and model memory footprint.
        
        Parameters
        ----------
        model_params : int
            Total number of model parameters
        param_dtype : np.dtype, default=np.float64
            Data type of model parameters
            
        Returns
        -------
        memory : MemoryResult
            Memory usage statistics
        """
        mem_info = self.process.memory_info()
        
        # Current memory usage
        current_mb = mem_info.rss / 1024 / 1024
        
        # Peak memory (platform-dependent)
        try:
            peak_mb = self.process.memory_info().peak_wset / 1024 / 1024  # Windows
        except AttributeError:
            peak_mb = current_mb  # Unix systems don't have peak_wset
        
        # Model parameter memory
        bytes_per_param = np.dtype(param_dtype).itemsize
        param_memory_mb = (model_params * bytes_per_param) / 1024 / 1024
        
        return MemoryResult(
            peak_mb=float(peak_mb),
            current_mb=float(current_mb),
            model_parameters=int(model_params),
            parameter_memory_mb=float(param_memory_mb)
        )
    
    def compute_throughput(
        self,
        inference_timing: TimingResult,
        n_steps: int,
        latent_dim: int
    ) -> Dict[str, float]:
        """
        Compute throughput metrics.
        
        Parameters
        ----------
        inference_timing : TimingResult
            Full trajectory inference timing
        n_steps : int
            Number of forecast steps
        latent_dim : int
            Latent dimension
            
        Returns
        -------
        throughput : dict
            Throughput metrics (steps/sec, predictions/sec, etc.)
        """
        mean_time = inference_timing.mean_seconds
        
        return {
            'steps_per_second': float(n_steps / mean_time),
            'predictions_per_second': float((n_steps * latent_dim) / mean_time),
            'seconds_per_step': float(mean_time / n_steps),
            'microseconds_per_step': float((mean_time / n_steps) * 1e6)
        }
    
    def analyze_complexity(
        self,
        model_name: str,
        model_params: int,
        latent_dim: int,
        lag: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        n_layers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze computational complexity.
        
        Parameters
        ----------
        model_name : str
            'MVAR' or 'LSTM'
        model_params : int
            Total number of parameters
        latent_dim : int
            Latent dimension
        lag : int, optional
            MVAR lag order
        hidden_dim : int, optional
            LSTM hidden dimension
        n_layers : int, optional
            LSTM number of layers
            
        Returns
        -------
        complexity : dict
            Complexity analysis
        """
        complexity = {
            'total_parameters': int(model_params),
            'latent_dimension': int(latent_dim)
        }
        
        if model_name.upper() == 'MVAR':
            if lag is not None:
                # MVAR: d parameters for bias + p*d² for coefficient matrices
                expected_params = latent_dim + lag * latent_dim * latent_dim
                complexity['lag_order'] = int(lag)
                complexity['expected_parameters'] = int(expected_params)
                complexity['parameters_per_lag'] = int(latent_dim * latent_dim)
                complexity['inference_flops_per_step'] = int(lag * latent_dim * latent_dim * 2)  # MACs
                complexity['training_complexity'] = f"O(N_samples * p * d²)" 
                complexity['inference_complexity'] = f"O(p * d²)"
                
        elif model_name.upper() == 'LSTM':
            if hidden_dim is not None and n_layers is not None:
                # LSTM gates: 4 * (hidden * (hidden + latent + 1))
                params_per_layer = 4 * (hidden_dim * (hidden_dim + latent_dim + 1))
                complexity['hidden_dimension'] = int(hidden_dim)
                complexity['num_layers'] = int(n_layers)
                complexity['expected_parameters'] = int(params_per_layer * n_layers)
                complexity['parameters_per_layer'] = int(params_per_layer)
                complexity['inference_flops_per_step'] = int(4 * hidden_dim * (hidden_dim + latent_dim) * 2 * n_layers)
                complexity['training_complexity'] = f"O(N_samples * N_steps * h * (h + d))"
                complexity['inference_complexity'] = f"O(h * (h + d))"
        
        return complexity
    
    def build_profile(
        self,
        model_name: str,
        training_time: float,
        inference_times: Dict[str, TimingResult],
        model_params: int,
        latent_dim: int,
        n_forecast_steps: int,
        **complexity_kwargs
    ) -> RuntimeProfile:
        """
        Build complete runtime profile.
        
        Parameters
        ----------
        model_name : str
            Model identifier ('MVAR' or 'LSTM')
        training_time : float
            Total training time in seconds
        inference_times : dict
            Dictionary with 'single_step' and 'full_trajectory' TimingResult objects
        model_params : int
            Total number of model parameters
        latent_dim : int
            Latent space dimension
        n_forecast_steps : int
            Number of steps used in inference benchmark
        **complexity_kwargs
            Additional arguments for complexity analysis (lag, hidden_dim, etc.)
            
        Returns
        -------
        profile : RuntimeProfile
            Complete runtime profile
        """
        training_timing = TimingResult(total_seconds=training_time)
        
        memory = self.measure_memory(model_params)
        
        throughput = self.compute_throughput(
            inference_times['full_trajectory'],
            n_forecast_steps,
            latent_dim
        )
        
        complexity = self.analyze_complexity(
            model_name,
            model_params,
            latent_dim,
            **complexity_kwargs
        )
        
        return RuntimeProfile(
            model_name=model_name,
            training=training_timing,
            inference_single_step=inference_times['single_step'],
            inference_full_trajectory=inference_times['full_trajectory'],
            memory=memory,
            throughput=throughput,
            complexity=complexity
        )
    
    def compare_models(
        self,
        profiles: List[RuntimeProfile]
    ) -> Dict[str, Any]:
        """
        Generate comparative analysis between multiple models.
        
        Parameters
        ----------
        profiles : list of RuntimeProfile
            Model profiles to compare
            
        Returns
        -------
        comparison : dict
            Comparative metrics and speedup factors
        """
        if len(profiles) < 2:
            return {}
        
        comparison = {
            'models': [p.model_name for p in profiles],
            'training_time_ratio': {},
            'inference_speedup': {},
            'memory_ratio': {},
            'parameter_ratio': {}
        }
        
        # Use first model as baseline
        baseline = profiles[0]
        
        for profile in profiles[1:]:
            model_name = profile.model_name
            
            # Training time ratio
            training_ratio = profile.training.total_seconds / baseline.training.total_seconds
            comparison['training_time_ratio'][f'{model_name}_vs_{baseline.model_name}'] = float(training_ratio)
            
            # Inference speedup (lower time = higher speedup)
            inference_speedup = baseline.inference_full_trajectory.mean_seconds / profile.inference_full_trajectory.mean_seconds
            comparison['inference_speedup'][f'{model_name}_vs_{baseline.model_name}'] = float(inference_speedup)
            
            # Memory ratio
            memory_ratio = profile.memory.parameter_memory_mb / baseline.memory.parameter_memory_mb
            comparison['memory_ratio'][f'{model_name}_vs_{baseline.model_name}'] = float(memory_ratio)
            
            # Parameter count ratio
            param_ratio = profile.memory.model_parameters / baseline.memory.model_parameters
            comparison['parameter_ratio'][f'{model_name}_vs_{baseline.model_name}'] = float(param_ratio)
        
        # Determine winner for each category
        fastest_training = min(profiles, key=lambda p: p.training.total_seconds)
        fastest_inference = min(profiles, key=lambda p: p.inference_full_trajectory.mean_seconds)
        smallest_memory = min(profiles, key=lambda p: p.memory.parameter_memory_mb)
        
        comparison['winners'] = {
            'fastest_training': fastest_training.model_name,
            'fastest_inference': fastest_inference.model_name,
            'smallest_memory': smallest_memory.model_name
        }
        
        return comparison
    
    def save_profile(self, profile: RuntimeProfile, filepath: Path) -> None:
        """Save runtime profile to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(profile.to_dict(), f, indent=2)
        
        print(f"✓ Runtime profile saved: {filepath}")
    
    def save_comparison(self, comparison: Dict, filepath: Path) -> None:
        """Save model comparison to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"✓ Runtime comparison saved: {filepath}")


def compute_mvar_params(mvar_model: Dict) -> int:
    """
    Compute total parameter count for MVAR model.
    
    Parameters
    ----------
    mvar_model : dict
        MVAR model dictionary with 'A_matrices' (coefficient matrices) and 'A_companion' (companion form)
        
    Returns
    -------
    n_params : int
        Total number of parameters
    """
    # The model uses A_companion which is the full coefficient matrix
    # Shape: (d, p*d) where p is lag and d is latent dimension
    A_companion = mvar_model['A_companion']  # Shape: (d, p*d)
    
    return A_companion.size


def compute_lstm_params(lstm_model) -> int:
    """
    Compute total parameter count for LSTM model.
    
    Parameters
    ----------
    lstm_model : torch.nn.Module or LatentLSTMROM
        LSTM model
        
    Returns
    -------
    n_params : int
        Total number of parameters
    """
    import torch
    
    if hasattr(lstm_model, 'model'):
        model = lstm_model.model  # LatentLSTMROM wrapper
    else:
        model = lstm_model
    
    return sum(p.numel() for p in model.parameters())


# Convenience function for quick benchmarking
def quick_benchmark(
    model_name: str,
    forecast_fn: Callable,
    z0: np.ndarray,
    training_time: float,
    model_params: int,
    latent_dim: int,
    n_steps: int = 100,
    n_trials: int = 50,
    **complexity_kwargs
) -> RuntimeProfile:
    """
    Quick benchmarking utility for a single model.
    
    Returns complete RuntimeProfile ready for JSON export.
    """
    analyzer = RuntimeAnalyzer()
    
    inference_times = analyzer.benchmark_inference(
        forecast_fn, z0, n_steps=n_steps, n_trials=n_trials
    )
    
    profile = analyzer.build_profile(
        model_name=model_name,
        training_time=training_time,
        inference_times=inference_times,
        model_params=model_params,
        latent_dim=latent_dim,
        n_forecast_steps=n_steps,
        **complexity_kwargs
    )
    
    return profile


__all__ = [
    'RuntimeAnalyzer',
    'RuntimeProfile',
    'TimingResult',
    'MemoryResult',
    'compute_mvar_params',
    'compute_lstm_params',
    'quick_benchmark'
]
