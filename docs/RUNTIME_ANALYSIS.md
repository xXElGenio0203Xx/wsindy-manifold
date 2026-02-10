# Runtime Analysis Module Documentation

## Overview

The runtime analysis module (`src/rectsim/runtime_analyzer.py`) provides comprehensive performance profiling for ROM models, enabling fair comparison between MVAR and LSTM implementations.

## Features

### 1. **Training Time Tracking**
- Measures total training time
- Supports context manager for easy timing
- Minimal overhead

### 2. **Inference Benchmarking**
- Single-step prediction timing
- Full trajectory forecast timing
- Multiple trials with warmup
- Statistical analysis (mean, std, min, max)

### 3. **Memory Analysis**
- Peak memory usage
- Current process memory
- Model parameter count
- Parameter memory footprint

### 4. **Throughput Metrics**
- Steps per second
- Predictions per second
- Microseconds per step

### 5. **Complexity Analysis**
- Parameter counting
- FLOPs estimation
- Big-O notation for training/inference
- Model-specific metrics (lag order, hidden dimensions, etc.)

### 6. **Model Comparison**
- Training time ratios
- Inference speedup factors
- Memory efficiency ratios
- Winner determination by category

## Integration with Pipeline

The runtime analyzer is automatically integrated into the unified ROM pipeline (`ROM_pipeline.py`):

1. **Training Phase**: Automatically times MVAR and LSTM training
2. **Benchmarking Phase** (Step 4.5): Benchmarks inference performance
3. **Output**: Saves individual profiles and comparative analysis

## Output Files

### Individual Model Profiles

**Location**: `oscar_output/<experiment>/MVAR/runtime_profile.json` (or `LSTM/runtime_profile.json`)

**Structure**:
```json
{
  "model_name": "MVAR",
  "training": {
    "total_seconds": 12.5
  },
  "inference": {
    "single_step": {
      "mean_seconds": 0.00023,
      "std_seconds": 0.00005,
      "min_seconds": 0.00018,
      "max_seconds": 0.00045,
      "samples": 50
    },
    "full_trajectory": {
      "mean_seconds": 0.023,
      "std_seconds": 0.002,
      "min_seconds": 0.021,
      "max_seconds": 0.027,
      "samples": 50
    }
  },
  "memory": {
    "peak_mb": 245.3,
    "current_mb": 234.1,
    "model_parameters": 2020,
    "parameter_memory_mb": 0.016
  },
  "throughput": {
    "steps_per_second": 4347.8,
    "predictions_per_second": 86956.5,
    "seconds_per_step": 0.00023,
    "microseconds_per_step": 230.0
  },
  "complexity": {
    "total_parameters": 2020,
    "latent_dimension": 20,
    "lag_order": 5,
    "expected_parameters": 2020,
    "parameters_per_lag": 400,
    "inference_flops_per_step": 4000,
    "training_complexity": "O(N_samples * p * d²)",
    "inference_complexity": "O(p * d²)"
  }
}
```

### Comparative Analysis

**Location**: `oscar_output/<experiment>/runtime_comparison.json`

**Structure**:
```json
{
  "models": ["MVAR", "LSTM"],
  "training_time_ratio": {
    "LSTM_vs_MVAR": 3.45
  },
  "inference_speedup": {
    "LSTM_vs_MVAR": 0.82
  },
  "memory_ratio": {
    "LSTM_vs_MVAR": 2.15
  },
  "parameter_ratio": {
    "LSTM_vs_MVAR": 2.15
  },
  "winners": {
    "fastest_training": "MVAR",
    "fastest_inference": "MVAR",
    "smallest_memory": "MVAR"
  }
}
```

### Pipeline Summary Integration

Runtime analysis results are automatically included in `summary.json`:

```json
{
  "experiment_name": "production_test",
  "runtime_analysis": {
    "profiles": [
      { "model_name": "MVAR", ... },
      { "model_name": "LSTM", ... }
    ],
    "comparison": { ... }
  },
  ...
}
```

## Usage Examples

### Standalone Usage

```python
from rectsim.runtime_analyzer import RuntimeAnalyzer, quick_benchmark

# Create analyzer
analyzer = RuntimeAnalyzer()

# Time training
with analyzer.time_operation('training') as timer:
    model = train_model(data)

training_time = timer.elapsed

# Benchmark inference
profile = quick_benchmark(
    model_name='MVAR',
    forecast_fn=mvar_forecast,
    z0=initial_state,
    training_time=training_time,
    model_params=2020,
    latent_dim=20,
    n_steps=100,
    n_trials=50,
    lag=5
)

# Save results
analyzer.save_profile(profile, output_dir / 'runtime_profile.json')
```

### Manual Profiling

```python
from rectsim.runtime_analyzer import RuntimeAnalyzer

analyzer = RuntimeAnalyzer()

# 1. Time training
with analyzer.time_operation('model_training') as timer:
    model = train_my_model(data)

# 2. Benchmark inference
inference_times = analyzer.benchmark_inference(
    forecast_fn=my_forecast_function,
    z0=initial_state,
    n_steps=100,
    n_trials=50,
    warmup_trials=5
)

# 3. Measure memory
memory = analyzer.measure_memory(
    model_params=count_parameters(model),
    param_dtype=np.float64
)

# 4. Build complete profile
profile = analyzer.build_profile(
    model_name='MyModel',
    training_time=timer.elapsed,
    inference_times=inference_times,
    model_params=count_parameters(model),
    latent_dim=20,
    n_forecast_steps=100,
    lag=5  # Model-specific kwargs
)

# 5. Save
analyzer.save_profile(profile, 'my_profile.json')
```

### Comparing Multiple Models

```python
from rectsim.runtime_analyzer import RuntimeAnalyzer

analyzer = RuntimeAnalyzer()

# Profile model A
profile_a = analyzer.build_profile(
    model_name='ModelA',
    training_time=train_time_a,
    inference_times=inference_times_a,
    model_params=params_a,
    latent_dim=20,
    n_forecast_steps=100
)

# Profile model B
profile_b = analyzer.build_profile(
    model_name='ModelB',
    training_time=train_time_b,
    inference_times=inference_times_b,
    model_params=params_b,
    latent_dim=20,
    n_forecast_steps=100
)

# Compare
comparison = analyzer.compare_models([profile_a, profile_b])
analyzer.save_comparison(comparison, 'comparison.json')

print(f"Training speedup: {comparison['training_time_ratio']['ModelB_vs_ModelA']:.2f}x")
print(f"Inference speedup: {comparison['inference_speedup']['ModelB_vs_ModelA']:.2f}x")
print(f"Winner (training): {comparison['winners']['fastest_training']}")
```

## Benchmarking Best Practices

### 1. **Warmup Trials**
Always use warmup trials (default: 5) to account for:
- JIT compilation
- Cache effects
- CPU frequency scaling

### 2. **Trial Count**
- Use `n_trials >= 50` for stable statistics
- Increase to 100+ for very fast operations (<1ms)
- Decrease for very slow operations (>1s)

### 3. **Step Count**
- Benchmark with realistic forecast horizons
- Default: 100 steps
- Match to your actual use case

### 4. **Environment**
- Run on representative hardware
- Close other applications
- Disable CPU power management if possible
- Use consistent Python environment

### 5. **Interpretation**
- Focus on mean times, not min/max
- Consider std deviation for stability
- Compare ratios, not absolute times
- Account for model capacity differences

## API Reference

### Classes

#### `RuntimeAnalyzer`
Main analysis class with methods for timing, benchmarking, and profiling.

**Methods**:
- `time_operation(name)`: Context manager for timing code blocks
- `benchmark_inference(forecast_fn, z0, n_steps, n_trials)`: Benchmark inference performance
- `measure_memory(model_params, param_dtype)`: Measure memory usage
- `compute_throughput(inference_timing, n_steps, latent_dim)`: Compute throughput metrics
- `analyze_complexity(model_name, model_params, latent_dim, **kwargs)`: Analyze computational complexity
- `build_profile(...)`: Build complete RuntimeProfile
- `compare_models(profiles)`: Generate comparative analysis
- `save_profile(profile, filepath)`: Save profile to JSON
- `save_comparison(comparison, filepath)`: Save comparison to JSON

#### `RuntimeProfile`
Complete runtime profile dataclass.

**Attributes**:
- `model_name`: Model identifier
- `training`: Training time results
- `inference_single_step`: Single-step timing
- `inference_full_trajectory`: Full trajectory timing
- `memory`: Memory usage
- `throughput`: Throughput metrics
- `complexity`: Complexity analysis

**Methods**:
- `to_dict()`: Convert to dictionary for JSON serialization

#### `TimingResult`
Timing measurement dataclass.

**Attributes**:
- `total_seconds`: Total time
- `mean_seconds`: Mean time
- `std_seconds`: Standard deviation
- `min_seconds`: Minimum time
- `max_seconds`: Maximum time
- `samples`: Number of samples

#### `MemoryResult`
Memory measurement dataclass.

**Attributes**:
- `peak_mb`: Peak memory usage (MB)
- `current_mb`: Current memory usage (MB)
- `model_parameters`: Total parameter count
- `parameter_memory_mb`: Memory for parameters (MB)

### Utility Functions

#### `compute_mvar_params(mvar_model)`
Count MVAR model parameters.

**Parameters**:
- `mvar_model`: Dict with 'A_0' and 'A_j' arrays

**Returns**:
- `n_params`: Total parameter count

#### `compute_lstm_params(lstm_model)`
Count LSTM model parameters.

**Parameters**:
- `lstm_model`: PyTorch LSTM model

**Returns**:
- `n_params`: Total parameter count

#### `quick_benchmark(...)`
Convenience function for quick single-model benchmarking.

**Returns**:
- `profile`: Complete RuntimeProfile

## Testing

Run the test script to verify installation:

```bash
python test_runtime_analyzer.py
```

This creates a mock MVAR model and profiles it, outputting results to `test_outputs/test_runtime_profile.json`.

## Performance Overhead

The runtime analyzer adds minimal overhead:
- Training timing: <0.1% overhead (single timing call)
- Inference benchmarking: Runs after training (no impact)
- Memory measurement: <1ms per call
- Profile saving: <10ms for typical profiles

## Limitations

1. **Memory Profiling**: Peak memory measurement is platform-dependent
2. **GPU Timing**: Currently optimized for CPU; GPU timing needs synchronization
3. **Multithreading**: Timings assume single-threaded execution
4. **Cache Effects**: First prediction may be slower (warmup mitigates this)

## Future Enhancements

- [ ] GPU-aware timing with CUDA events
- [ ] Energy consumption measurement (platform-dependent)
- [ ] Automatic scaling analysis (vary problem size)
- [ ] Statistical significance testing for comparisons
- [ ] Real-time monitoring dashboard
- [ ] Profiling for distributed/parallel training

## Related Modules

- `src/rectsim/mvar_trainer.py`: MVAR training (profiled by runtime analyzer)
- `rom/lstm_rom.py`: LSTM training (profiled by runtime analyzer)
- `visualizations/summary_json.py`: Integrates runtime profiles into summary JSON
- `ROM_pipeline.py`: Main pipeline with automatic profiling

## Contact

For issues or questions about runtime analysis, please refer to the main project repository.
