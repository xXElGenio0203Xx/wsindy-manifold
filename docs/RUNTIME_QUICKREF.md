# Runtime Analysis Quick Reference

## 🎯 Quick Start

### Automatic (Unified Pipeline)
```bash
python ROM_pipeline.py \
    --config configs/my_config.yaml \
    --experiment_name my_experiment
```

Runtime analysis is **automatically** included in:
- `oscar_output/my_experiment/MVAR/runtime_profile.json`
- `oscar_output/my_experiment/LSTM/runtime_profile.json`
- `oscar_output/my_experiment/runtime_comparison.json`
- `oscar_output/my_experiment/summary.json` (includes runtime section)

### Manual (Custom Scripts)
```python
from rectsim.runtime_analyzer import quick_benchmark

profile = quick_benchmark(
    model_name='MVAR',
    forecast_fn=my_forecast_function,
    z0=initial_state,
    training_time=12.5,
    model_params=2020,
    latent_dim=20,
    lag=5
)
```

## 📊 Key Metrics

### Training Time
- **What**: Total time to train model
- **Units**: seconds
- **Location**: `profile.training.total_seconds`

### Inference Time (Single Step)
- **What**: Time to predict one step ahead
- **Units**: milliseconds (typically)
- **Location**: `profile.inference_single_step.mean_seconds`

### Inference Time (Full Trajectory)
- **What**: Time to forecast 100 steps
- **Units**: seconds
- **Location**: `profile.inference_full_trajectory.mean_seconds`

### Throughput
- **What**: Predictions per second
- **Location**: `profile.throughput.predictions_per_second`

### Memory
- **What**: Model parameter memory footprint
- **Units**: MB
- **Location**: `profile.memory.parameter_memory_mb`

### Parameters
- **What**: Total trainable parameters
- **Location**: `profile.memory.model_parameters`

## 🔍 Reading Results

### Example MVAR Profile
```json
{
  "model_name": "MVAR",
  "training": {"total_seconds": 8.3},
  "inference": {
    "single_step": {"mean_seconds": 0.00018},
    "full_trajectory": {"mean_seconds": 0.018}
  },
  "throughput": {"steps_per_second": 5555},
  "memory": {"model_parameters": 2020}
}
```

**Interpretation**:
- Training took 8.3 seconds
- Each prediction takes ~0.18ms
- Can forecast ~5,555 steps per second
- Model has 2,020 parameters

### Example Comparison
```json
{
  "training_time_ratio": {"LSTM_vs_MVAR": 3.2},
  "inference_speedup": {"LSTM_vs_MVAR": 0.85},
  "winners": {
    "fastest_training": "MVAR",
    "fastest_inference": "MVAR"
  }
}
```

**Interpretation**:
- LSTM takes 3.2× longer to train
- LSTM is 0.85× as fast (15% slower) for inference
- MVAR wins on both speed metrics

## 📈 Common Comparisons

### MVAR vs LSTM: Typical Results

| Metric | MVAR | LSTM | Winner |
|--------|------|------|--------|
| Training Speed | ✅ Fast | ❌ Slow (2-5×) | MVAR |
| Inference Speed | ✅ Fast | ⚠️ Similar | MVAR |
| Memory | ✅ Small | ❌ Large (2-3×) | MVAR |
| Accuracy (R²) | ⚠️ Linear | ✅ Nonlinear | Depends |
| Interpretability | ✅ High | ❌ Low | MVAR |

### When LSTM Wins
- **Highly nonlinear dynamics** (R² > 0.95 vs MVAR < 0.90)
- **Long-term dependencies** (beyond MVAR lag window)
- **GPU available** (LSTM inference can be faster on GPU)

### When MVAR Wins
- **Near-linear dynamics** (MVAR R² > 0.90)
- **Fast inference required** (CPU deployment)
- **Interpretability needed** (coefficient matrices)
- **Small memory budget** (embedded systems)

## 🛠️ Customization

### Change Benchmark Parameters
Edit `ROM_pipeline.py`:
```python
# Around line 300
benchmark_steps = 200  # Default: 100
benchmark_trials = 100  # Default: 50
```

### Add Custom Metrics
```python
from rectsim.runtime_analyzer import RuntimeAnalyzer

analyzer = RuntimeAnalyzer()

# Your custom timing
with analyzer.time_operation('custom_preprocessing') as timer:
    result = my_custom_function()

print(f"Custom operation took {timer.elapsed:.2f}s")
```

## 📁 Output Files

```
oscar_output/
└── my_experiment/
    ├── summary.json                    # Includes runtime_analysis section
    ├── runtime_comparison.json         # MVAR vs LSTM comparison
    ├── MVAR/
    │   ├── runtime_profile.json        # MVAR detailed profile
    │   └── ...
    └── LSTM/
        ├── runtime_profile.json        # LSTM detailed profile
        └── ...
```

## 🚨 Troubleshooting

### Import Error
```python
ModuleNotFoundError: No module named 'psutil'
```
**Fix**: `pip install psutil`

### Memory Profile Shows Zero
- Platform limitation (Windows/macOS differences)
- Use `parameter_memory_mb` instead (always available)

### Inconsistent Timings
- Disable CPU power management
- Close background applications
- Increase `n_trials` (default: 50)
- Check for competing processes

### GPU Not Used
- LSTM training uses GPU automatically if available
- Benchmarking runs on CPU by default
- Modify forecast function to use GPU tensors

## 📚 See Also

- **Full Documentation**: `docs/RUNTIME_ANALYSIS.md`
- **Test Script**: `test_runtime_analyzer.py`
- **Source Code**: `src/rectsim/runtime_analyzer.py`
- **Pipeline Integration**: `ROM_pipeline.py`
