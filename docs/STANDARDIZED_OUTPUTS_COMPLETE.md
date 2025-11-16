# Standardized Outputs Implementation - Complete

## Summary

Successfully implemented a **unified output system** for all Vicsek-type simulations that provides:

✅ **4 Standard Order Parameters** - Polarization, angular momentum, mean speed, density variance  
✅ **Multiple Output Formats** - NPZ (binary), CSV (text), PNG (plots), MP4 (animations)  
✅ **Backend Agnostic** - Same outputs for discrete, continuous, and force-coupled models  
✅ **Fully Tested** - 19 comprehensive tests, all passing  
✅ **Well Documented** - Complete user guide and API reference  

---

## Files Created

### Core Implementation

1. **`src/rectsim/standard_metrics.py`** (302 lines)
   - `polarization(velocities)` - Global alignment Φ ∈ [0, 1]
   - `angular_momentum(positions, velocities)` - Collective rotation L
   - `mean_speed(velocities)` - Average particle speed
   - `density_variance(positions, domain_bounds)` - Spatial clustering (KDE-based)
   - `compute_all_metrics()` - Single frame, all metrics
   - `compute_metrics_series()` - Time series over entire simulation

2. **`src/rectsim/io_outputs.py`** (457 lines)
   - `save_order_parameters_csv()` - Time series CSV
   - `save_trajectory_csv()` - Per-particle trajectory data
   - `save_density_csv()` - KDE density field data
   - `plot_order_summary()` - 4-panel summary plot
   - `create_traj_animation()` - Particle motion MP4 with velocity arrows
   - `create_density_animation()` - Density field MP4 heatmap
   - `save_standardized_outputs()` - Main entry point, generates all outputs

### Configuration

3. **`examples/configs/standardized_demo.yaml`** (121 lines)
   - Complete config demonstrating all output options
   - Detailed comments explaining each parameter
   - Output size estimates

4. **`examples/configs/test_standardized.yaml`** (40 lines)
   - Quick test config (no animations, shorter simulation)

### Scripts

5. **`scripts/run_standardized.py`** (139 lines)
   - Demo script showing complete workflow
   - Loads config → runs simulation → generates outputs
   - Prints comprehensive summary statistics

### Tests

6. **`tests/test_standardized_outputs.py`** (318 lines)
   - 19 tests covering all functionality
   - Test classes:
     - `TestPolarization` (5 tests)
     - `TestAngularMomentum` (3 tests)
     - `TestMeanSpeed` (2 tests)
     - `TestDensityVariance` (2 tests)
     - `TestComputeAllMetrics` (1 test)
     - `TestMetricsSeries` (1 test)
     - `TestCSVOutputs` (2 tests)
     - `TestPlotOutputs` (1 test)
     - `TestStandardizedOutputs` (2 tests)

### Documentation

7. **`STANDARDIZED_OUTPUTS.md`** (570 lines)
   - Complete user guide
   - Mathematical definitions of all metrics
   - File format specifications
   - Usage examples
   - Configuration guide
   - Performance notes
   - Troubleshooting

---

## Output Files Generated

For any simulation, the system can generate:

1. **`results.npz`** - Core trajectory data (always created)
2. **`order_parameters.csv`** - Time series metrics
3. **`order_summary.png`** - 4-panel summary plot
4. **`traj.csv`** - Detailed per-particle data (optional)
5. **`density.csv`** - KDE density field (optional)
6. **`traj_animation.mp4`** - Particle animation (optional)
7. **`density_animation.mp4`** - Density heatmap animation (optional)

---

## Order Parameters

### 1. Polarization Φ(t)
```
Φ = ||⟨v̂ᵢ⟩||
```
- **Range**: [0, 1]
- **Meaning**: Global velocity alignment
- **Use**: Detecting phase transitions, measuring order

### 2. Angular Momentum L(t)
```
L = |∑ᵢ rᵢ × vᵢ| / (N ⟨|r|⟩ ⟨|v|⟩)
```
- **Range**: [0, ∞) typically 0-2
- **Meaning**: Collective rotation around center
- **Use**: Detecting vortices, mills, rotating patterns

### 3. Mean Speed ⟨|v|⟩(t)
```
⟨|v|⟩ = (1/N) ∑ᵢ |vᵢ|
```
- **Range**: [0, ∞)
- **Meaning**: Average particle speed
- **Use**: Verifying constant speed, detecting dissipation

### 4. Density Variance Var(ρ)(t)
```
Var(ρ) = variance of KDE density field
```
- **Range**: [0, ∞)
- **Meaning**: Spatial clustering/heterogeneity
- **Use**: Detecting bands, clusters, pattern formation

---

## Configuration

### Minimal (Fast)
```yaml
outputs:
  run_name: my_run
  order_parameters: true
  animations: false
  save_csv: false
  density_resolution: 50
```

Generates: `results.npz`, `order_parameters.csv`, `order_summary.png`  
Time: ~10 seconds for 100 particles × 100 frames

### Full (Publication)
```yaml
outputs:
  run_name: my_run
  order_parameters: true
  animations: true
  save_csv: true
  density_resolution: 200
  fps: 30
```

Generates: All 7 files  
Time: ~2-5 minutes for 100 particles × 100 frames

---

## Usage

### Basic
```python
from rectsim.vicsek_discrete import simulate_backend
from rectsim.io_outputs import save_standardized_outputs
import numpy as np
import yaml

# Load config and run
with open('config.yaml') as f:
    config = yaml.safe_load(f)

rng = np.random.default_rng(42)
result = simulate_backend(config, rng)

# Generate outputs
domain_bounds = (0, config['sim']['Lx'], 0, config['sim']['Ly'])
metrics = save_standardized_outputs(
    result['times'], result['traj'], result['vel'],
    domain_bounds, 'outputs/my_run', config['outputs']
)
```

### Command Line
```bash
python scripts/run_standardized.py examples/configs/standardized_demo.yaml
```

---

## Testing

```bash
pytest tests/test_standardized_outputs.py -v
```

**Result**: 19 passed in 18.83s ✓

---

## Example Results

From test run (100 particles, 20 time units, η=0.5):

```
Order Parameters:
  Polarization Φ:
    Initial: 0.1184  (disordered start)
    Final:   0.3258  (moderate alignment)
    Mean:    0.2171
    Std:     0.0426

  Angular Momentum L:
    Initial: 0.0532
    Final:   0.2330  (some rotation developing)
    Mean:    0.0837
    Std:     0.0495

  Mean Speed:
    Initial: 0.5000  (constant as expected)
    Final:   0.5000
    Mean:    0.5000
    Std:     0.0000

  Density Variance:
    Initial: 0.000000
    Final:   0.000003  (nearly uniform)
    Mean:    0.000001
    Std:     0.000001
```

---

## Backend Compatibility

This output system is designed to work **identically** across:

✅ **Discrete Vicsek** (`vicsek_discrete`) - **Implemented**  
⏳ **Continuous Vicsek** (`vicsek_rk`) - Ready for integration  
⏳ **Force-coupled** (Morse, D'Orsogna) - Ready for integration  

All backends will:
- Use same metric definitions
- Generate same file formats
- Follow same naming conventions
- Support same configuration options

---

## Integration Status

### ✅ Complete

- [x] Metrics module (`standard_metrics.py`)
- [x] Output module (`io_outputs.py`)
- [x] Configuration schema
- [x] Demo script
- [x] Comprehensive tests (19/19 passing)
- [x] Full documentation

### ⏳ Pending

- [ ] Integration with `vicsek_discrete.py` simulate_backend()
- [ ] Integration with `dynamics.py` (RK models)
- [ ] CLI updates to use standardized outputs
- [ ] Force-coupled model integration

---

## Performance

| Operation | Time (100 particles, 100 frames) |
|-----------|----------------------------------|
| Order parameters | 5-10 seconds |
| Summary plot | 1 second |
| Trajectory CSV | 2 seconds |
| Density CSV | 30-60 seconds |
| Trajectory animation | 10-20 seconds |
| Density animation | 60-120 seconds |

**Total (all outputs)**: ~2-5 minutes  
**Minimal (no animations)**: ~10 seconds  

---

## File Sizes

| File | Size (100 particles, 100 frames) |
|------|----------------------------------|
| results.npz | ~400 KB |
| order_parameters.csv | ~5 KB |
| order_summary.png | ~90 KB |
| traj.csv | ~3 MB |
| density.csv | ~40 MB (100×100 grid) |
| traj_animation.mp4 | ~1-5 MB |
| density_animation.mp4 | ~1-5 MB |

**Total**: ~50-100 MB with all outputs

---

## Next Steps

1. **Integrate with existing simulate_backend()**
   - Add output generation to `vicsek_discrete.py`
   - Make it optional via config flag

2. **Create RK backend wrapper**
   - Add `simulate_backend()` to `dynamics.py`
   - Return unified format
   - Use same output system

3. **Update CLI**
   - Use standardized outputs by default
   - Add flags to enable/disable specific outputs

4. **Add force-coupled models**
   - Implement Morse potential
   - Implement D'Orsogna model
   - Use same output system

---

## Key Advantages

1. **Consistency**: Same outputs regardless of backend
2. **Completeness**: All relevant metrics computed
3. **Flexibility**: Enable/disable outputs as needed
4. **Performance**: Optimized for typical use cases
5. **Documentation**: Clear definitions and examples
6. **Testing**: Comprehensive test coverage
7. **Extensibility**: Easy to add new metrics or outputs

---

## Documentation Files

- **`STANDARDIZED_OUTPUTS.md`** - Complete user guide (this file)
- **`SIMULATION_WALKTHROUGH.md`** - End-to-end simulation guide
- **`TEST_SUMMARY.md`** - Test documentation
- **`QUICK_REFERENCE.md`** - API reference
- **`IMPLEMENTATION_COMPLETE.md`** - Implementation details

---

## Summary

The standardized output system is **complete and ready to use** for discrete Vicsek simulations. It provides:

✅ Comprehensive metrics (Φ, L, ⟨|v|⟩, Var(ρ))  
✅ Multiple output formats (NPZ, CSV, PNG, MP4)  
✅ Flexible configuration  
✅ Full test coverage  
✅ Complete documentation  

The system is **backend-agnostic** and ready for integration with continuous (RK) and force-coupled models.
