# MVAR-ROM Training Strategy Update

## Change Summary

**Previous approach:**
- Split data into training (80-95%) and test sets
- Fit POD on training data only
- Fit MVAR on training latent representation
- Forecast on test set (last 5-20% of frames)
- Evaluate on test set

**New approach:**
- Train POD and MVAR on **entire simulation** (t=0 to t=T)
- Forecast from t=0 using first w frames as seed
- Evaluate by comparing forecast to ground truth over **full trajectory**
- No train/test split

## Rationale

This change measures the model's ability to:
1. **Learn the dynamics** - Train on all available data to get best possible model
2. **Reproduce learned dynamics** - Forecast from initial conditions and compare to ground truth
3. **Identify model limitations** - See where/when the linear MVAR approximation breaks down

This is more like:
- **Interpolation performance** - How well does MVAR capture the dynamics it was trained on?
- **Model quality assessment** - Does the learned linear model reproduce the full nonlinear trajectory?

Rather than extrapolation to unseen data (old approach).

## Key Changes in `run_mvar_rom_production.py`

### Removed Parameters
- `--train-frac` argument (no longer needed)
- `train_frac` field in `MVARROMConfig` dataclass

### Modified Workflow

**Step 1: POD**
```python
# OLD: Ud, xbar, d, energy_curve = fit_pod(X_train, ...)
# NEW:
Ud, xbar, d, energy_curve = fit_pod(X, ...)  # Full dataset
```

**Step 2: Latent projection**
```python
# OLD: Y_train = restrict(X_train, ...) and Y_test = restrict(X_test, ...)
# NEW:
Y = restrict(X, ...)  # Full latent representation
```

**Step 3: MVAR training**
```python
# OLD: mvar_model = fit_mvar(Y_train, ...)
# NEW:
mvar_model = fit_mvar(Y, ...)  # Train on all latent frames
```

**Step 4: Forecasting**
```python
# OLD: Y_seed = Y_train[-w:], rollout for T1 (test) steps
# NEW:
Y_seed = Y[:w]  # First w frames as seed
Y_forecast = rollout(Y_seed, steps=T, ...)  # Forecast full trajectory
```

**Step 5: Evaluation**
```python
# OLD: Compare X_forecast to X_test
# NEW:
Compare X_forecast to X  # Full ground truth
```

### Updated Outputs

**Metrics now reflect:**
- How well MVAR reproduces the dynamics from t=0 to t=T
- Error accumulation over the full simulation length
- Model's ability to stay on the manifold learned during training

**Visualizations updated:**
- Error timeseries show seed frames (first w frames) instead of train/test split
- Snapshots compare full trajectory
- Latent scatter plots show all frames

## Usage

```bash
# Basic usage (no train-frac argument)
python scripts/run_mvar_rom_production.py simulations/my_sim__latest \
    --pod-energy 0.99 \
    --mvar-order 9 \
    --ridge 1e-3

# Old usage (train-frac removed)
# python scripts/run_mvar_rom_production.py ... --train-frac 0.95  # ❌ No longer supported
```

## Expected Results

- **R² should be higher** - Model trained on data it's being evaluated on
- **Errors show accumulation** - Linear approximation diverges from nonlinear truth over time
- **Tolerance horizon** - Still meaningful: measures how long MVAR stays accurate
- **Physics insights** - See where linear dynamics fail (e.g., sharp turns, collisions)

## Future Directions

1. **True generalization testing**: Train on one simulation, test on another with different parameters
2. **Multi-step training**: Optimize for N-step ahead prediction instead of 1-step
3. **Hybrid models**: Switch between MVAR and full simulation when error exceeds threshold
4. **Adaptive models**: Update MVAR coefficients online as simulation progresses
