# LSTM ROM Integration Guide

## Overview

This guide explains how to integrate the LSTM ROM into the existing evaluation pipeline, reusing all MVAR visualization and metrics code.

**Key Design Principle:** MVAR and LSTM use the **same evaluation function** with different forecast functions and output folders.

## Architecture

```
                    ┌─────────────────────┐
                    │   POD Reduction     │
                    │   (Shared by both)  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Latent Trajectories│
                    │     [K, d]          │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ build_latent_dataset│
                    │  X_all, Y_all       │
                    └──────────┬──────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
     ┌──────────▼─────────┐       ┌──────────▼─────────┐
     │   MVAR Training    │       │   LSTM Training    │
     │  (Ridge Regr.)     │       │  (train_lstm_rom)  │
     └──────────┬─────────┘       └──────────┬─────────┘
                │                             │
     ┌──────────▼─────────┐       ┌──────────▼─────────┐
     │  mvar_forecast_fn  │       │ lstm_forecast_fn   │
     │  (latent -> latent)│       │ (latent -> latent) │
     └──────────┬─────────┘       └──────────┬─────────┘
                │                             │
                └──────────────┬──────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   evaluate_rom()    │
                    │   (Generic Eval)    │
                    └──────────┬──────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
     ┌──────────▼─────────┐       ┌──────────▼─────────┐
     │  results/MVAR/     │       │  results/LSTM/     │
     │  - r2_MVAR.csv     │       │  - r2_LSTM.csv     │
     │  - plots_MVAR.png  │       │  - plots_LSTM.png  │
     └────────────────────┘       └────────────────────┘
```

## Implementation Steps

### Step 1: Folder Structure

Create separate output folders for each model:

```python
base_dir = os.path.join("results", config.outputs.run_name)

# Shared POD basis and latent data
rom_common_dir = os.path.join(base_dir, "rom_common")

# Model-specific outputs
mvar_dir = os.path.join(base_dir, "MVAR")
lstm_dir = os.path.join(base_dir, "LSTM")

os.makedirs(rom_common_dir, exist_ok=True)
os.makedirs(mvar_dir, exist_ok=True)
os.makedirs(lstm_dir, exist_ok=True)
```

### Step 2: Generic Evaluation Function

Refactor existing MVAR evaluation to be model-agnostic:

```python
def evaluate_rom(
    model_name: str,
    forecast_next_latent_sequence_fn: callable,
    config: dict,
    R: callable,  # Restriction: density -> latent
    L: callable,  # Lifting: latent -> density
    test_trajectories: list,
    out_dir: str
):
    """
    Generic ROM evaluation and visualization.
    
    This function works for BOTH MVAR and LSTM by accepting a
    model-agnostic forecast function.
    
    Parameters
    ----------
    model_name : str
        "MVAR" or "LSTM" (used for labeling and filenames).
    forecast_next_latent_sequence_fn : callable
        A function with signature:
            forecast_fn(y_init_window, n_steps) -> ys_pred
        where:
            y_init_window : np.ndarray [lag, d]
            n_steps : int
            ys_pred : np.ndarray [n_steps, d]
    config : dict or config object
        Full configuration.
    R : callable
        Restriction operator: density [nx, ny] -> latent [d].
    L : callable
        Lifting operator: latent [d] -> density [nx, ny].
    test_trajectories : list
        List of test density trajectories (unseen ICs).
    out_dir : str
        Output directory for this model's results.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Get evaluation config
    lag = config.rom.models[model_name.lower()].lag
    forecast_start = config.evaluation.forecast_start
    forecast_end = config.evaluation.forecast_end
    dt = config.simulation.dt
    
    # Convert times to indices
    k_start = int(forecast_start / dt)
    k_end = int(forecast_end / dt)
    n_steps = k_end - k_start
    
    # Storage for metrics
    all_r2_vs_time = []
    
    for traj_idx, X_truth in enumerate(test_trajectories):
        # X_truth: [K, nx, ny] density snapshots
        
        # Restrict to latent
        Y_truth = np.array([R(X_truth[k]) for k in range(len(X_truth))])
        
        # Initial window from truth
        y_init_window = Y_truth[k_start - lag : k_start]  # [lag, d]
        
        # Forecast in latent space
        ys_pred = forecast_next_latent_sequence_fn(y_init_window, n_steps)
        
        # Lift to density space
        xs_pred = np.array([L(ys_pred[i]) for i in range(n_steps)])
        
        # Get ground truth densities
        xs_truth = X_truth[k_start : k_end]
        
        # Compute R² vs time
        r2_vs_time = compute_r2_trajectory(xs_pred, xs_truth)
        all_r2_vs_time.append(r2_vs_time)
    
    # Aggregate across trajectories
    r2_mean = np.mean(all_r2_vs_time, axis=0)
    r2_std = np.std(all_r2_vs_time, axis=0)
    
    # Save to CSV with model name in filename
    csv_path = os.path.join(out_dir, f"r2_vs_time_{model_name}.csv")
    save_r2_csv(csv_path, r2_mean, r2_std)
    
    # Create plots with model name in filename
    plot_path = os.path.join(out_dir, f"r2_vs_time_{model_name}.png")
    plot_r2_vs_time(plot_path, r2_mean, r2_std, model_name=model_name)
    
    print(f"\n{model_name} ROM evaluation complete:")
    print(f"  Mean R²: {r2_mean.mean():.4f}")
    print(f"  Results saved to: {out_dir}")
```

### Step 3: MVAR Forecast Function

Wrap existing MVAR forecaster:

```python
def create_mvar_forecast_fn(A, b, lag):
    """
    Create MVAR forecast function.
    
    Parameters
    ----------
    A : np.ndarray [d, lag*d]
        MVAR coefficient matrix.
    b : np.ndarray [d]
        MVAR bias vector.
    lag : int
        MVAR lag order.
    
    Returns
    -------
    forecast_fn : callable
        Function with signature forecast_fn(y_init_window, n_steps).
    """
    def mvar_forecast_fn(y_init_window, n_steps):
        """MVAR closed-loop forecast."""
        d = y_init_window.shape[-1]
        ys_pred = []
        
        # Initialize window
        y_window = y_init_window.copy()  # [lag, d]
        
        for _ in range(n_steps):
            # Flatten window: [lag, d] -> [lag*d]
            y_flat = y_window.flatten()
            
            # Predict next state: y_next = A @ y_flat + b
            y_next = A @ y_flat + b  # [d]
            
            ys_pred.append(y_next)
            
            # Update window: drop oldest, append newest
            y_window = np.vstack([y_window[1:], y_next])
        
        return np.array(ys_pred)  # [n_steps, d]
    
    return mvar_forecast_fn
```

### Step 4: LSTM Forecast Function

Use the factory function from `lstm_rom.py`:

```python
from src.rom.lstm_rom import LatentLSTMROM, lstm_forecast_fn_factory

# Load trained LSTM model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LatentLSTMROM(
    d=latent_dim,
    hidden_units=config.rom.models.lstm.hidden_units,
    num_layers=config.rom.models.lstm.num_layers
)
model.load_state_dict(torch.load(lstm_model_path, map_location=device))
model.to(device)
model.eval()

# Create forecast function
lstm_forecast_fn = lstm_forecast_fn_factory(model)
```

### Step 5: Main Driver

Integrate both models in the main pipeline:

```python
def ROM_pipeline(config):
    """
    Run unified ROM pipeline with both MVAR and LSTM.
    """
    # Setup directories
    base_dir = os.path.join("results", config.outputs.run_name)
    rom_common_dir = os.path.join(base_dir, "rom_common")
    mvar_dir = os.path.join(base_dir, "MVAR")
    lstm_dir = os.path.join(base_dir, "LSTM")
    
    os.makedirs(rom_common_dir, exist_ok=True)
    os.makedirs(mvar_dir, exist_ok=True)
    os.makedirs(lstm_dir, exist_ok=True)
    
    # 1. Run simulations
    print("Running training simulations...")
    train_trajectories = run_simulations(config, "train")
    
    print("Running test simulations...")
    test_trajectories = run_simulations(config, "test")
    
    # 2. Build POD (shared by both models)
    print("\nBuilding POD basis...")
    Phi, singular_values, rho_mean = build_pod_basis(
        train_trajectories, 
        config.rom.pod_energy
    )
    
    # Save POD to common directory
    np.save(os.path.join(rom_common_dir, "POD_basis.npy"), Phi)
    np.save(os.path.join(rom_common_dir, "singular_values.npy"), singular_values)
    np.save(os.path.join(rom_common_dir, "mean_density.npy"), rho_mean)
    
    # Define R and L operators
    def R(x):
        """Restriction: density -> latent."""
        return Phi.T @ (x.flatten() - rho_mean)
    
    def L(y):
        """Lifting: latent -> density."""
        rho_flat = Phi @ y + rho_mean
        return rho_flat.reshape(config.simulation.nx, config.simulation.ny)
    
    # 3. Build shared dataset
    print("\nBuilding latent dataset...")
    y_trajs = []
    for X_c in train_trajectories:
        Y_c = np.array([R(X_c[k]) for k in range(len(X_c))])
        y_trajs.append(Y_c)
    
    # 4. Train MVAR (if enabled)
    if config.rom.models.mvar.enabled:
        print("\n" + "="*80)
        print("TRAINING MVAR ROM")
        print("="*80)
        
        from src.rectsim.rom_data_utils import build_latent_dataset
        
        lag = config.rom.models.mvar.lag
        X_all, Y_all = build_latent_dataset(y_trajs, lag)
        
        # Train MVAR
        A, b = train_mvar_ridge(X_all, Y_all, config.rom.models.mvar.ridge_alpha)
        
        # Save MVAR parameters
        np.save(os.path.join(mvar_dir, "mvar_A.npy"), A)
        np.save(os.path.join(mvar_dir, "mvar_b.npy"), b)
        
        # Create forecast function
        mvar_forecast_fn = create_mvar_forecast_fn(A, b, lag)
        
        # Evaluate
        print("\nEvaluating MVAR...")
        evaluate_rom(
            model_name="MVAR",
            forecast_next_latent_sequence_fn=mvar_forecast_fn,
            config=config,
            R=R, L=L,
            test_trajectories=test_trajectories,
            out_dir=mvar_dir
        )
    
    # 5. Train LSTM (if enabled)
    if config.rom.models.lstm.enabled:
        print("\n" + "="*80)
        print("TRAINING LSTM ROM")
        print("="*80)
        
        from src.rectsim.rom_data_utils import build_latent_dataset
        from src.rom.lstm_rom import train_lstm_rom, lstm_forecast_fn_factory
        
        lag = config.rom.models.lstm.lag
        X_all, Y_all = build_latent_dataset(y_trajs, lag)
        
        # Train LSTM
        lstm_model_path, val_loss = train_lstm_rom(X_all, Y_all, config, lstm_dir)
        
        print(f"\nLSTM training complete. Val loss: {val_loss:.6f}")
        
        # Load trained model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LatentLSTMROM(
            d=X_all.shape[-1],
            hidden_units=config.rom.models.lstm.hidden_units,
            num_layers=config.rom.models.lstm.num_layers
        )
        model.load_state_dict(torch.load(lstm_model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Create forecast function
        lstm_forecast_fn = lstm_forecast_fn_factory(model)
        
        # Evaluate
        print("\nEvaluating LSTM...")
        evaluate_rom(
            model_name="LSTM",
            forecast_next_latent_sequence_fn=lstm_forecast_fn,
            config=config,
            R=R, L=L,
            test_trajectories=test_trajectories,
            out_dir=lstm_dir
        )
    
    print("\n" + "="*80)
    print("UNIFIED ROM PIPELINE COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {base_dir}")
    print(f"  - Shared POD: {rom_common_dir}")
    if config.rom.models.mvar.enabled:
        print(f"  - MVAR results: {mvar_dir}")
    if config.rom.models.lstm.enabled:
        print(f"  - LSTM results: {lstm_dir}")
```

## Output Structure

```
results/
  <run_name>/
    rom_common/
      POD_basis.npy           # Shared POD basis Phi
      singular_values.npy     # Shared singular values
      mean_density.npy        # Shared mean density
    
    MVAR/
      mvar_A.npy              # MVAR coefficient matrix
      mvar_b.npy              # MVAR bias vector
      r2_vs_time_MVAR.csv     # R² vs time (mean, std)
      r2_vs_time_MVAR.png     # R² plot
      reconstruction_MVAR.mp4 # Optional: video
    
    LSTM/
      lstm_state_dict.pt      # Trained LSTM weights
      training_log.csv        # Training history
      r2_vs_time_LSTM.csv     # R² vs time (mean, std)
      r2_vs_time_LSTM.png     # R² plot
      reconstruction_LSTM.mp4 # Optional: video
```

## Key Benefits

1. **Code Reuse**: Same evaluation function for both models
2. **Fair Comparison**: Same POD basis, same test data, same metrics
3. **Parallel Outputs**: Easy to compare MVAR vs LSTM results
4. **Consistent Naming**: All files have model suffix (MVAR/LSTM)
5. **Extensibility**: Easy to add new models (just implement forecast_fn)

## Testing

Run the test suite to verify integration:

```bash
# Test LSTM forecasting
python test_lstm_forecasting.py

# Test full pipeline integration
python test_unified_rom_pipeline.py  # (to be created)
```

## Configuration

Ensure your YAML config has both model sections:

```yaml
rom:
  subsample: 1
  pod_energy: 0.99
  
  models:
    mvar:
      enabled: true
      lag: 20
      ridge_alpha: 1.0e-6
    
    lstm:
      enabled: true
      lag: 20
      hidden_units: 64
      num_layers: 2
      batch_size: 32
      learning_rate: 0.0001
      weight_decay: 0.0001
      max_epochs: 100
      patience: 10
      gradient_clip: 1.0
```

## Next Steps

1. Modify existing `run_unified_mvar_pipeline.py` to use new structure
2. Update evaluation functions to accept `model_name` parameter
3. Test with production configs
4. Generate comparative plots (MVAR vs LSTM R² curves on same axes)
