# Ensemble Simulation Pipeline

## Overview

The ensemble simulation pipeline generates multiple simulation runs with varied initial conditions (ICs) for building diverse training datasets for POD + MVAR models. This implements the data generation layer for the EF-ROM (Empirical Flow Reduced-Order Model) approach.

## Quick Start

### 1. Create an ensemble configuration

```yaml
# configs/my_ensemble.yaml
model: social_force

sim:
  N: 200
  T: 100.0
  dt: 0.01
  save_every: 10

params:
  alpha: 1.5
  beta: 0.5
  Cr: 2.0
  Ca: 1.0
  lr: 0.9
  la: 1.0

ensemble:
  n_runs: 20
  base_seed: 42
  ic_types: [gaussian, uniform, ring, cluster]
  ic_weights: [0.4, 0.3, 0.2, 0.1]

outputs:
  animate: true
  efrom:
    auto_run: false  # Don't run EF-ROM per run (do it later on ensemble)
```

### 2. Generate the ensemble

```bash
# On Oscar
conda activate wsindy
cd ~/src/wsindy-manifold
rectsim ensemble --config configs/my_ensemble.yaml
```

### 3. Output structure

```
simulations/
└── social_force_N200_T100_alpha1.5_beta0.5_Cr2_Ca1_lr0.9_la1/
    ├── run_0001/
    │   ├── traj.npz          # Positions, velocities, times
    │   ├── density.npz       # KDE density movie
    │   ├── metrics.csv       # Order parameters time series
    │   ├── run.json          # Run metadata
    │   ├── traj_final.png    # Final snapshot
    │   ├── order_params.png  # Order parameter plots
    │   ├── speed_hist.png    # Speed distribution
    │   ├── energy_time.png   # Kinetic energy evolution
    │   └── density_anim.mp4  # Density movie
    ├── run_0002/
    │   └── ...
    ├── ...
    └── ensemble_runs.csv     # Summary of all runs
```

## Initial Condition Types

### 1. **uniform**
- Particles uniformly distributed in the domain
- Good baseline for comparison
- Example: `ic: {type: uniform}`

### 2. **gaussian**
- Single Gaussian blob centered randomly
- Concentrated initial distribution
- Tests swarm cohesion dynamics
- Controlled by `sigma_factor` (default: 0.1 × min(Lx, Ly))

### 3. **ring**
- Particles arranged around a ring
- Tests rotational and collective motion patterns
- Controlled by `radius_factor` (default: 0.3 × min(Lx, Ly))

### 4. **cluster**
- Multiple Gaussian clusters (2-4 clusters)
- Tests multi-group interactions
- Cluster positions and proportions randomized
- Controlled by `n_clusters` and `sigma_factor`

## Ensemble Configuration Options

### Basic settings

```yaml
ensemble:
  n_runs: 20              # Number of simulation runs
  base_seed: 0            # Starting seed (run k uses base_seed + k)
```

### Explicit seeds (alternative)

```yaml
ensemble:
  seeds: [42, 43, 44, 45, 46]  # Explicit list overrides n_runs
```

### IC type sampling

```yaml
ensemble:
  ic_types: [gaussian, uniform, ring, cluster]
  ic_weights: [0.4, 0.3, 0.2, 0.1]  # Must sum to 1 (or will be normalized)
  # If ic_weights omitted, uses uniform distribution
```

## Model ID Generation

The pipeline automatically generates a unique model identifier from the configuration:

**Social force model:**
```
social_force_N200_T100_alpha1.5_beta0.5_Cr2_Ca1_lr0.9_la1
```

**Vicsek discrete model:**
```
vicsek_discrete_N400_T1000_v01_R1_sigma0.2
```

**With alignment:**
```
social_force_N200_T100_alpha1.5_beta0.5_Cr2_Ca1_lr0.9_la1_alignR1.5_alignRate0.1
```

The model ID is used as the directory name under `simulations/` to organize different parameter configurations.

## CLI Overrides

You can override config values from the command line:

```bash
# Override number of runs
rectsim ensemble --config configs/my_ensemble.yaml --ensemble.n_runs 50

# Override simulation parameters
rectsim ensemble --config configs/my_ensemble.yaml --sim.N 500 --sim.T 200.0

# Override force parameters
rectsim ensemble --config configs/my_ensemble.yaml --params.alpha 2.0 --params.Ca 0.5
```

## Oscar Workflow

### Full workflow

```bash
# 1. On Mac: Edit config and push to GitHub
cd ~/Desktop/wsindy-manifold
git add configs/my_ensemble.yaml
git commit -m "Add ensemble config"
git push origin main

# 2. Connect to Oscar
oscar-wsindy

# 3. Pull latest code
cd ~/src/wsindy-manifold
git pull origin main

# 4. Generate ensemble (login node for small runs)
rectsim ensemble --config configs/my_ensemble.yaml

# Or submit as SLURM job for large ensembles:
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=ensemble
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=batch

module load miniconda3/23.11.0s
conda activate wsindy

cd \$HOME/src/wsindy-manifold
rectsim ensemble --config configs/my_ensemble.yaml
EOF

# 5. Sync results back to Mac
# On Mac:
oscar-sync sim social_force_N200_T100_alpha1.5_beta0.5_Cr2_Ca1_lr0.9_la1
```

## Ensemble Summary CSV

The `ensemble_runs.csv` file contains metadata for all runs:

| run_id | seed | ic_type | final_polarization | final_speed | final_angular_momentum | final_dnn |
|--------|------|---------|-------------------|-------------|----------------------|-----------|
| run_0001 | 42 | gaussian | 0.8234 | 2.1234 | -0.0123 | 1.2345 |
| run_0002 | 43 | uniform | 0.7891 | 2.0987 | 0.0234 | 1.3456 |
| ... | ... | ... | ... | ... | ... | ... |

Use this to:
- Filter runs by IC type
- Identify interesting dynamics (high/low polarization)
- Select runs for POD training
- Analyze IC type effects on final state

## Next Steps (MVAR/LSTM Training)

After generating the ensemble:

1. **Compute global POD basis** across all runs
2. **Project density movies** to latent coordinates
3. **Train MVAR model** on latent time series
4. **Evaluate forecast horizon** and reconstruction accuracy

These steps will be implemented in Prompt 2.

## Troubleshooting

### Python version error
```
ERROR: Package 'rectsim' requires a different Python: 3.9.13 not in '>=3.10'
```
**Solution:** Use Python 3.10+ environment on Oscar:
```bash
module load miniconda3/23.11.0s
conda activate wsindy  # Should have Python 3.11
```

### Animation fails
If density movies fail to generate, ensure `imageio-ffmpeg` is installed:
```bash
pip install imageio-ffmpeg
```

### Memory issues
For large ensembles (N > 1000, T > 1000):
- Disable animations: `outputs.animate: false`
- Reduce grid resolution: `outputs.grid_density.nx: 64`
- Increase SLURM memory: `#SBATCH --mem=32G`

## References

- Alvarez et al. (2022) "Autoregressive data-driven method for long-time predictions" 
  - Inspired IC diversity strategy
- Bhaskar & Ziegelmeier (2019) "D'Orsogna swarm dynamics"
  - KDE-based density representation for POD compression
