# ROM Pipeline Quick Reference

## Complete Workflow (One Command Per Stage)

```bash
# Stage 1: Generate ensemble
rectsim ensemble --config configs/rom_test.yaml

# Stage 2: Build POD basis
python scripts/rom_build_pod.py \
  --experiment_name my_exp \
  --sim_root simulations/MODEL_ID/runs \
  --train_runs 0 1 2 3 4 5 6 7 \
  --test_runs 8 9

# Stage 3: Train MVAR
python scripts/rom_train_mvar.py \
  --experiment_name my_exp \
  --mvar_order 4 \
  --ridge 1e-6

# Stage 4: Evaluate
python scripts/rom_evaluate.py \
  --experiment_name my_exp \
  --sim_root simulations/MODEL_ID/runs \
  --no_videos  # Optional: faster without videos
```

## Oscar HPC (One Command)

```bash
# Edit parameters in scripts/slurm/job_*.sh, then:
bash scripts/slurm/submit_pipeline.sh

# Monitor
squeue -u $USER
tail -f logs/ensemble_JOBID.out
```

## Output Locations

```
rom/my_exp/
├── pod/basis.npz           # POD modes + metadata
├── latent/run_*.npz        # Latent trajectories  
├── mvar/mvar_model.npz     # MVAR coefficients
└── mvar/forecast/
    ├── metrics_run_*.json  # Per-run metrics
    └── aggregate_metrics.json  # Summary
```

## Key Metrics

```bash
# Aggregate performance
cat rom/my_exp/mvar/forecast/aggregate_metrics.json | grep -E "mean_r2|mean_rmse|all_mass_ok"

# Per-run breakdown
for f in rom/my_exp/mvar/forecast/metrics_run_*.json; do
  echo "$f: R²=$(jq .r2 $f)"
done
```

## Common Parameter Adjustments

```bash
# More POD modes (better accuracy)
--energy_threshold 0.998

# Fewer POD modes (faster)
--energy_threshold 0.95

# Higher MVAR order (longer memory)
--mvar_order 8

# Stronger regularization (more stable)
--ridge 1e-5

# More training data (less test)
--train_frac 0.9
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Low R² | Increase `--energy_threshold` or `--mvar_order` |
| NaN forecasts | Increase `--ridge` or decrease `--mvar_order` |
| Out of memory | Reduce `nx, ny` in config or use `--no_videos` |
| Mass not conserved | Expected with truncated POD, use more modes |

## File Formats

**basis.npz:**
- `Phi`: (d, r) POD modes
- `S`: Singular values
- `mean`: (d,) Global mean
- `energy`: Cumulative energy fractions

**run_XXXX_latent.npz:**
- `Y`: (T, r) Latent coefficients
- `times`: (T,) Time stamps

**forecast_run_XXXX.npz:**
- `density_true`: (T_forecast, ny, nx)
- `density_pred`: (T_forecast, ny, nx)
- `errors_e2`: (T_forecast,) L2 errors
- `rmse`: (T_forecast,) Root mean square errors
- `mass_error`: (T_forecast,) Mass drift

## Directory Structure Reference

```
project/
├── simulations/EXPERIMENT/runs/  # Raw simulations
├── rom/EXPERIMENT/               # ROM outputs
│   ├── pod/                      # POD basis
│   ├── latent/                   # Latent trajectories
│   └── mvar/forecast/            # Evaluation results
├── configs/*.yaml                # Configurations
└── scripts/
    ├── rom_*.py                  # Python scripts
    └── slurm/*.sh                # SLURM job scripts
```

## Full Documentation

- [rom/README.md](rom/README.md): Complete guide (500+ lines)
- [ENSEMBLE_GUIDE.md](ENSEMBLE_GUIDE.md): Stage 1 details
- [MVAR_GUIDE.md](MVAR_GUIDE.md): Original MVAR (deprecated)

## Getting Help

```bash
# See all options
python scripts/rom_build_pod.py --help
python scripts/rom_train_mvar.py --help
python scripts/rom_evaluate.py --help

# Test on small dataset
rectsim ensemble --config configs/rom_test.yaml --sim.N 20 --ensemble.n_runs 5
```
