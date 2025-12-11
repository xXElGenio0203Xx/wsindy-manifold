# Oscar Submission: Optimal MVAR+LSTM Joint Configuration

## Quick Start

### 1. Transfer files to Oscar
```bash
# From local machine
rsync -avz configs/vicsek_rom_joint_optimal.yaml oscar:~/wsindy-manifold/configs/
rsync -avz slurm_scripts/run_vicsek_joint_optimal.slurm oscar:~/wsindy-manifold/slurm_scripts/
rsync -avz run_unified_rom_pipeline.py oscar:~/wsindy-manifold/
```

### 2. Submit job on Oscar
```bash
ssh oscar
cd ~/wsindy-manifold
sbatch slurm_scripts/run_vicsek_joint_optimal.slurm
```

### 3. Monitor progress
```bash
# Check job status
squeue -u $USER

# Watch output log
tail -f slurm_logs/vicsek_joint_optimal_*.out

# Check for errors
tail -f slurm_logs/vicsek_joint_optimal_*.err
```

### 4. Download results
```bash
# From local machine
rsync -avz oscar:~/wsindy-manifold/oscar_output/vicsek_joint_optimal/ ./oscar_output/
```

### 5. Run visualizations locally
```bash
python run_visualizations.py --experiment_name vicsek_joint_optimal
```

## Configuration Details

### Design Parameters
- **Latent dimension (d)**: 25 modes
- **Lag (w)**: 9 (aligned for MVAR and LSTM)
- **Training horizon**: 10s per trajectory
- **ROM subsampling**: 2 (effective dt = 0.2s)

### Data Statistics
- **Training runs**: 400 (150 gaussian, 150 uniform, 50 ring, 50 two-cluster)
- **Test runs**: 40 (10 per IC type)
- **ROM steps per trajectory**: 50
- **Windows per trajectory**: 41
- **Total training windows**: 16,400

### MVAR Configuration
- **Parameters**: 5,625 (d² × w = 25² × 9)
- **Data/param ratio**: 2.92 (well-determined)
- **Ridge regularization**: α = 1e-4 (strong)
- **Eigenvalue threshold**: 0.999

### LSTM Configuration
- **Hidden units**: 16
- **Layers**: 1
- **Batch size**: 64
- **Learning rate**: 1e-3
- **Max epochs**: 500
- **Early stopping patience**: 20
- **Gradient clipping**: 1.0

### Evaluation
- **Forecast start**: 2.0s
- **Forecast end**: 10.0s
- **Time-resolved analysis**: Enabled
- **Per-model outputs**: Separate directories for MVAR and LSTM

## Expected Outputs

### Directory Structure
```
oscar_output/vicsek_joint_optimal/
├── config_used.yaml
├── summary.json
├── rom_common/              # Shared POD basis
│   ├── pod_basis.npz
│   ├── X_train_mean.npy
│   └── latent_dataset.npz
├── MVAR/                    # MVAR-specific
│   ├── mvar_model.npz
│   └── test_results.csv
├── LSTM/                    # LSTM-specific
│   ├── lstm_state_dict.pt
│   ├── training_log.csv
│   └── test_results.csv
├── train/                   # Training trajectories
└── test/                    # Test trajectories + predictions
    └── test_NNN/
        ├── trajectory.npz
        ├── density_true.npz
        ├── density_pred_mvar.npz
        ├── density_pred_lstm.npz
        └── r2_vs_time_*.csv
```

### Visualization Outputs
```
predictions/vicsek_joint_optimal/
├── best_runs/
│   ├── MVAR/               # Best run videos for MVAR
│   └── LSTM/               # Best run videos for LSTM
├── time_analysis/
│   ├── MVAR/               # Time evolution for MVAR
│   └── LSTM/               # Time evolution for LSTM
├── plots/
│   ├── lstm_training.png
│   ├── mvar_lstm_comparison.png
│   └── ...
└── pipeline_summary.json   # Model comparison statistics
```

## Computational Resources

### Job Specifications
- **Time limit**: 24 hours (estimated ~1.3 hours actual)
- **Memory**: 32GB
- **CPUs**: 8
- **Partition**: batch

### Expected Timeline
- Training simulations: ~67 minutes
- Test simulations: ~7 minutes
- ROM training (POD + MVAR + LSTM): ~5 minutes
- **Total**: ~80 minutes

## Validation Results

All checks passed ✓:
- Lags aligned between MVAR and LSTM
- Data/parameter ratio: 2.92 (adequate)
- 400 training runs (sufficient)
- 40 test runs (adequate)
- Ridge α = 1e-4 (appropriate)
- LSTM properly configured

## Key Advantages Over Previous Runs

1. **Well-determined system**: Data/param ratio of 2.92 (vs <1 in overfitted runs)
2. **Strong regularization**: Ridge α = 1e-4 (vs 1e-6 in overfitted runs)
3. **Aligned lags**: Same window structure for fair MVAR vs LSTM comparison
4. **Longer horizon**: 10s training (vs 2-3s in previous tests)
5. **Modest dimensionality**: 25 modes (vs 287 in overfitted run)
6. **Fair comparison**: LSTM with 16 hidden units comparable to MVAR capacity

## Troubleshooting

### If job fails
```bash
# Check error log
cat slurm_logs/vicsek_joint_optimal_*.err

# Check output log
cat slurm_logs/vicsek_joint_optimal_*.out

# Check pipeline log
cat oscar_output/vicsek_joint_optimal_pipeline.log
```

### If memory issues
Reduce batch size in config:
```yaml
lstm:
  batch_size: 32  # reduce from 64
```

### If time limit exceeded
Either:
1. Request more time: `#SBATCH --time=48:00:00`
2. Reduce training runs in config (not recommended)

## Post-Processing

After downloading results, generate comprehensive visualizations:

```bash
# Run visualization pipeline
python run_visualizations.py --experiment_name vicsek_joint_optimal

# Outputs will be in: predictions/vicsek_joint_optimal/
# - Per-model best run videos
# - Per-model time analysis
# - LSTM training curves
# - MVAR vs LSTM comparison (4-panel plot)
# - Summary JSON with model comparison
```

## Questions or Issues?

Check the comprehensive documentation:
- `LSTM_INTEGRATION_GUIDE.md` - Full integration details
- `ROM_MVAR_GUIDE.md` - MVAR specifics
- `OSCAR_WORKFLOW.md` - Oscar cluster workflow
