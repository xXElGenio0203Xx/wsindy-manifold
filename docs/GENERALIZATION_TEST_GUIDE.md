# ROM/MVAR Generalization Testing Pipeline

## Overview

This pipeline tests how well the trained ROM/MVAR model generalizes to new initial conditions with different spatial distributions. It's designed to run **locally on your computer** after training on Oscar.

## Workflow

### Step 1: Train Model on Oscar (Heavy Compute)
```bash
# On Oscar
cd ~/src/wsindy-manifold
bash scripts/slurm/submit_test_pipeline.sh  # or full production pipeline
```

This generates:
- `rom_mvar/{experiment}/model/pod_basis.npz`
- `rom_mvar/{experiment}/model/mvar_params.npz`
- `rom_mvar/{experiment}/model/train_summary.json`

### Step 2: Sync Results to Local Machine
```bash
# On your Mac
bash scripts/oscar_sync_results.sh vicsek_morse_test
```

Downloads model files to `~/wsindy-results/rom_mvar/{experiment}/`

### Step 3: Run Generalization Test (Local)
```bash
# Test with default settings (10 uniform, 10 gaussian per cluster count)
python scripts/rom_mvar_generalization_test.py \
    --experiment vicsek_morse_test \
    --config configs/vicsek_morse_test.yaml \
    --rom_root ~/wsindy-results/rom_mvar

# Custom IC counts
python scripts/rom_mvar_generalization_test.py \
    --experiment vicsek_morse_test \
    --config configs/vicsek_morse_test.yaml \
    --rom_root ~/wsindy-results/rom_mvar \
    --num_uniform 20 \
    --num_gaussian 15 \
    --cluster_counts 1 2 3 4
```

**What this does:**
1. Loads trained ROM/MVAR model
2. Generates new ICs:
   - Uniform random positions
   - Gaussian clustered positions (1, 2, 3, 4 clusters)
3. For each IC:
   - Runs ground truth simulation
   - Projects IC to latent space
   - Forecasts with MVAR
   - Reconstructs density predictions
   - Computes metrics (R², RMSE, tolerance horizon)
4. Saves lightweight outputs:
   - Individual IC results (densities, metrics, summaries)
   - Aggregate statistics (uniform vs gaussian performance)
   - Identifies best R² for each IC type

**Outputs:**
```
rom_mvar/{experiment}/generalization_test/
├── uniform_ic_2000/
│   ├── densities.npz         # True and predicted densities
│   ├── metrics.csv           # R², RMSE timeseries
│   └── summary.json          # Aggregate metrics
├── uniform_ic_2001/
├── ...
├── gaussian_1clust_ic_3000/
├── gaussian_2clust_ic_4000/
├── ...
├── all_summaries.csv         # All IC summaries
└── aggregate_stats.json      # Uniform vs gaussian stats + best ICs
```

### Step 4: Visualize Best ICs Only (Local)
```bash
python scripts/rom_mvar_visualize_best.py \
    --experiment vicsek_morse_test \
    --rom_root ~/wsindy-results/rom_mvar
```

**What this does:**
1. Reads `aggregate_stats.json` to find best R² ICs
2. Generates videos **only** for:
   - Best uniform IC
   - Best gaussian IC for each cluster count (1, 2, 3, 4)
3. Generates order parameter plots for best ICs
4. Creates comparison plot (uniform vs gaussian performance)

**Outputs:**
```
rom_mvar/{experiment}/generalization_test/best_ic_videos/
├── best_uniform_seed2003.gif
├── best_uniform_seed2003_order_params.png
├── best_gaussian_1clust_seed3002.gif
├── best_gaussian_1clust_seed3002_order_params.png
├── best_gaussian_2clust_seed4005.gif
├── best_gaussian_2clust_seed4005_order_params.png
├── best_gaussian_3clust_seed5001.gif
├── best_gaussian_3clust_seed5001_order_params.png
├── best_gaussian_4clust_seed6007.gif
├── best_gaussian_4clust_seed6007_order_params.png
└── performance_comparison.png
```

## Understanding the Results

### Aggregate Statistics File
`aggregate_stats.json` contains:

```json
{
  "uniform": {
    "count": 10,
    "r2_mean": 0.85,
    "r2_std": 0.05,
    "rmse_mean": 0.12,
    "rmse_std": 0.02,
    "best_ic_seed": 2003,
    "best_ic_r2": 0.92
  },
  "gaussian": {
    "1_clusters": {
      "count": 10,
      "r2_mean": 0.88,
      "r2_std": 0.04,
      "best_ic_seed": 3002,
      "best_ic_r2": 0.94
    },
    "2_clusters": { ... },
    "3_clusters": { ... },
    "4_clusters": { ... }
  }
}
```

### Key Questions Answered

1. **Does the model generalize to uniform ICs?**
   - Check `uniform.r2_mean` and `uniform.rmse_mean`

2. **How does clustering affect prediction quality?**
   - Compare R² across different `gaussian.*_clusters`

3. **Which IC type is easiest to predict?**
   - Compare `uniform` vs `gaussian` R² means

4. **How does cluster count affect performance?**
   - Look at trend: `1_clusters` → `2_clusters` → `3_clusters` → `4_clusters`

## Complete Example

```bash
# 1. On Oscar: Train model (already done)
# 2. Sync to local
bash scripts/oscar_sync_results.sh vicsek_morse_test

# 3. Run generalization test
python scripts/rom_mvar_generalization_test.py \
    --experiment vicsek_morse_test \
    --config configs/vicsek_morse_test.yaml \
    --rom_root ~/wsindy-results/rom_mvar \
    --num_uniform 15 \
    --num_gaussian 10 \
    --cluster_counts 1 2 3 4

# 4. Visualize best ICs
python scripts/rom_mvar_visualize_best.py \
    --experiment vicsek_morse_test \
    --rom_root ~/wsindy-results/rom_mvar

# 5. View results
open ~/wsindy-results/rom_mvar/vicsek_morse_test/generalization_test/best_ic_videos/
cat ~/wsindy-results/rom_mvar/vicsek_morse_test/generalization_test/aggregate_stats.json | python -m json.tool
```

## Customization

### Change Number of Test ICs
```bash
--num_uniform 20      # Test 20 uniform ICs
--num_gaussian 15     # Test 15 gaussian ICs per cluster count
```

### Test Different Cluster Counts
```bash
--cluster_counts 1 2 3 4 5 6    # Test 1 to 6 clusters
--cluster_counts 2 4            # Test only 2 and 4 clusters
```

### Change Base Seed
```bash
--base_seed 5000      # Start seeds from 5000 instead of 2000
```

### Adjust Video Frame Rate
```bash
python scripts/rom_mvar_visualize_best.py \
    --experiment vicsek_morse_test \
    --rom_root ~/wsindy-results/rom_mvar \
    --fps 20    # Faster video (default: 10)
```

## Performance Tips

- **Parallel execution**: The generalization test runs simulations sequentially. For faster results, consider splitting IC ranges and running multiple instances.
- **Memory**: Each IC stores full density movies. With default settings (10+40 ICs), expect ~2-3 GB of output data.
- **Time**: Each IC takes ~30-60 seconds depending on simulation length. Default run (~50 ICs) takes ~30-45 minutes.

## Output Files Summary

| File | Description |
|------|-------------|
| `uniform_ic_*/densities.npz` | True and predicted density movies |
| `uniform_ic_*/metrics.csv` | R², RMSE, mass error timeseries |
| `uniform_ic_*/order_params.csv` | Polarization, angular momentum, speeds |
| `uniform_ic_*/summary.json` | Mean R², RMSE, tolerance horizon |
| `gaussian_*clust_ic_*/...` | Same as above for gaussian ICs |
| `all_summaries.csv` | All IC summaries in one table |
| `aggregate_stats.json` | **Main results**: uniform vs gaussian stats + best ICs |
| `best_ic_videos/*.gif` | Density comparison videos (best ICs only) |
| `best_ic_videos/*_order_params.png` | Order parameter plots (best ICs only) |
| `best_ic_videos/performance_comparison.png` | Bar plots comparing IC types |

## Troubleshooting

### "Model directory not found"
Make sure you synced results from Oscar:
```bash
bash scripts/oscar_sync_results.sh vicsek_morse_test
```

### "initial_positions not supported"
The generalization test directly modifies the config to set initial positions. If you get this error, check that your simulation code supports the `initial_positions` config parameter.

### Videos not generating
Check that you have `imageio` and `pillow` installed:
```bash
pip install imageio pillow
```

### Out of memory
Reduce the number of test ICs:
```bash
--num_uniform 5 --num_gaussian 5
```
