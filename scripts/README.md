# Scripts Directory

This directory contains executable scripts for running simulations, analysis, and utilities.

---

## 🌟 Main Pipeline (Start Here)

### `run_complete_pipeline.py` ⭐
**Complete MVAR-ROM pipeline with IC type stratification**

Self-contained orchestrator that runs the full workflow:
1. Generate training ensemble (100 sims, 4 IC types)
2. Compute global POD + train MVAR model
3. Generate test ensemble (20 sims, stratified)
4. Evaluate predictions with comprehensive metrics
5. Generate visualizations (best runs per IC type)
6. Output summary JSON with all metadata

**Usage:**
```bash
python run_complete_pipeline.py
```

**Outputs:** `outputs/complete_pipeline/`
- `pipeline_summary.json` - Complete metadata
- `best_runs/{ic_type}/` - Videos, plots, metrics for best run per IC
- `plots/` - POD analysis, performance by IC type
- `training/`, `test/` - All simulation data
- `pod/`, `mvar/` - Trained models

**Key Features:**
- Dynamic POD mode selection (99.5% energy target)
- Stratified IC testing (uniform, gaussian_cluster, ring, two_clusters)
- Order parameter analysis (polarization, speed, nematic)
- Comprehensive JSON output for reproducibility

---

## 🔬 Research & Testing Scripts

### `rom_mvar_generalization_test.py`
**Test ROM/MVAR generalization across IC distributions**

Systematic testing on multiple IC types with varying parameters:
- Uniform vs Gaussian distributions
- 1-4 cluster configurations
- Statistical analysis of performance

**Usage:**
```bash
python scripts/rom_mvar_generalization_test.py \
    --experiment vicsek_morse_test \
    --config configs/vicsek_morse_test.yaml \
    --num_uniform 20 \
    --num_gaussian 15
```

**Use case:** Research on IC sensitivity and generalization

---

## 🎬 Post-Processing & Visualization

### `rom_mvar_visualize.py`
**Generate visualizations from saved evaluation results**

Pure visualization script (no computation) for post-processing:
- Reads saved NPZ/CSV data
- Generates density comparison videos
- Creates error plots and dashboards

**Usage:**
```bash
# After rsync from Oscar
python scripts/rom_mvar_visualize.py \
    --experiment my_rom_experiment \
    --ic_ids 0 1 2
```

**Use case:** Local visualization after cluster evaluation

---

## 📊 Alternative Evaluation Scripts

### `rom_mvar_eval.py`
**Standalone evaluation on unseen ICs (Oscar-friendly)**

Config-based evaluation that:
- Generates new test simulations
- Runs ROM predictions
- Computes metrics (no videos by default)

**Usage:**
```bash
python scripts/rom_mvar_eval.py \
    --experiment my_rom_experiment \
    --config configs/rom_eval.yaml
```

**Use case:** Oscar batch evaluation jobs

### `rom_mvar_eval_unseen.py`
**Evaluate on pre-existing simulation datasets**

Lightweight evaluation for archived data:
- Loads existing simulations
- Runs predictions and metrics only

**Usage:**
```bash
python scripts/rom_mvar_eval_unseen.py \
    --rom_dir rom_mvar/exp1/model \
    --unseen_root simulations_unseen \
    --out_dir results/eval_unseen
```

**Use case:** Testing on external/archived datasets

---

## 🛠️ Other Utilities

### Simulation Scripts
- `run_single.py` - Single simulation runs
- `run_dorsogna.py` - D'Orsogna model simulations
- `run_vicsek_discrete.py` - Vicsek discrete model
- `run_grid.py` - Parameter grid searches

### ROM Building Blocks
- `rom_build_pod.py` - Build POD basis from ensemble
- `rom_evaluate.py` - ROM evaluation utilities

### Animation & Media
- `create_animations.py` - Animation generation tools
- `curate_latent_timeseries.py` - Latent space analysis

### Oscar Utilities
- `oscar_env.sh` - Environment setup
- `oscar_sync_results.sh` - Rsync results from cluster
- `oscar_check.sh` - Check cluster job status

---

## 📦 Archived Scripts

Scripts moved to `.archive/scripts/` (historical reference):
- `rom_mvar_train.py` - Config-based training (superseded by main pipeline)
- `rom_train_mvar.py` - Modular MVAR-only training
- `rom_mvar_full_eval_local.py` - Alternative evaluation pipeline
- `rom_mvar_best_plots.py` - Redundant visualization
- `rom_mvar_visualize_best.py` - Redundant visualization

---

## 🎯 Quick Reference

**Running a complete workflow:**
```bash
python run_complete_pipeline.py
```

**Testing IC generalization:**
```bash
python scripts/rom_mvar_generalization_test.py \
    --experiment test_ics \
    --config configs/vicsek_morse_test.yaml
```

**Post-processing cluster results:**
```bash
# 1. Sync from Oscar
bash scripts/oscar_sync_results.sh

# 2. Generate videos locally
python scripts/rom_mvar_visualize.py --experiment my_exp
```

**Evaluating on archived data:**
```bash
python scripts/rom_mvar_eval_unseen.py \
    --rom_dir rom_mvar/model \
    --unseen_root old_simulations
```

---

## 📝 Notes

- **Main pipeline is self-contained** - No dependencies on other scripts
- **Modular scripts** use `src/rectsim/rom_mvar.py` module
- **Oscar scripts** designed for cluster execution (no visualization)
- **Local scripts** include full visualization capabilities

For detailed implementation, see `FILE_USAGE_ANALYSIS.md` in the root directory.
