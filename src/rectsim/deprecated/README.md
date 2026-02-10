# Deprecated Modules

**Status**: These files are NOT used by the production ROM-MVAR pipeline  
**Date Deprecated**: February 2, 2026  
**Reason**: Replaced by newer implementations or not imported by `run_unified_mvar_pipeline.py`  

---

## Active Production Pipeline

**Entry Point**: `run_unified_mvar_pipeline.py`

**Active Modules** (still in parent directory):
- `config_loader.py` - YAML parsing
- `ic_generator.py` - IC generation (408 training + 31 test)
- `simulation_runner.py` - Parallel orchestration
- `pod_builder.py` - POD basis construction
- `mvar_trainer.py` - MVAR training
- `test_evaluator.py` - Forecast evaluation
- `vicsek_discrete.py` - Vicsek backend
- `legacy_functions.py` - KDE, videos, metrics
- `standard_metrics.py` - Order parameters
- `utils.py` - Neighbor search

**See**: `docs/OFFICIAL_PIPELINE_ARCHITECTURE.md` for complete dependency map

---

## Deprecated Files by Category

### Category A: Alternative Implementations (4 files)
- `pod.py` - Full POD class (overkill, `pod_builder.py` is sufficient)
- `mvar.py` - Old MVAR utilities (superseded by `mvar_trainer.py`)
- `rom_mvar.py` - Alternative ROM-MVAR implementation
- `density.py` - Old KDE (superseded by `legacy_functions.kde_density_movie()`)

### Category B: Evaluation/Visualization (6 files)
- `rom_eval.py` - Evaluation orchestrator (inlined in `test_evaluator.py`)
- `rom_eval_metrics.py` - Metrics functions (inlined)
- `rom_eval_viz.py` - Visualization functions (use `legacy_functions.side_by_side_video()`)
- `rom_eval_data.py` - Data loading utilities
- `rom_eval_pipeline.py` - Old evaluation pipeline
- `rom_video_utils.py` - Video utilities (superseded by legacy_functions)

### Category C: Alternative Models (3 files)
- `rom_mvar_model.py` - Object-oriented MVAR wrapper
- `forecast_utils.py` - Forecast function factories
- `rom_data_utils.py` - LSTM data prep (not in MVAR-only pipeline)

### Category D: Alternative Dynamics (4 files)
- `dynamics.py` - General dynamics interface
- `morse.py` - Morse potential model
- `integrators.py` - RK4/Euler integrators
- `noise.py` - Noise generation utilities

### Category E: I/O and Config (4 files)
- `io.py` - Old I/O functions (superseded by `io_outputs.py`)
- `io_outputs.py` - Standardized outputs (not imported by pipeline)
- `config.py` - Old config system (superseded by `config_loader.py`)
- `unified_config.py` - Experiment config builder

### Category F: Initial Conditions (2 files)
- `ic.py` - Old IC module
- `initial_conditions.py` - Alternative IC generator

### Category G: Miscellaneous (4 files)
- `domain.py` - Domain/boundary utilities
- `metrics.py` - Alternative metrics
- `cli.py` - Command-line interface (for standalone use, not pipeline)
- `rom_eval_smoke_test.py` - Test script

---

## Total Deprecated

**Files**: 28  
**Lines**: ~8,000  
**Reason**: Not imported by production pipeline, replaced by active modules, or experimental code

---

## Restoration

If you need to restore any of these files:

```bash
# Move back to parent directory
git mv src/rectsim/deprecated/FILENAME.py src/rectsim/

# Or copy without git
cp src/rectsim/deprecated/FILENAME.py src/rectsim/
```

---

**Document Version**: 1.0  
**Last Updated**: February 2, 2026
