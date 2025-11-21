# Legacy wsindy_manifold Module - DEPRECATED

**Status:** ARCHIVED  
**Date:** November 21, 2025  
**Superseded By:** `rectsim` package

---

## Deprecation Notice

This module is **NO LONGER MAINTAINED**. All functionality has been migrated to the `rectsim` package.

## Migration Guide

| Legacy Module | Current Replacement |
|--------------|---------------------|
| `wsindy_manifold.latent.pod` | `rectsim.mvar` (pod_fit, pod_project, pod_reconstruct) |
| `wsindy_manifold.latent.mvar` | `rectsim.mvar.MVARModel` |
| `wsindy_manifold.latent.kde` | `rectsim.density.compute_density_grid` |
| `wsindy_manifold.latent.metrics` | `rectsim.metrics` |
| `wsindy_manifold.latent.anim` | `rectsim.io_outputs` (animation functions) |
| `wsindy_manifold.standard_metrics` | `rectsim.standard_metrics` |
| `wsindy_manifold.density` | `rectsim.density` |
| `wsindy_manifold.io` | `rectsim.io_outputs` (partial), inline helpers |
| `wsindy_manifold.pod` | `rectsim.mvar` |
| `wsindy_manifold.mvar_rom` | `rectsim.rom_mvar` |
| `wsindy_manifold.efrom` | Use `scripts/rom_mvar_*.py` pipeline |

## Current ROM/MVAR Pipeline

Instead of using this legacy module, use the current pipeline:

```bash
# Training
python scripts/rom_mvar_train.py --config CONFIG

# Evaluation
python scripts/rom_mvar_eval.py --experiment NAME --config CONFIG

# Or use SLURM pipeline on Oscar
bash scripts/slurm/submit_mvar_pipeline.sh CONFIG
```

## Why Was This Deprecated?

1. **Duplicate Code:** `wsindy_manifold` and `rectsim` had overlapping functionality
2. **Confusing Structure:** Two separate packages doing similar things
3. **Maintenance Burden:** Had to update two codebases
4. **Better Organization:** `rectsim` has cleaner API and better docs

## Scripts That Used This Module

These scripts have been updated or deprecated:

**Updated:**
- `examples/quickstart_rect2d.py` - Now has try/except with helpful error
- `demo_mvar_rom_with_videos.py` - Now has deprecation notice

**Deprecated:**
- `scripts/run_sim_production.py` - Use `rectsim-single` instead

**Archived:**
- 11 test files moved to `tests/legacy_wsindy/`

## Module Structure (Historical Reference)

```
wsindy_manifold/
├── __init__.py
├── density.py              # KDE density computation
├── io.py                   # Run directory management, git tracking
├── pod.py                  # POD utilities
├── standard_metrics.py     # Order parameters, error metrics
└── latent/
    ├── __init__.py
    ├── anim.py            # Animation helpers
    ├── flow.py            # Training/forecasting workflow
    ├── kde.py             # KDE grid generation
    ├── metrics.py         # Frame/series metrics
    ├── mvar.py            # MVAR model fitting/forecasting
    └── pod.py             # POD fit/restrict/lift
```

## If You Need This Code

If you absolutely need to use this legacy module:

1. **Copy it back:**
   ```bash
   cp -r .archive/legacy_wsindy_manifold src/wsindy_manifold
   ```

2. **Install in editable mode:**
   ```bash
   pip install -e .
   ```

3. **Consider migrating to `rectsim` instead** - it's better maintained!

## Related Documentation

- **Current ROM/MVAR Guide:** [ROM_MVAR.md](../ROM_MVAR.md)
- **Rectsim Package:** [src/rectsim/](../src/rectsim/)
- **Consolidation Summary:** [DOCUMENTATION_CLEANUP.md](../DOCUMENTATION_CLEANUP.md)

---

**Last Updated:** November 21, 2025  
**Archived From:** `src/wsindy_manifold/`  
**Reason:** Superseded by `rectsim` package
