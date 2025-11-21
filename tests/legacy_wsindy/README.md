# Legacy wsindy_manifold Tests

These tests use the deprecated `wsindy_manifold` module and are no longer maintained.

## Status

**ARCHIVED** - November 21, 2025

## Reason

The `wsindy_manifold` module has been superseded by the `rectsim` package, which provides:
- Better organization
- Unified interface
- Active maintenance
- Integration with current ROM/MVAR pipeline

## Migrated Functionality

| Legacy Module | Current Location |
|---------------|------------------|
| `wsindy_manifold.latent.pod` | `rectsim.mvar.pod_*` functions |
| `wsindy_manifold.latent.mvar` | `rectsim.mvar.MVARModel` |
| `wsindy_manifold.standard_metrics` | `rectsim.standard_metrics` |
| `wsindy_manifold.density` | `rectsim.density` |
| `wsindy_manifold.io` | `rectsim.io_outputs` (partially) |

## Current Tests

For active ROM/MVAR tests, see:
- `tests/test_mvar.py` - MVAR model tests
- `tests/test_rom_eval_*.py` - ROM evaluation tests
- `tests/test_rom_video_utils.py` - Video generation tests

## Archived Test Files

- `test_pod_old.py` - Old POD tests
- `test_efrom.py` - Old EF-ROM tests
- `test_mvar_enhanced.py` - Enhanced MVAR tests
- `test_density_pod.py` - Density POD tests
- `test_heatmap_flow.py` - Heatmap flow tests
- `test_latent_metrics.py` - Latent metrics tests
- `test_flow.py` - Flow tests
- `test_kde.py` - KDE tests
- `test_anim.py` - Animation tests
- `test_pod.py` - POD tests
- `test_alignment_vicsek.py` - Alignment tests
- `test_mvar_rom.py` - MVAR ROM pipeline tests (uses wsindy_manifold.mvar_rom)
- `test_kde_density.py` - KDE density tests (uses rectsim.kde_density - module doesn't exist)
- `test_crowdrom_pipeline.py` - CrowdROM pipeline tests
- `test_efrom_data.py` - EF-ROM data pipeline tests

## Running These Tests

If you need to run these legacy tests:

```bash
# Ensure wsindy_manifold is still in src/
python -m pytest tests/legacy_wsindy/test_*.py
```

Note: These tests may fail if `wsindy_manifold` has been fully archived.
