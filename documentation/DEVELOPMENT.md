# Development Notes & Maintenance

**Last Updated:** November 21, 2025

This document consolidates development notes, code audits, patches, and maintenance information for the rectsim codebase.

---

## Table of Contents

1. [Recent Changes](#recent-changes)
2. [Config Schema Migration](#config-schema-migration)
3. [Code Patches Applied](#code-patches-applied)
4. [Compatibility Reports](#compatibility-reports)
5. [Known Issues](#known-issues)
6. [Development Workflow](#development-workflow)

---

## Recent Changes

### November 21, 2025 - Config Schema Unification

**Status:** ✅ COMPLETE

**Summary:**
- Fixed two incompatible config schemas (OLD vs NEW)
- Migrated 14 configs from OLD → NEW schema
- Added automatic OLD schema detection with helpful errors
- All 22 active configs now use NEW schema

**Commits:**
- `6c48ee8` - Fix config schema compatibility and improve error handling
- `a815bac` - Migrate all 14 OLD schema configs to NEW schema

**Files Modified:**
- `src/rectsim/config.py` - Complete DEFAULT_CONFIG, add schema validation
- `src/rectsim/cli.py` - Import unified backends, add imageio checks
- `src/rectsim/unified_config.py` - Fix noise.kind field name
- 14 config files migrated with `.old` backups

---

## Config Schema Migration

### NEW Schema (Current - Use This!)

```yaml
model: vicsek_discrete  # String, not dict!
seed: 42

sim:                    # Replaces: domain + particles + integration
  N: 200
  Lx: 20.0
  Ly: 20.0
  bc: periodic
  T: 100.0
  dt: 0.01
  save_every: 10
  integrator: euler
  neighbor_rebuild: 5

model_config:           # NEW section
  speed: 0.5
  speed_mode: constant

params:                 # Replaces: dynamics.alignment + self_propulsion
  R: 2.0                # Alignment radius
  alpha: 1.5            # (continuous only)
  beta: 0.5             # (continuous only)

noise:                  # Replaces: dynamics.noise (promoted to top-level)
  kind: gaussian        # Changed from "type"
  eta: 0.3
  match_variance: true

forces:                 # Replaces: dynamics.forces (promoted to top-level)
  enabled: true
  type: morse
  params:
    Cr: 2.0
    Ca: 1.0
    lr: 0.9
    la: 1.0
    rcut_factor: 3.0
    mu_t: 1.0

ic:
  type: uniform

outputs:
  directory: simulations/my_sim
  order_parameters: true
  plot_order_params: true
  animate_traj: false
  animate_density: false
  video_ics: 1
  order_params_ics: 1
```

### OLD Schema (Deprecated - Don't Use!)

OLD configs with `domain`, `particles`, `dynamics`, `integration` sections are automatically rejected with migration instructions.

**Archived configs:** 14 files backed up with `.old` extension in `configs/`

### Migration Tool

Use `migrate_config.py` to convert OLD → NEW:

```bash
python migrate_config.py configs/old_config.yaml configs/new_config.yaml
```

**Full migration guide:** See `MIGRATION_GUIDE.md` (archived)

---

## Code Patches Applied

### Patch 1: Fix DEFAULT_CONFIG
**File:** `src/rectsim/config.py`  
**Status:** ✅ Applied & Verified

Added missing sections:
- `model_config` (speed, speed_mode)
- Top-level `forces` section
- Top-level `noise` section
- `rom` section (enabled, rank, mvar_order, etc.)

### Patch 2: Add OLD Schema Detection
**File:** `src/rectsim/config.py` (load_config function)  
**Status:** ✅ Applied & Verified

Detects OLD schema keys and provides clear error:
```
ConfigError: OLD config schema detected in configs/xxx.yaml.
Found keys: domain, particles, dynamics, integration.
The config schema has been updated. Please migrate your config file.
See MIGRATION_GUIDE.md for instructions, or use:
  python migrate_config.py configs/xxx.yaml configs/xxx_new.yaml
```

### Patch 3: Use Unified Backends
**File:** `src/rectsim/cli.py`  
**Status:** ✅ Applied & Verified

Changed CLI to use `simulate_backend()` consistently:
- Import `simulate_backend as simulate_backend_continuous` from `dynamics`
- Import `simulate_backend as simulate_backend_discrete` from `vicsek_discrete`
- Modified `_run_single()` to create RNG and call backends

### Patch 4: Add imageio Checks
**File:** `src/rectsim/cli.py`  
**Status:** ✅ Applied & Verified

Added graceful degradation for video generation:
```python
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
```

Emits warning if imageio not installed instead of crashing.

### Patch 5: Fix noise.type → noise.kind
**File:** `src/rectsim/unified_config.py`  
**Status:** ✅ Applied & Verified

Changed `noise.type` to `noise.kind` for consistency with code expectations.

---

## Compatibility Reports

### ROM/MVAR Pipeline ✅

**Status:** Fully compatible with NEW schema

**Verified:**
- NPZ format matches expected structure (T, N, D)
- Contains required fields: `x`, `v`, `times`
- Metadata preserved in appropriate fields
- POD + MVAR training scripts work correctly
- Evaluation pipeline functions properly

**Test run output:**
```
trajectories.npz: x:(21,50,2), v:(21,50,2), times:(21)
density.npz: rho, xgrid, ygrid, times
order_parameter.csv: timestep metrics
```

### Config Inventory

**Active Configs (22 files):**
- All using NEW schema ✅
- All tested and verified loading successfully ✅
- Ready for production use ✅

**Archived Configs (14 files):**
- OLD schema backed up with `.old` extension 📦
- Can be restored if needed 📦
- Serve as reference for migration 📦

---

## Known Issues

### Resolved Issues (Nov 21, 2025)

1. ✅ Two incompatible config schemas → OLD schema now rejected with clear error
2. ✅ Incomplete DEFAULT_CONFIG → All required sections added
3. ✅ Return format inconsistency → Using unified backends
4. ✅ No imageio check → Graceful degradation implemented
5. ✅ Config field name conflicts → noise.kind now consistent

### Outstanding Minor Issues

1. **CLI override support:** Some nested fields may need explicit override patterns
2. **Config validation:** Consider adding comprehensive `_validate_config()` function
3. **Deprecation warnings:** Old `simulate()` functions could emit deprecation warnings

**Priority:** LOW - Core functionality working correctly

---

## Development Workflow

### Making Config Changes

1. **Always use NEW schema** - check example configs in `configs/`
2. **Test loading:** `python -c "from src.rectsim.config import load_config; load_config('configs/myconfig.yaml')"`
3. **CLI overrides:** Use dotted notation: `--sim.N 100 --noise.eta 0.5`

### Adding New Config Fields

1. Update `DEFAULT_CONFIG` in `src/rectsim/config.py`
2. Add validation in `_validate()` function
3. Update example configs
4. Test loading and CLI overrides

### Running Tests

```bash
# Test config loading
python -c "from src.rectsim.config import load_config; cfg = load_config('configs/vicsek_morse_base.yaml'); print('✅ OK')"

# Test simulation
python -m rectsim.cli single --config configs/vicsek_morse_base.yaml --sim.T 10

# Test ROM pipeline
python scripts/rom_mvar_train.py --config configs/rom_mvar_example.yaml
```

### Commit Guidelines

- **Config changes:** Document schema changes, provide migration path
- **Code patches:** Include before/after, verification tests
- **Breaking changes:** Update all docs, provide clear migration guide

---

## Archived Documentation

The following detailed reports are preserved for reference:

- **CODEBASE_AUDIT_REPORT.md** - Full 28-issue audit (archived)
- **CONFIG_SCHEMA_PATCHES.md** - Detailed patch code (archived)
- **MIGRATION_GUIDE.md** - Complete migration instructions (archived)
- **TEST_PATCHES.md** - Verification test results (archived)
- **COMPLETION_SUMMARY.md** - Session summary (archived)
- **COMPATIBILITY_VERIFICATION_REPORT.md** - ROM compatibility details (archived)
- **SCRIPTS_AND_MODULES_AUDIT.md** - Scripts compatibility audit (archived)

To view archived docs: `git show HEAD:FILENAME.md`

---

## Quick Reference

### Config Schema
- **Location:** `src/rectsim/config.py`
- **Default:** `DEFAULT_CONFIG` dict
- **Validation:** `_validate()` function
- **Loading:** `load_config(path, overrides)`

### Migration
- **Script:** `migrate_config.py`
- **Usage:** `python migrate_config.py OLD.yaml NEW.yaml`
- **Backups:** `.old` extension in `configs/`

### Testing
- **Single sim:** `python -m rectsim.cli single --config FILE`
- **Ensemble:** `python scripts/run_sim_production.py --config FILE`
- **ROM/MVAR:** `python scripts/rom_mvar_train.py --config FILE`

---

**For detailed patch information, see archived documentation files or git history.**
