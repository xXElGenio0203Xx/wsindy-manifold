# CRITICAL BUG FIX: POD Mode Selection Priority

## Problem Identified

The `best_run_extended_test` experiment used **287 POD modes** instead of the specified **25 modes**, causing severe overfitting and catastrophic prediction failure.

### Root Cause

**Config files use `fixed_modes: 25` but pipeline code only checked `fixed_d`**, causing it to fall back to the energy threshold (`pod_energy: 0.995`) which selected 287 modes (99.5% of 4096-dimensional data).

### Temporal Analysis Reveals the Issue

```
Training window = lag × dt = 20 × 0.1s = 2.0s
Training duration = 2.0s

Result: Model memorizes ONE WINDOW per trajectory!
```

**Performance degradation timeline:**
- t=2.0→3.1s (R²≥0.5): Lag window is 50%→95% ground truth ✓
- t=3.1→4.0s (R²=0.5→0): Lag window becomes 0%→50% ground truth ⚠️
- t=4.0s+ (R²<0): **FULLY CONTAMINATED** - entire window is predicted data ❌

**At t=4.0s, the 2.0s lag window is 100% predicted values → cascade failure**

## Solution Implemented

Fixed POD mode selection priority in all pipeline files:

### Before (WRONG):
```python
TARGET_ENERGY = rom_config.get('pod_energy', 0.995)
FIXED_D = rom_config.get('fixed_d', None)  # Config uses 'fixed_modes' not 'fixed_d'!

if FIXED_D is not None:
    R_POD = FIXED_D  # Never executed because FIXED_D is always None
else:
    R_POD = np.searchsorted(cumulative_energy, TARGET_ENERGY) + 1  # Always used
```

**Result**: Config specifies `fixed_modes: 25` → Code sees `None` → Uses energy threshold → Selects 287 modes

### After (CORRECT):
```python
# Priority: fixed_modes/fixed_d (if specified) > pod_energy (threshold)
FIXED_D = rom_config.get('fixed_modes', None)  # Check standard name FIRST
if FIXED_D is None:
    FIXED_D = rom_config.get('fixed_d', None)  # Backward compatibility

TARGET_ENERGY = rom_config.get('pod_energy', 0.995)

if FIXED_D is not None:
    R_POD = min(FIXED_D, len(S))  # PRIORITY: explicit mode count
    print(f"✓ Using FIXED d={R_POD} modes (hard cap from config)")
else:
    R_POD = np.searchsorted(cumulative_energy, TARGET_ENERGY) + 1
    print(f"✓ R_POD = {R_POD} modes (threshold={TARGET_ENERGY})")
```

**Result**: Config specifies `fixed_modes: 25` → Code reads it → Uses exactly 25 modes ✓

## Files Modified

1. `run_stable_mvar_pipeline.py` (line ~540)
2. `run_robust_mvar_pipeline.py` (line ~425)  
3. `run_gaussians_pipeline.py` (line ~346)

## Impact

### Before Fix:
- Config: `fixed_modes: 25` → Actual: **287 modes** (99.5% energy)
- Severe overfitting: Training R²=1.0, Test R²=-3.85
- Model fails at t=4.0s when lag window fully contaminated

### After Fix:
- Config: `fixed_modes: 25` → Actual: **25 modes** (will be ~49% energy)
- Expected: Better generalization through aggressive dimensionality reduction
- Historical result: R²=0.509 with 25 modes

## Recommended Actions

1. **Rerun `best_run_extended_test`** with fixed code to verify it now uses 25 modes
2. **Compare results**: 25 modes vs 287 modes on same test set
3. **Validate temporal degradation**: Check if 25 modes maintain R²>0.5 longer
4. **Test training window hypothesis**: Try T_train=10s to give model more dynamics

## Configuration Guidance

For future experiments, the priority order is now:

1. **`fixed_modes: N`** (standard name, highest priority)
2. **`fixed_d: N`** (backward compatibility)
3. **`pod_energy: 0.XX`** (fallback if no fixed mode count specified)

Example configs:
```yaml
rom:
  fixed_modes: 25      # PRIORITY 1: Use exactly 25 modes
  pod_energy: 0.995    # IGNORED when fixed_modes specified
  
rom:
  pod_energy: 0.80     # Use energy threshold if fixed_modes not specified
```

## Temporal Window Recommendations

Current issue: **lag window = training duration = 2.0s** → model memorizes one window!

Solutions:
- **Option A**: Increase T_train to 10s+ (lag window << training duration)
- **Option B**: Decrease lag to 5-10 (window = 0.5-1.0s << 2.0s training)
- **Option C**: Both (recommended for best long-term performance)

**Target ratio**: lag window should be ≤ 20% of training duration for good generalization.

Current: 2.0s / 2.0s = 100% ❌
Recommended: 2.0s / 10s = 20% ✓ or 0.5s / 2.0s = 25% ✓
