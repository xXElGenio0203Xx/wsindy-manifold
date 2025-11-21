# Config Schema Unification Patches

## Patch 1: Fix DEFAULT_CONFIG in config.py

**File:** `src/rectsim/config.py`  
**Lines:** 31-150

```python
# BEFORE (INCOMPLETE):
DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 0,
    "out_dir": "outputs/single",
    "device": "cpu",
    "model": "social_force",  # ❌ Doesn't match NEW schema
    "sim": {
        "N": 200,
        # ... existing fields
    },
    "params": {
        "alpha": 1.5,
        "beta": 0.5,
        "Cr": 2.0,
        # ...
    },
    # ❌ MISSING: model_config, forces (top-level), rom
}

# AFTER (COMPLETE):
DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 0,
    "out_dir": "simulations/single",  # ✅ Changed path
    "device": "cpu",
    "model": "vicsek_discrete",  # ✅ String for NEW schema
    
    # Simulation parameters
    "sim": {
        "N": 200,
        "Lx": 20.0,
        "Ly": 20.0,
        "bc": "periodic",
        "T": 100.0,
        "dt": 0.1,  # ✅ Discrete-appropriate default
        "save_every": 10,
        "integrator": "euler_semiimplicit",  # ✅ Stable for forces
        "neighbor_rebuild": 1,  # ✅ Discrete needs frequent rebuild
    },
    
    # ✅ ADD: Model-specific configuration
    "model_config": {
        "speed": 0.5,  # Natural particle speed v0
        "speed_mode": "constant",  # "constant" or "variable"
    },
    
    # Model parameters (alignment, self-propulsion)
    "params": {
        "R": 2.0,  # Alignment radius
        "alpha": 1.5,  # Self-propulsion (continuous only)
        "beta": 0.5,   # Friction (continuous only)
    },
    
    # ✅ FIX: Noise configuration
    "noise": {
        "kind": "gaussian",  # ✅ Changed from "type"
        "eta": 0.3,  # Angular noise strength
        "sigma": None,  # Auto-computed if match_variance=True
        "match_variance": True,
    },
    
    # ✅ ADD: Forces at top level (not under params)
    "forces": {
        "enabled": False,
        "type": "morse",
        "params": {
            "Cr": 2.0,
            "Ca": 1.0,
            "lr": 0.9,
            "la": 1.0,
            "rcut_factor": 3.0,
            "mu_t": 1.0,
        },
    },
    
    # Initial conditions
    "ic": {
        "type": "uniform",  # uniform, gaussian, ring, cluster
    },
    
    # Ensemble simulation
    "ensemble": {
        "n_runs": 20,  # ✅ Changed from "cases"
        "seeds": None,  # List of specific seeds (or None for auto)
        "base_seed": 0,  # Starting seed for auto-generation
        "ic_types": ["gaussian", "uniform", "ring", "cluster"],
        "ic_weights": None,  # Weights for sampling (or None for uniform)
    },
    
    # ✅ ADD: ROM/MVAR configuration
    "rom": {
        "train": {
            "latent_dim": 10,
            "mvar_order": 3,
            "ridge": 1e-6,
            "train_frac": 0.8,
        },
        "eval": {
            "tol": 0.1,
            "generate_plots": True,
            "generate_videos": False,
            "unseen_root": "simulations_unseen",
            "out_root": "rom_mvar/evaluation",
        },
    },
    
    # Outputs
    "outputs": {
        "save_npz": True,
        "save_csv": True,
        "plots": True,
        "animate_traj": False,  # ✅ Videos OFF by default
        "animate_density": False,  # ✅ Videos OFF by default
        "video_ics": 1,  # Number of ICs to generate videos for
        "plot_order_params": True,
        "order_params_ics": 1,  # Number of ICs for order param plots
        "grid_density": {
            "enabled": True,
            "nx": 128,
            "ny": 128,
            "bandwidth": 0.5,
        },
        "plot_options": {
            "traj_marker_size": 20,
            "traj_quiver": True,
            "traj_quiver_scale": 3.0,
            "traj_quiver_width": 0.004,
            "traj_quiver_alpha": 0.8,
        },
    },
}
```

---

## Patch 2: Add Schema Detection to load_config()

**File:** `src/rectsim/config.py`  
**Lines:** 240-280

```python
def load_config(
    config_path: Path | str,
    overrides: Iterable[Tuple[str, str]] | None = None,
) -> Dict[str, Any]:
    """Load and validate configuration with schema detection."""
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        user_config = yaml.safe_load(f) or {}
    
    # ✅ ADD: Schema detection
    old_schema_keys = {"domain", "particles", "dynamics", "integration"}
    found_old_keys = set(user_config.keys()) & old_schema_keys
    
    if found_old_keys:
        raise ConfigError(
            f"Config file '{config_path}' uses OLD schema. "
            f"Found deprecated keys: {found_old_keys}\\n\\n"
            "The OLD schema (domain/particles/dynamics/integration) is no longer supported.\\n"
            "Please migrate to NEW schema (model/sim/params/forces/noise).\\n\\n"
            "Migration guide:\\n"
            "  - domain.Lx/Ly/bc → sim.Lx/Ly/bc\\n"
            "  - particles.N → sim.N\\n"
            "  - integration.T/dt → sim.T/dt\\n"
            "  - dynamics.alignment/forces/noise → params/forces/noise (top-level)\\n"
            "  - model.type: 'discrete' → model: 'vicsek_discrete' (string)\\n\\n"
            "See MIGRATION_GUIDE.md for complete examples."
        )
    
    # Deep merge with defaults
    config = _deep_update(DEFAULT_CONFIG.copy(), user_config)
    
    # Apply CLI overrides
    if overrides:
        config = _apply_overrides(config, overrides)
    
    # Basic validation
    _validate_config(config)
    
    return config
```

---

## Patch 3: Use Unified Backends in CLI

**File:** `src/rectsim/cli.py`  
**Lines:** 580-630

```python
def _run_single(config: Dict, prog_bar: bool = True) -> Dict:
    """Run a single simulation and return standardized results.
    
    This function now ALWAYS uses the unified simulate_backend() interface
    to ensure consistent return format across all model types.
    """
    
    # Create RNG from seed
    rng = np.random.default_rng(config.get("seed", 0))
    
    # Determine model type
    model_type = config.get("model", "social_force")
    
    # ✅ ALWAYS use unified backend
    if model_type in ["vicsek_discrete", "discrete"]:
        from .vicsek_discrete import simulate_backend
        result = simulate_backend(config, rng)
    elif model_type in ["social_force", "continuous", "dorsogna"]:
        from .dynamics import simulate_backend
        result = simulate_backend(config, rng)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # ✅ Result now has CONSISTENT format:
    # {
    #     "times": np.ndarray,  # (T,)
    #     "traj": np.ndarray,   # (T, N, 2)
    #     "vel": np.ndarray,    # (T, N, 2)
    #     "meta": dict,         # Metadata including config
    # }
    
    return result
```

---

## Patch 4: Add imageio Availability Check

**File:** `src/rectsim/cli.py`  
**Lines:** 200-250 (various video functions)

```python
# ✅ ADD: Module-level availability check
_IMAGEIO_AVAILABLE = None

def _check_imageio():
    """Check if imageio is available for video generation."""
    global _IMAGEIO_AVAILABLE
    
    if _IMAGEIO_AVAILABLE is None:
        try:
            import imageio
            _IMAGEIO_AVAILABLE = True
        except ImportError:
            _IMAGEIO_AVAILABLE = False
    
    return _IMAGEIO_AVAILABLE


def _create_trajectory_video(out_dir: Path, result: Dict, config: Dict):
    """Create trajectory animation video."""
    
    # ✅ ADD: Check at function start
    if not _check_imageio():
        warnings.warn(
            "imageio not installed. Skipping trajectory video generation. "
            "Install with: pip install imageio imageio-ffmpeg",
            RuntimeWarning,
            stacklevel=2
        )
        return
    
    import imageio  # Safe to import now
    
    # ... rest of function unchanged


def _create_density_video(out_dir: Path, density: np.ndarray, times: np.ndarray, config: Dict):
    """Create density heatmap animation video."""
    
    # ✅ ADD: Check at function start
    if not _check_imageio():
        warnings.warn(
            "imageio not installed. Skipping density video generation. "
            "Install with: pip install imageio imageio-ffmpeg",
            RuntimeWarning,
            stacklevel=2
        )
        return
    
    import imageio  # Safe to import now
    
    # ... rest of function unchanged
```

---

## Patch 5: Fix noise.type → noise.kind in unified_config.py

**File:** `src/rectsim/unified_config.py`  
**Lines:** 59

```python
# BEFORE:
'noise': {
    'type': 'gaussian',  # ❌ Conflicts with vicsek_discrete.py

# AFTER:
'noise': {
    'kind': 'gaussian',  # ✅ Matches vicsek_discrete.py and NEW schema
    'eta': 0.3,
    'match_variance': True,
    'Dtheta': 0.001,  # Continuous only
},
```

---

## Patch 6: Add Deprecation Warning to Old Functions

**File:** `src/rectsim/vicsek_discrete.py`  
**Lines:** 195 (start of simulate_vicsek)

```python
def simulate_vicsek(cfg: dict) -> dict:
    """Simulate the discrete-time Vicsek model controlled by ``cfg``.
    
    .. deprecated:: 2024.12
        Use :func:`simulate_backend` instead for consistent interface across model types.
        This function will be removed in a future version.
    """
    
    # ✅ ADD: Deprecation warning
    warnings.warn(
        "simulate_vicsek() is deprecated. Use simulate_backend() for unified interface. "
        "This function will be removed in version 2025.x",
        DeprecationWarning,
        stacklevel=2
    )
    
    # ... rest of function unchanged
```

---

## Patch 7: Standardize Config Access Patterns

**File:** `src/rectsim/cli.py`  
**Function:** `_footer_text()` and others

```python
# BEFORE (inconsistent):
def _footer_text(config: Dict) -> str:
    params = config["params"]  # ❌ Assumes flat structure
    sim = config["sim"]        # ✅ Correct
    
    return (
        f"N={sim['N']}  Lx={sim['Lx']}..."
        f"Cr={params['Cr']}..."  # ⚠️ params.Cr might not exist
    )

# AFTER (consistent):
def _footer_text(config: Dict) -> str:
    sim = config.get("sim", {})
    params = config.get("params", {})
    forces = config.get("forces", {}).get("params", {})
    
    # Try both locations for backward compatibility
    Cr = forces.get("Cr", params.get("Cr", 0.0))
    Ca = forces.get("Ca", params.get("Ca", 0.0))
    
    return (
        f"N={sim.get('N', 0)}  Lx={sim.get('Lx', 0.0)}  Ly={sim.get('Ly', 0.0)}  bc={sim.get('bc', 'periodic')}\\n"
        f"Cr={Cr}  Ca={Ca}  dt={sim.get('dt', 0.01)}"
    )
```

---

## Patch 8: Add Config Validation

**File:** `src/rectsim/config.py`  
**Lines:** Add new function after load_config()

```python
def _validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration and raise ConfigError if invalid."""
    
    # Check required top-level keys
    required_keys = ["model", "sim"]
    for key in required_keys:
        if key not in config:
            raise ConfigError(f"Missing required config key: '{key}'")
    
    # Validate sim section
    sim = config.get("sim", {})
    if sim.get("N", 0) <= 0:
        raise ConfigError("sim.N must be positive")
    if sim.get("Lx", 0) <= 0 or sim.get("Ly", 0) <= 0:
        raise ConfigError("sim.Lx and sim.Ly must be positive")
    if sim.get("T", 0) <= 0:
        raise ConfigError("sim.T must be positive")
    if sim.get("dt", 0) <= 0:
        raise ConfigError("sim.dt must be positive")
    if sim.get("bc") not in ["periodic", "reflecting"]:
        raise ConfigError(f"Invalid boundary condition: {sim.get('bc')}")
    
    # Validate model type
    model = config.get("model", "")
    valid_models = ["vicsek_discrete", "discrete", "social_force", "continuous", "dorsogna"]
    if model not in valid_models:
        raise ConfigError(
            f"Invalid model type: '{model}'. "
            f"Valid options: {', '.join(valid_models)}"
        )
    
    # Validate noise configuration
    noise = config.get("noise", {})
    noise_kind = noise.get("kind", "gaussian")
    if noise_kind not in ["gaussian", "uniform"]:
        raise ConfigError(f"Invalid noise.kind: '{noise_kind}'. Use 'gaussian' or 'uniform'")
    
    # Validate forces if enabled
    forces = config.get("forces", {})
    if forces.get("enabled", False):
        force_params = forces.get("params", {})
        if force_params.get("Cr", 0) < 0 or force_params.get("Ca", 0) < 0:
            raise ConfigError("Force strengths (Cr, Ca) must be non-negative")
        if force_params.get("lr", 0) <= 0 or force_params.get("la", 0) <= 0:
            raise ConfigError("Force length scales (lr, la) must be positive")
    
    # Validate ensemble configuration
    ensemble = config.get("ensemble", {})
    if ensemble.get("n_runs", 1) <= 0:
        raise ConfigError("ensemble.n_runs must be positive")
    
    # Warn about suspicious values
    if sim.get("dt", 0) > 1.0 and model in ["vicsek_discrete", "discrete"]:
        warnings.warn(
            f"Large time step dt={sim.get('dt')} for discrete model. "
            "Consider dt < 1.0 for stability.",
            RuntimeWarning
        )
```

---

## Application Instructions

### Step 1: Backup Current Config System
```bash
cp src/rectsim/config.py src/rectsim/config.py.backup
cp src/rectsim/cli.py src/rectsim/cli.py.backup
cp src/rectsim/unified_config.py src/rectsim/unified_config.py.backup
```

### Step 2: Apply Patches in Order
```bash
# Apply patches 1-8 to respective files
# Test after each patch to ensure no breakage
```

### Step 3: Create Migration Guide
```bash
# See MIGRATION_GUIDE.md in next file
```

### Step 4: Update README
```bash
# Add sections for:
# - Config schema overview
# - Single simulation examples
# - Ensemble + ROM pipeline examples
# - Common errors and solutions
```

### Step 5: Test Suite
```bash
# Run all test commands from audit report Section 7
python -m rectsim.cli run --config configs/vicsek_morse_base.yaml --sim.T 50
python -m rectsim.cli run --config configs/strong_clustering.yaml --sim.T 50
# ... etc
```

---

## Notes

1. **Patch order matters**: Apply patches 1-5 first (critical), then 6-8 (polish)
2. **Test incrementally**: After each patch, run quick smoke test
3. **Backward compatibility**: Patches preserve ability to run ROM scripts
4. **Migration required**: All OLD schema configs must be updated manually
5. **Documentation**: Update all example commands in docs to use NEW schema
