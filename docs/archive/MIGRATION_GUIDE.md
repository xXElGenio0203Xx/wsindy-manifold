# Configuration Schema Migration Guide

## Overview

The rectsim configuration system has been unified to a **single, consistent schema**. If you have existing config files using the OLD schema, they must be migrated to continue working.

**Timeline:**
- **OLD schema** (pre-Dec 2024): `domain`, `particles`, `dynamics`, `integration` sections
- **NEW schema** (current): `model`, `sim`, `params`, `forces`, `noise` sections

---

## Quick Reference: Key Changes

| OLD Schema | NEW Schema | Notes |
|------------|------------|-------|
| `domain.Lx` | `sim.Lx` | Domain parameters moved |
| `domain.Ly` | `sim.Ly` | Domain parameters moved |
| `domain.bc` | `sim.bc` | Domain parameters moved |
| `particles.N` | `sim.N` | Particle count moved |
| `particles.initial_distribution` | `ic.type` | IC specification changed |
| `integration.T` | `sim.T` | Time parameters moved |
| `integration.dt` | `sim.dt` | Time parameters moved |
| `integration.seed` | `seed` | Moved to top level |
| `model.type: "discrete"` | `model: "vicsek_discrete"` | Now a string, not dict |
| `dynamics.alignment` | `params.R` | Alignment radius extracted |
| `dynamics.forces` | `forces` | Top-level section |
| `dynamics.noise.type` | `noise.kind` | Key renamed |
| N/A | `model_config` | NEW section for speed mode |
| N/A | `rom` | NEW section for ROM/MVAR |

---

## Complete Migration Example

### OLD Schema (gentle_clustering.yaml)

```yaml
# OLD - NO LONGER WORKS
domain:
  Lx: 20.0
  Ly: 20.0
  bc: periodic

particles:
  N: 200
  initial_distribution: uniform
  initial_speed: 0.5
  velocity_distribution: random

model:
  type: discrete
  speed_mode: constant

dynamics:
  alignment:
    enabled: true
    radius: 2.0
    rate: 1.0
  
  forces:
    enabled: true
    Cr: 0.5
    Ca: 1.0
    lr: 0.5
    la: 1.0
    rcut_factor: 3.0
    mu_t: 0.5
  
  noise:
    type: gaussian
    eta: 0.3
    match_variance: true
  
  self_propulsion:
    alpha: 1.5
    beta: 0.5

integration:
  T: 100.0
  dt: 0.01
  save_every: 10
  neighbor_rebuild: 5
  integrator: euler
  seed: 42

outputs:
  run_name: gentle_clustering
  order_parameters: true
  animations: true
```

### NEW Schema (migrated)

```yaml
# NEW - CURRENT STANDARD
model: vicsek_discrete  # String, not dict!

sim:
  N: 200                # Moved from particles.N
  Lx: 20.0              # Moved from domain.Lx
  Ly: 20.0              # Moved from domain.Ly
  bc: periodic          # Moved from domain.bc
  T: 100.0              # Moved from integration.T
  dt: 0.01              # Moved from integration.dt
  save_every: 10        # Moved from integration.save_every
  neighbor_rebuild: 5   # Moved from integration.neighbor_rebuild
  integrator: euler     # Moved from integration.integrator

seed: 42                # Moved to top level

model_config:           # NEW section
  speed: 0.5            # From particles.initial_speed
  speed_mode: constant  # From model.speed_mode

params:
  R: 2.0                # From dynamics.alignment.radius
  alpha: 1.5            # From dynamics.self_propulsion.alpha (continuous only)
  beta: 0.5             # From dynamics.self_propulsion.beta (continuous only)

noise:
  kind: gaussian        # Changed from dynamics.noise.type → noise.kind
  eta: 0.3              # From dynamics.noise.eta
  match_variance: true  # From dynamics.noise.match_variance

forces:                 # Promoted to top level
  enabled: true         # From dynamics.forces.enabled
  type: morse           # Implicit
  params:               # Nested params
    Cr: 0.5             # From dynamics.forces.Cr
    Ca: 1.0             # From dynamics.forces.Ca
    lr: 0.5             # From dynamics.forces.lr
    la: 1.0             # From dynamics.forces.la
    rcut_factor: 3.0    # From dynamics.forces.rcut_factor
    mu_t: 0.5           # From dynamics.forces.mu_t

ic:                     # Renamed from particles.initial_distribution
  type: uniform         # uniform, gaussian, ring, cluster

outputs:
  directory: simulations/gentle_clustering  # Renamed from run_name
  order_parameters: true
  animate_traj: false   # NEW: explicit video control
  animate_density: false
  video_ics: 1          # NEW: control video generation
  plot_order_params: true
  order_params_ics: 1
```

---

## Step-by-Step Migration Procedure

### Step 1: Update Top-Level Structure

**OLD:**
```yaml
domain:
  ...
particles:
  ...
model:
  type: discrete
dynamics:
  ...
integration:
  ...
```

**NEW:**
```yaml
model: vicsek_discrete  # or "social_force" for continuous

sim:
  # Combine domain + particles + integration here
  ...

params:
  # Alignment and self-propulsion params
  ...

forces:
  # Top-level forces section
  ...

noise:
  # Top-level noise section
  ...
```

### Step 2: Move Domain Parameters

**OLD:**
```yaml
domain:
  Lx: 20.0
  Ly: 20.0
  bc: periodic
```

**NEW:**
```yaml
sim:
  Lx: 20.0
  Ly: 20.0
  bc: periodic
  # ... other sim params
```

### Step 3: Move Particle Parameters

**OLD:**
```yaml
particles:
  N: 200
  initial_distribution: uniform
  initial_speed: 0.5
```

**NEW:**
```yaml
sim:
  N: 200
  # ... other sim params

model_config:
  speed: 0.5
  speed_mode: constant

ic:
  type: uniform
```

### Step 4: Move Integration Parameters

**OLD:**
```yaml
integration:
  T: 100.0
  dt: 0.01
  save_every: 10
  neighbor_rebuild: 5
  integrator: euler
  seed: 42
```

**NEW:**
```yaml
seed: 42  # Top level

sim:
  T: 100.0
  dt: 0.01
  save_every: 10
  neighbor_rebuild: 5
  integrator: euler
```

### Step 5: Restructure Dynamics Section

**OLD:**
```yaml
dynamics:
  alignment:
    enabled: true
    radius: 2.0
    rate: 1.0  # Continuous only
  
  forces:
    enabled: true
    Cr: 0.5
    Ca: 1.0
    lr: 0.5
    la: 1.0
    rcut_factor: 3.0
    mu_t: 0.5
  
  noise:
    type: gaussian
    eta: 0.3
```

**NEW:**
```yaml
params:
  R: 2.0  # From alignment.radius

forces:
  enabled: true
  type: morse
  params:
    Cr: 0.5
    Ca: 1.0
    lr: 0.5
    la: 1.0
    rcut_factor: 3.0
    mu_t: 0.5

noise:
  kind: gaussian  # Changed from "type"
  eta: 0.3
```

### Step 6: Update Model Type

**OLD:**
```yaml
model:
  type: discrete
  speed_mode: constant
```

**NEW:**
```yaml
model: vicsek_discrete  # String!

model_config:
  speed: 0.5
  speed_mode: constant
```

**Model Type Mapping:**
- `model.type: "discrete"` → `model: "vicsek_discrete"`
- `model.type: "continuous"` → `model: "social_force"` or `"dorsogna"`

### Step 7: Update Outputs Section

**OLD:**
```yaml
outputs:
  run_name: my_simulation
  order_parameters: true
  animations: true
```

**NEW:**
```yaml
outputs:
  directory: simulations/my_simulation  # Explicit path
  order_parameters: true
  plot_order_params: true
  animate_traj: false      # Explicit video control
  animate_density: false
  video_ics: 1             # How many ICs to generate videos for
  order_params_ics: 1      # How many ICs to plot order params for
```

---

## Automated Migration Script

Save this as `migrate_config.py`:

```python
#!/usr/bin/env python3
"""Migrate OLD schema config to NEW schema."""

import sys
import yaml
from pathlib import Path

def migrate_config(old_config):
    """Convert OLD schema to NEW schema."""
    
    new_config = {
        "seed": old_config.get("integration", {}).get("seed", 42),
        "model": "vicsek_discrete" if old_config.get("model", {}).get("type") == "discrete" else "social_force",
    }
    
    # Sim section
    domain = old_config.get("domain", {})
    particles = old_config.get("particles", {})
    integration = old_config.get("integration", {})
    
    new_config["sim"] = {
        "N": particles.get("N", 200),
        "Lx": domain.get("Lx", 20.0),
        "Ly": domain.get("Ly", 20.0),
        "bc": domain.get("bc", "periodic"),
        "T": integration.get("T", 100.0),
        "dt": integration.get("dt", 0.01),
        "save_every": integration.get("save_every", 10),
        "neighbor_rebuild": integration.get("neighbor_rebuild", 5),
        "integrator": integration.get("integrator", "euler"),
    }
    
    # Model config
    model = old_config.get("model", {})
    new_config["model_config"] = {
        "speed": particles.get("initial_speed", 0.5),
        "speed_mode": model.get("speed_mode", "constant"),
    }
    
    # Params
    dynamics = old_config.get("dynamics", {})
    alignment = dynamics.get("alignment", {})
    self_prop = dynamics.get("self_propulsion", {})
    
    new_config["params"] = {
        "R": alignment.get("radius", 2.0),
    }
    if "alpha" in self_prop:
        new_config["params"]["alpha"] = self_prop["alpha"]
    if "beta" in self_prop:
        new_config["params"]["beta"] = self_prop["beta"]
    
    # Noise
    old_noise = dynamics.get("noise", {})
    new_config["noise"] = {
        "kind": old_noise.get("type", "gaussian"),  # type → kind
        "eta": old_noise.get("eta", 0.3),
        "match_variance": old_noise.get("match_variance", True),
    }
    
    # Forces
    old_forces = dynamics.get("forces", {})
    if old_forces.get("enabled", False):
        new_config["forces"] = {
            "enabled": True,
            "type": "morse",
            "params": {
                "Cr": old_forces.get("Cr", 2.0),
                "Ca": old_forces.get("Ca", 1.0),
                "lr": old_forces.get("lr", 0.9),
                "la": old_forces.get("la", 1.0),
                "rcut_factor": old_forces.get("rcut_factor", 3.0),
                "mu_t": old_forces.get("mu_t", 1.0),
            },
        }
    
    # IC
    new_config["ic"] = {
        "type": particles.get("initial_distribution", "uniform"),
    }
    
    # Outputs
    old_outputs = old_config.get("outputs", {})
    run_name = old_outputs.get("run_name", "simulation")
    new_config["outputs"] = {
        "directory": f"simulations/{run_name}",
        "order_parameters": old_outputs.get("order_parameters", True),
        "plot_order_params": True,
        "animate_traj": old_outputs.get("animations", False),
        "animate_density": old_outputs.get("animations", False),
        "video_ics": 1,
        "order_params_ics": 1,
    }
    
    return new_config


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python migrate_config.py OLD_CONFIG.yaml NEW_CONFIG.yaml")
        sys.exit(1)
    
    old_path = Path(sys.argv[1])
    new_path = Path(sys.argv[2])
    
    with open(old_path) as f:
        old_config = yaml.safe_load(f)
    
    new_config = migrate_config(old_config)
    
    with open(new_path, "w") as f:
        yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Migrated {old_path} → {new_path}")
```

**Usage:**
```bash
python migrate_config.py configs/gentle_clustering.yaml configs/gentle_clustering_new.yaml
```

---

## Common Migration Errors

### Error 1: "Missing required config key: 'sim'"
**Cause:** Using OLD schema with NEW code  
**Fix:** Migrate config file using guide above

### Error 2: "KeyError: 'particles'"
**Cause:** Code trying to access OLD schema keys  
**Fix:** Update code to use `config["sim"]["N"]` instead of `config["particles"]["N"]`

### Error 3: "Invalid noise.kind: 'None'"
**Cause:** Missing noise configuration  
**Fix:** Add noise section to config:
```yaml
noise:
  kind: gaussian
  eta: 0.3
```

### Error 4: "'model' is a dict, not string"
**Cause:** Using OLD `model.type` format  
**Fix:** Change from:
```yaml
model:
  type: discrete
```
to:
```yaml
model: vicsek_discrete
```

---

## Testing After Migration

```bash
# 1. Syntax check
python -c "import yaml; yaml.safe_load(open('configs/my_new_config.yaml'))"

# 2. Schema validation (NEW code required)
python -m rectsim.config --validate configs/my_new_config.yaml

# 3. Dry run
python -m rectsim.cli run --config configs/my_new_config.yaml --sim.T 1.0

# 4. Full run
python -m rectsim.cli run --config configs/my_new_config.yaml
```

---

## FAQ

**Q: Do I need to migrate all configs at once?**  
A: No, but NEW code will reject OLD configs with a clear error message.

**Q: Will OLD configs ever work again?**  
A: No, OLD schema support is permanently removed. Migrate to NEW schema.

**Q: What if my config has custom fields?**  
A: Custom fields are preserved. Just migrate the standard fields.

**Q: Can I use CLI overrides during migration?**  
A: Yes! CLI overrides work with NEW schema: `--sim.N 100 --noise.eta 0.5`

**Q: Where can I find NEW schema examples?**  
A: See `configs/vicsek_morse_base.yaml` and `configs/strong_clustering.yaml`

---

## Migration Checklist

- [ ] Backup OLD config file
- [ ] Create NEW config with migrated fields
- [ ] Update `model` to string (not dict)
- [ ] Move domain → sim
- [ ] Move particles.N → sim.N
- [ ] Move integration → sim
- [ ] Update dynamics → params/forces/noise
- [ ] Change noise.type → noise.kind
- [ ] Add model_config section
- [ ] Update outputs section
- [ ] Test with syntax checker
- [ ] Run dry run with --sim.T 1.0
- [ ] Verify output folder structure
- [ ] Delete OLD config or add ".old" suffix

---

**Last Updated:** December 2024  
**Schema Version:** NEW (2024.12+)  
**Support:** See CODEBASE_AUDIT_REPORT.md for technical details
