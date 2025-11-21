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
    
    if not old_path.exists():
        print(f"Error: {old_path} not found")
        sys.exit(1)
    
    with open(old_path) as f:
        old_config = yaml.safe_load(f)
    
    new_config = migrate_config(old_config)
    
    with open(new_path, "w") as f:
        yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Migrated {old_path} → {new_path}")
