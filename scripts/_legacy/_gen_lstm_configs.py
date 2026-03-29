#!/usr/bin/env python3
"""Generate LST2-LST8 configs from LST1 template with variations."""
import yaml
from pathlib import Path

# Load LST1 as template
with open("configs/LST1_raw_none_align_h32_L1.yaml") as f:
    base = yaml.safe_load(f)

experiments = {
    "LST2_raw_none_align_h64_L2": {
        "desc": "LSTM medium: raw, none, align, hidden=64, 2 layers",
        "lstm": {"hidden_units": 64, "num_layers": 2, "dropout": 0.1},
    },
    "LST3_raw_none_align_h64_L2_multistep": {
        "desc": "LSTM medium + multistep loss: raw, none, align, hidden=64, 2 layers, k=5",
        "lstm": {"hidden_units": 64, "num_layers": 2, "dropout": 0.1,
                 "multistep_loss": True, "multistep_k": 5, "multistep_alpha": 0.3},
    },
    "LST4_sqrt_simplex_align_h64_L2": {
        "desc": "LSTM medium + sqrt/simplex/align: hidden=64, 2 layers",
        "rom": {"density_transform": "sqrt", "mass_postprocess": "simplex"},
        "lstm": {"hidden_units": 64, "num_layers": 2, "dropout": 0.1},
    },
    "LST5_raw_none_noAlign_h64_L2": {
        "desc": "LSTM NO alignment: raw, none, noAlign, hidden=64, 2 layers",
        "rom": {"shift_align": False},
        "lstm": {"hidden_units": 64, "num_layers": 2, "dropout": 0.1},
    },
    "LST6_sqrt_none_noAlign_h64_L2": {
        "desc": "LSTM sqrt/noAlign (MVAR failed badly here): hidden=64, 2 layers",
        "rom": {"density_transform": "sqrt", "shift_align": False},
        "lstm": {"hidden_units": 64, "num_layers": 2, "dropout": 0.1},
    },
    "LST7_raw_none_align_h128_L2": {
        "desc": "LSTM large: raw, none, align, hidden=128, 2 layers",
        "lstm": {"hidden_units": 128, "num_layers": 2, "dropout": 0.15},
    },
    "LST8_raw_none_align_h64_L2_ss": {
        "desc": "LSTM medium + scheduled sampling: raw, none, align, hidden=64, 2 layers",
        "lstm": {"hidden_units": 64, "num_layers": 2, "dropout": 0.1,
                 "scheduled_sampling": True,
                 "ss_warmup": 20, "ss_phase1_end": 150, "ss_phase1_ratio": 0.3,
                 "ss_phase2_end": 250, "ss_max_ratio": 0.5,
                 "max_epochs": 400, "patience": 50},
    },
}

for name, overrides in experiments.items():
    cfg = yaml.safe_load(yaml.dump(base))  # deep copy
    cfg["experiment_name"] = name

    # ROM-level overrides
    if "rom" in overrides:
        for k, v in overrides["rom"].items():
            cfg["rom"][k] = v

    # LSTM-specific overrides
    if "lstm" in overrides:
        for k, v in overrides["lstm"].items():
            cfg["rom"]["models"]["lstm"][k] = v

    # Update header comment
    path = Path(f"configs/{name}.yaml")
    with open(path, "w") as f:
        f.write(f"---\n# {'='*76}\n")
        f.write(f"# {name}\n")
        f.write(f"# {'='*76}\n")
        f.write(f"# {overrides['desc']}\n")
        f.write(f"# {'='*76}\n\n")
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Created {path}")
    h = cfg["rom"]["models"]["lstm"]["hidden_units"]
    nl = cfg["rom"]["models"]["lstm"]["num_layers"]
    dt = cfg["rom"].get("density_transform", "raw")
    mp = cfg["rom"].get("mass_postprocess", "none")
    sa = cfg["rom"].get("shift_align", True)
    ms = cfg["rom"]["models"]["lstm"].get("multistep_loss", False)
    ss = cfg["rom"]["models"]["lstm"].get("scheduled_sampling", False)
    print(f"  h={h}, L={nl}, transform={dt}, mass_pp={mp}, align={sa}, multistep={ms}, sched_samp={ss}")

print(f"\nDone! Created {len(experiments)} configs.")
