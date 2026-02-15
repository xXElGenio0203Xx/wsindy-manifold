#!/usr/bin/env python3
"""Generate 6 Suite Y configs (H100 mid-horizon) from Suite X templates."""
import yaml, copy

def load_cfg(name):
    with open(f"configs/{name}.yaml") as f:
        lines = f.readlines()
    content = "".join(lines[2:])  # skip --- and comment
    return yaml.safe_load(content)

templates = {
    "V1":   load_cfg("X1_V1_raw_H37"),
    "V3.3": load_cfg("X5_V33_raw_H37"),
    "V3.4": load_cfg("X9_V34_raw_H37"),
}

experiments = [
    ("Y1_V1_raw_H100",          "V1",   "raw",  "none"),
    ("Y2_V1_sqrtSimplex_H100",  "V1",   "sqrt", "simplex"),
    ("Y3_V33_raw_H100",         "V3.3", "raw",  "none"),
    ("Y4_V33_sqrtSimplex_H100", "V3.3", "sqrt", "simplex"),
    ("Y5_V34_raw_H100",         "V3.4", "raw",  "none"),
    ("Y6_V34_sqrtSimplex_H100", "V3.4", "sqrt", "simplex"),
]

for name, regime, transform, postprocess in experiments:
    cfg = copy.deepcopy(templates[regime])
    cfg["experiment_name"] = name
    cfg["test_sim"] = {"T": 12.6}

    cfg["rom"]["density_transform"] = transform
    if transform == "sqrt":
        cfg["rom"]["density_transform_eps"] = 1e-10
    elif "density_transform_eps" in cfg["rom"]:
        del cfg["rom"]["density_transform_eps"]

    if postprocess != "none":
        cfg["eval"]["mass_postprocess"] = postprocess
    elif "mass_postprocess" in cfg["eval"]:
        del cfg["eval"]["mass_postprocess"]

    path = f"configs/{name}.yaml"
    with open(path, "w") as f:
        f.write("---\n")
        label = "simplex postprocess" if postprocess == "simplex" else "no postprocess"
        f.write(f"# Suite Y: {regime} regime, {transform} transform, {label}, H100\n")
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"  created {path}")

print(f"\n==> {len(experiments)} Suite Y configs created")
