#!/usr/bin/env python3
"""Generate N=10 and N=20 configs for N-convergence study."""
import yaml

with open("configs/n_convergence/NDYN08_pure_vicsek_N0050.yaml") as f:
    base = yaml.safe_load(f)

for n in [10, 20]:
    cfg = yaml.safe_load(yaml.dump(base))
    cfg["sim"]["N"] = n
    cfg["experiment_name"] = "NDYN08_pure_vicsek_N{:04d}".format(n)
    cfg["meta"]["is_baseline"] = False

    out = "configs/n_convergence/NDYN08_pure_vicsek_N{:04d}.yaml".format(n)
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print("Created " + out)

    ws_cfg = yaml.safe_load(yaml.dump(cfg))
    ws_cfg["experiment_name"] = "NDYN08_pure_vicsek_N{:04d}_ws".format(n)
    ws_cfg["rom"]["models"]["mvar"]["enabled"] = False
    ws_cfg["rom"]["models"]["lstm"]["enabled"] = False
    ws_cfg["wsindy"]["bootstrap"]["B"] = 0
    ws_cfg["wsindy"]["bootstrap"]["enabled"] = False
    ws_cfg["wsindy"]["model_selection"]["n_ell"] = 12
    ws_cfg["wsindy"]["model_selection"]["p"] = [2, 2, 2]

    out_ws = "configs/n_convergence_ws/NDYN08_pure_vicsek_N{:04d}.yaml".format(n)
    with open(out_ws, "w") as f:
        yaml.dump(ws_cfg, f, default_flow_style=False, sort_keys=False)
    print("Created " + out_ws)

print("Done")
