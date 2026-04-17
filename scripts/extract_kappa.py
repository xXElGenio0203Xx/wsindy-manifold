#!/usr/bin/env python3
import json, glob, os

files = sorted(glob.glob(os.path.expanduser(
    "~/wsindy-manifold/oscar_output/*_wsindy_v3/WSINDy/multifield_diagnostics.json"
)))
for f in files:
    regime = f.split("/oscar_output/")[1].split("_wsindy_v3")[0]
    d = json.load(open(f))
    fd = d.get("fit_diagnostics", {})
    rho_k = fd.get("rho", {}).get("condition_number", "N/A")
    px_k = fd.get("px", {}).get("condition_number", "N/A")
    py_k = fd.get("py", {}).get("condition_number", "N/A")
    print(f"{regime} | rho={rho_k} | px={px_k} | py={py_k}")
