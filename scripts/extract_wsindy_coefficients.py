#!/usr/bin/env python3
"""Extract WSINDy coefficients for thesis tables."""
import json, os, csv

base = os.path.expanduser("~/scratch/oscar_output")

for d in sorted(os.listdir(base)):
    mf = os.path.join(base, d, "WSINDy", "multifield_model.json")
    if not os.path.exists(mf):
        continue
    with open(mf) as f:
        m = json.load(f)
    meta = m.get("metadata", {})
    fd = meta.get("fit_diagnostics", {})
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {d}")
    print(f"{'='*60}")
    
    for field_name in ["rho", "px", "py"]:
        fd_field = fd.get(field_name, {})
        cond = fd_field.get("condition_number", "N/A")
        
        # Coefficients are at top level of metadata
        # Try different structures
        coeffs = {}
        r2w = ""
        nact = ""
        
        # Check flat structure: metadata.fit_diagnostics.{field}.coefficients
        if "coefficients" in fd_field:
            coeffs = fd_field["coefficients"]
        
        # Also search at root level
        for key in m:
            if key in ["metadata"]:
                continue
            if isinstance(m[key], dict) and "coefficients" in m[key]:
                # This might be the equation for this field
                pass
        
        # The real structure from grep output: coefficients are directly under root keys
        # Check if field_name or rho/px/py exists as top-level keys
        for k, v in m.items():
            if k == field_name and isinstance(v, dict):
                coeffs = v.get("coefficients", coeffs)
                r2w = v.get("r2_weak", r2w)
                nact = v.get("n_active", nact)
        
        # If still empty, the metadata has different structure
        # Let's just dump all top-level keys
        if not coeffs:
            for k, v in m.items():
                if k not in ["metadata"] and isinstance(v, dict):
                    if "r2_weak" in v and field_name in k:
                        coeffs = v.get("coefficients", {})
                        r2w = v.get("r2_weak", "")
                        nact = v.get("n_active", "")
        
        print(f"\n  {field_name}_t:")
        print(f"    r2_weak = {r2w}")
        print(f"    n_active = {nact}")
        print(f"    condition_number = {cond}")
        if coeffs:
            for term, val in coeffs.items():
                if isinstance(val, (int, float)):
                    print(f"    {term} = {val:.6e}")
                else:
                    print(f"    {term} = {val}")
        else:
            print(f"    (no coefficients found)")
    
    # Selected ell
    tr = os.path.join(base, d, "WSINDy", "test_results.csv")
    if os.path.exists(tr):
        with open(tr) as f:
            rows = list(csv.DictReader(f))
        if rows:
            ell = rows[0].get("selected_ell", "N/A")
            print(f"\n  selected_ell = {ell}")

# Also dump top-level keys for first experiment to understand structure
first = None
for d in sorted(os.listdir(base)):
    mf = os.path.join(base, d, "WSINDy", "multifield_model.json")
    if os.path.exists(mf):
        first = mf
        break
if first:
    with open(first) as f:
        m = json.load(f)
    print(f"\n\n{'='*60}")
    print("TOP-LEVEL KEYS OF FIRST MODEL:")
    print(f"{'='*60}")
    for k in m:
        print(f"  {k}: {type(m[k]).__name__}", end="")
        if isinstance(m[k], dict):
            print(f" keys={list(m[k].keys())[:10]}")
        else:
            print()
