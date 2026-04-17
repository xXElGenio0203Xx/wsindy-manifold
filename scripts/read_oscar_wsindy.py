#!/usr/bin/env python3
"""Read WSINDy results from OSCAR output directories."""
import json, os

base = os.path.expanduser("~/wsindy-manifold/oscar_output")
dirs = [
    "NDYN04_gas_VS_wsindy_v3",
    "NDYN06_supernova_wsindy_v3",
    "NDYN07_crystal_wsindy_v3",
    "NDYN07_crystal_VS_wsindy_v3",
]

for d in dirs:
    model_path = os.path.join(base, d, "WSINDy", "multifield_model.json")
    print(f"\n{'='*60}")
    print(f"  {d}")
    print(f"{'='*60}")
    if not os.path.exists(model_path):
        print("  [no model file yet - still running]")
        continue
    model = json.load(open(model_path))
    for field in ["rho", "px", "py"]:
        fm = model[field]
        r2 = fm.get("r2_weak", "?")
        na = fm.get("n_active", "?")
        lam = fm.get("best_lambda", "?")
        r2_s = f"{r2:.6f}" if isinstance(r2, float) else str(r2)
        lam_s = f"{lam:.4e}" if isinstance(lam, float) else str(lam)
        print(f"  {field}: R2_wf={r2_s}, n_active={na}, lambda={lam_s}")
    print()
    for field in ["rho", "px", "py"]:
        fm = model[field]
        active = fm.get("active_terms", [])
        coeffs = fm.get("coefficients", {})
        print(f"  --- {field} active terms ({len(active)}) ---")
        for term in active:
            c = coeffs.get(term, 0.0)
            print(f"    {term:30s}  {c:+.6e}")
