#!/usr/bin/env python3
"""Quick reader for suite experiment results."""
import json, glob, os, sys

pattern = sys.argv[1] if len(sys.argv) > 1 else "oscar_output/suite_*/summary.json"

for s in sorted(glob.glob(pattern)):
    name = os.path.basename(os.path.dirname(s))
    d = json.load(open(s))
    
    # Find model results
    m = d.get("mvar", d.get("lstm", {}))
    r2 = m.get("mean_r2_test", "?")
    r2_1s = m.get("mean_r2_1step_test", "?")
    rho = m.get("spectral_radius", "?")
    neg = m.get("mean_negativity_frac", "?")
    
    # POD info (new format has pod dict, old format has r_pod at top level)
    pod = d.get("pod", {})
    pod_d = pod.get("r_pod", d.get("r_pod", "?"))
    pod_e = pod.get("energy_captured", "?")
    xform = pod.get("density_transform", "raw")
    clamp = d.get("clamp_mode", "?")
    
    # Format
    parts = [f"{name:50s}"]
    if isinstance(r2, (int, float)):
        parts.append(f"R2_roll={r2:+.4f}")
    else:
        parts.append(f"R2_roll={r2}")
    if isinstance(r2_1s, (int, float)):
        parts.append(f"R2_1step={r2_1s:+.4f}")
    if isinstance(rho, (int, float)):
        parts.append(f"rho={rho:.4f}")
    if isinstance(neg, (int, float)):
        parts.append(f"neg={neg:.4f}")
    parts.append(f"d={pod_d}")
    if isinstance(pod_e, (int, float)):
        parts.append(f"E={pod_e:.1%}")
    if xform != "raw":
        parts.append(f"xform={xform}")
    if clamp not in ("?", None):
        parts.append(f"clamp={clamp}")
    
    print("  ".join(parts))
