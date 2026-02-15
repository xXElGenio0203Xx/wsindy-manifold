#!/usr/bin/env python3
import json, os, glob
results = []
for sfile in sorted(glob.glob("oscar_output/*/summary.json")):
    name = os.path.basename(os.path.dirname(sfile))
    try:
        with open(sfile) as f:
            s = json.load(f)
        mvar = s.get("mvar", {})
        if mvar and "mean_r2_test" in mvar:
            results.append(("MVAR", name, mvar["mean_r2_test"], mvar.get("mean_r2_1step_test","?"), mvar.get("spectral_radius","?")))
        lstm = s.get("lstm", {})
        if lstm and "mean_r2_test" in lstm:
            results.append(("LSTM", name, lstm["mean_r2_test"], lstm.get("mean_r2_1step_test","?"), "N/A"))
    except:
        pass
mvar_s = sorted([r for r in results if r[0]=="MVAR"], key=lambda x: -x[2])
lstm_s = sorted([r for r in results if r[0]=="LSTM"], key=lambda x: -x[2])
print("="*90)
print("TOP MVAR by R2_roll")
print("="*90)
for i,(m,name,r2r,r2_1s,rho) in enumerate(mvar_s[:15]):
    print(f"  {i+1:2d}. {name:55s} R2roll={r2r:+.4f}  R2_1s={r2_1s}  rho={rho}")
print()
print("="*90)
print("TOP LSTM by R2_roll")
print("="*90)
for i,(m,name,r2r,r2_1s,rho) in enumerate(lstm_s[:15]):
    print(f"  {i+1:2d}. {name:55s} R2roll={r2r:+.4f}  R2_1s={r2_1s}")
print(f"\nTotal MVAR: {len(mvar_s)}, LSTM: {len(lstm_s)}")
