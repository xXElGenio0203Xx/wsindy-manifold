import json, os
exps = ["ABL8_N200_sqrt_simplex_align_H300_v2", "DYN1_gentle_v2"]
for e in exps:
    p = f"oscar_output/{e}/summary.json"
    if os.path.exists(p):
        s = json.load(open(p))
        r2 = s.get("mean_r2_test", "?")
        rho = s.get("mvar", {}).get("spectral_radius", "?")
        nt = s.get("n_train", "?")
        print(f"{e}: R2={r2}, n_train={nt}, rho={rho}")
    else:
        print(f"{e}: not found")
