#!/usr/bin/env python3
"""Quick script to print summary of all completed experiments on Oscar."""
import json, os, glob

groups = {
    "AL (Alignment ablation)": "AL*",
    "VDYN (Variable speed dynamics)": "VDYN*",
    "LST (LSTM experiments)": "LST*",
    "S (Stability experiments)": "S[12]*",
    "ST (Stability threshold)": "ST*",
    "X (Cross-config)": "X*",
    "XABL (Extended ablation)": "XABL*",
    "Y (Knee-horizon)": "Y*",
    "DYN (Dynamics)": "DYN*",
    "K (Knee)": "K*",
    "ABL (Ablation)": "ABL*",
    "REP (Repulsive)": "REP*",
    "DEG (Degradation)": "DEG*",
    "CUR (Curriculum)": "CUR*",
}

for label, pattern in sorted(groups.items()):
    dirs = sorted(glob.glob(f"oscar_output/{pattern}/summary.json"))
    if not dirs:
        continue
    print(f"\n{'='*80}")
    print(f"  {label}  ({len(dirs)} experiments)")
    print(f"{'='*80}")
    for sf in dirs:
        s = json.load(open(sf))
        exp = os.path.basename(os.path.dirname(sf))
        mvar_r2 = s.get("mvar", {}).get("mean_r2_test", None)
        lstm_r2 = s.get("lstm", {}).get("mean_r2_test", None)
        n_tr = s.get("n_train", "?")
        n_te = s.get("n_test", "?")
        rpod = s.get("r_pod", s.get("pod", {}).get("r_pod", "?"))
        t = s.get("total_time_minutes", 0)
        mr = f"{mvar_r2:.4f}" if mvar_r2 is not None else "  --  "
        lr = f"{lstm_r2:.4f}" if lstm_r2 is not None else "  --  "
        print(f"  {exp:<52s} d={str(rpod):<3s} MVAR={mr}  LSTM={lr}  {t:.0f}m")
