#!/usr/bin/env python3
"""Extract WSINDy + MVAR metrics from all summary.json files."""
import json, os, sys

base = "oscar_output"
for d in sorted(os.listdir(base)):
    if not d.endswith("_wsindy_v3"):
        continue
    sj = os.path.join(base, d, "summary.json")
    if not os.path.exists(sj):
        continue
    with open(sj) as f:
        s = json.load(f)
    name = d.replace("_wsindy_v3", "")

    # MVAR
    mvar = s.get("mvar", {})
    mvar_r2 = mvar.get("mean_r2_test", "N/A")
    mvar_lag = mvar.get("lag_used", "N/A")

    print(f"=== {name} ===")
    print(f"  MVAR: R2_test={mvar_r2}, lag={mvar_lag}")

    # Evaluation section
    ev = s.get("evaluation", {})
    for field in ["rho", "px", "py"]:
        fd = ev.get(field, {})
        if fd:
            r2_test = fd.get("r2_test", fd.get("r2_snapshot_mean", "N/A"))
            rmse = fd.get("rmse_test", fd.get("rmse", "N/A"))
            print(f"  eval.{field}: r2_test={r2_test}, rmse={rmse}")

    # WSINDy model info
    ws_model = None
    ws_diag = None
    # Check WSINDy multifield_model.json
    mf = os.path.join(base, d, "WSINDy", "multifield_model.json")
    md = os.path.join(base, d, "WSINDy", "multifield_diagnostics.json")
    if os.path.exists(mf):
        with open(mf) as f:
            ws_model = json.load(f)
    if os.path.exists(md):
        with open(md) as f:
            ws_diag = json.load(f)

    if ws_model:
        for field_key in ["rho", "px", "py"]:
            fm = ws_model.get(field_key, {})
            if not fm:
                continue
            active = fm.get("active_terms", [])
            n_active = len(active) if isinstance(active, list) else active
            coeffs = fm.get("coefficients", {})
            r2_wf = fm.get("r2_wf", "N/A")
            bic = fm.get("bic", "N/A")
            print(f"  WSINDy {field_key}: n_active={n_active}, R2_wf={r2_wf}, BIC={bic}")
            if isinstance(active, list):
                for t in active[:5]:
                    if isinstance(t, dict):
                        tname = t.get("term", t.get("name", str(t)))
                        tcoeff = t.get("coefficient", t.get("coeff", ""))
                        print(f"    term: {tname}  coeff={tcoeff}")
                    else:
                        print(f"    term: {t}")

    if ws_diag:
        for field_key in ["rho", "px", "py"]:
            fd = ws_diag.get(field_key, {})
            if fd:
                bs = fd.get("bootstrap", {})
                incl_prob = bs.get("inclusion_probabilities", {})
                if incl_prob:
                    n_stable = sum(1 for v in incl_prob.values() if v > 0.9)
                    print(f"  WSINDy diag {field_key}: {n_stable} terms with P>0.9")

    print()
