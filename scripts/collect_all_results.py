#!/usr/bin/env python3
"""Collect ALL results from oscar_output for thesis table population."""
import csv, os, json, sys

base = os.path.expanduser("~/scratch/oscar_output")

# ── 1. WSINDy coefficients & weak R² ──
print("=" * 80)
print("WSINDY RESULTS")
print("=" * 80)
for d in sorted(os.listdir(base)):
    ws_dir = os.path.join(base, d, "WSINDy")
    mf = os.path.join(ws_dir, "multifield_model.json")
    if not os.path.exists(mf):
        continue
    with open(mf) as f:
        model = json.load(f)
    tr = os.path.join(ws_dir, "test_results.csv")
    weak_r2 = {}
    forecast = ""
    if os.path.exists(tr):
        with open(tr) as f:
            rows = list(csv.DictReader(f))
        if rows:
            weak_r2["rho"] = rows[0].get("weak_r2_rho", "")
            weak_r2["px"] = rows[0].get("weak_r2_px", "")
            weak_r2["py"] = rows[0].get("weak_r2_py", "")
            forecast = rows[0].get("forecast_status", "")
    print(f"\n--- {d} ---")
    if "equations" in model:
        for eq_name, eq_data in model["equations"].items():
            print(f"  {eq_name}: {eq_data.get('text', 'N/A')}")
            coeffs = eq_data.get("coefficients", {})
            for term, coeff in coeffs.items():
                if isinstance(coeff, (int, float)):
                    print(f"    {term}: {coeff:.6e}")
    ws = ", ".join(f"{k}={v}" for k, v in weak_r2.items() if v)
    print(f"  weak_R2: {ws}")
    print(f"  forecast: {forecast}")
    # selected ell
    diag = os.path.join(ws_dir, "multifield_diagnostics.json")
    if os.path.exists(diag):
        with open(diag) as f:
            dd = json.load(f)
        ell = dd.get("selected_ell", dd.get("ell_star", ""))
        bic = dd.get("bic", dd.get("best_bic", ""))
        nactive = dd.get("n_active_terms", "")
        print(f"  ell_star={ell}, n_active={nactive}")

# ── 2. MVAR + LSTM R² ──
print("\n" + "=" * 80)
print("MVAR & LSTM R2")
print("=" * 80)
for d in sorted(os.listdir(base)):
    exp_dir = os.path.join(base, d)
    parts = []
    for model_name in ["MVAR", "LSTM"]:
        model_csv = os.path.join(exp_dir, model_name, "test_results.csv")
        if os.path.exists(model_csv):
            with open(model_csv) as f:
                rows = list(csv.DictReader(f))
            r2s = []
            for r in rows:
                val = r.get("r2_reconstructed", "")
                if val:
                    try:
                        r2s.append(float(val))
                    except ValueError:
                        pass
            if r2s:
                mean_r2 = sum(r2s) / len(r2s)
                mp = rows[0].get("mass_postprocess", "?") if rows else "?"
                extra = f" (mp={mp})" if model_name == "LSTM" else ""
                parts.append(f"{model_name}={mean_r2:.4f}{extra}")
    if parts:
        print(f"{d}: {', '.join(parts)}")

# ── 3. Timings ──
print("\n" + "=" * 80)
print("TIMINGS")
print("=" * 80)
for d in sorted(os.listdir(base)):
    sj = os.path.join(base, d, "summary.json")
    if not os.path.exists(sj):
        continue
    with open(sj) as f:
        s = json.load(f)
    models = s.get("models_enabled", {})
    active = [k for k, v in models.items() if v]
    t = s.get("total_time_minutes", 0)
    print(f"{d}: {t:.0f}min, models={active}")

# ── 4. WSINDy runtime profiles ──
print("\n" + "=" * 80)
print("WSINDY RUNTIME PROFILES")
print("=" * 80)
for d in sorted(os.listdir(base)):
    rp = os.path.join(base, d, "WSINDy", "runtime_profile.json")
    if not os.path.exists(rp):
        continue
    with open(rp) as f:
        r = json.load(f)
    train_s = r.get("training_time_seconds", 0)
    print(f"{d}: WSINDy training={train_s:.0f}s ({train_s/60:.0f}min)")

# ── 5. Noise sweep specific ──
print("\n" + "=" * 80)
print("NOISE SWEEP (eta experiments)")
print("=" * 80)
for d in sorted(os.listdir(base)):
    if "eta" not in d:
        continue
    ws_dir = os.path.join(base, d, "WSINDy")
    mf = os.path.join(ws_dir, "multifield_model.json")
    if not os.path.exists(mf):
        print(f"{d}: NO WSINDy model")
        continue
    with open(mf) as f:
        model = json.load(f)
    tr = os.path.join(ws_dir, "test_results.csv")
    if os.path.exists(tr):
        with open(tr) as f:
            rows = list(csv.DictReader(f))
        if rows:
            wr = rows[0].get("weak_r2_rho", "?")
            nact = len([eq for eq in model.get("equations", {}).values() 
                       for c in eq.get("coefficients", {}).values()])
            print(f"{d}: weak_r2_rho={wr}, n_terms={nact}")

# ── 6. N-convergence ──
print("\n" + "=" * 80)
print("N-CONVERGENCE")
print("=" * 80)
for d in sorted(os.listdir(base)):
    if "_N0" not in d:
        continue
    ws_dir = os.path.join(base, d, "WSINDy")
    mf = os.path.join(ws_dir, "multifield_model.json")
    if os.path.exists(mf):
        with open(mf) as f:
            model = json.load(f)
        tr = os.path.join(ws_dir, "test_results.csv")
        wr = "?"
        if os.path.exists(tr):
            with open(tr) as f:
                rows = list(csv.DictReader(f))
            if rows:
                wr = rows[0].get("weak_r2_rho", "?")
        print(f"{d}: weak_r2_rho={wr}")
    else:
        print(f"{d}: NO WSINDy")
