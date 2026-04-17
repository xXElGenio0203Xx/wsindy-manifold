import csv, os

exps = [
    ("NDYN04_gas_tier1_w5",        "gas"),
    ("NDYN04_gas_tier1_bic",       "gas"),
    ("NDYN04_gas_VS_tier1_w5",     "gas_VS"),
    ("NDYN04_gas_VS_tier1_bic",    "gas_VS"),
    ("NDYN05_blackhole_tier1_w5",  "BH"),
    ("NDYN05_blackhole_tier1_bic", "BH"),
    ("NDYN05_blackhole_VS_tier1_w5",  "BH_VS"),
    ("NDYN05_blackhole_VS_tier1_bic", "BH_VS"),
    ("NDYN06_supernova_tier1_w5",  "SN"),
    ("NDYN06_supernova_tier1_bic", "SN"),
]

THESIS = {
    "gas":    {"mvar": 0.995, "lstm": 0.632},
    "BH":     {"mvar": 0.990, "lstm": 0.989},
    "SN":     {"mvar": 0.545, "lstm": 0.508},
    "gas_VS": {"mvar": 0.558, "lstm": None},
    "BH_VS":  {"mvar": -0.433, "lstm": None},
}

print(f"{'Experiment':<42} {'Regime':<8} {'MVAR R2':>10} {'LSTM R2':>10} | {'Thesis MVAR':>12} {'Thesis LSTM':>12}")
print("=" * 105)
for exp, regime in exps:
    results = {}
    for model in ["MVAR", "LSTM"]:
        path = os.path.join("oscar_output", exp, model, "test_results.csv")
        try:
            with open(path) as f:
                rows = list(csv.DictReader(f))
            r2s = [float(r["r2_reconstructed"]) for r in rows]
            results[model.lower()] = sum(r2s) / len(r2s)
        except Exception:
            results[model.lower()] = None

    tv = THESIS.get(regime, {})
    tm = tv.get("mvar")
    tl = tv.get("lstm")
    mr = results.get("mvar")
    lr = results.get("lstm")

    m_str = "{:.4f}".format(mr) if mr is not None else "---"
    l_str = "{:.4f}".format(lr) if lr is not None else "---"
    tm_str = "{:.3f}".format(tm) if tm is not None else "---"
    tl_str = "{:.3f}".format(tl) if tl is not None else "---"

    print("{:<42} {:<8} {:>10} {:>10} | {:>12} {:>12}".format(exp, regime, m_str, l_str, tm_str, tl_str))
