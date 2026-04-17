import csv, os, yaml
from pathlib import Path
from collections import defaultdict

# Current thesis LSTM values
THESIS_LSTM = {
    "gas": 0.632,
    "blackhole": 0.989,
    "supernova": 0.508,
    "pure_vicsek": None,
    "gas_VS": -0.533,
    "blackhole_VS": -0.656,
    "supernova_VS": None,
}

results = []
for root, dirs, files in os.walk("oscar_output"):
    if "test_results.csv" in files and root.endswith("/LSTM"):
        csv_path = os.path.join(root, "test_results.csv")
        exp_dir = root[:-5]  # strip "/LSTM"
        exp_name = exp_dir.split("/")[-1] if "/" in exp_dir else exp_dir

        try:
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            r2s = [float(r["r2_reconstructed"]) for r in rows]
            r2_mean = sum(r2s) / len(r2s)
        except Exception:
            continue

        config = {}
        for cfg_name in ["config_used.yaml", "config.yaml"]:
            cfg_path = os.path.join(exp_dir, cfg_name)
            if os.path.exists(cfg_path):
                try:
                    with open(cfg_path) as f:
                        config = yaml.safe_load(f) or {}
                except Exception:
                    pass
                break

        mp = config.get("mass_postprocess", "?")
        N = config.get("N", "?")

        # Classify regime
        regime = "other"
        name_lower = exp_name.lower()
        if "pure_vicsek" in name_lower or "purevicsek" in name_lower:
            regime = "pure_vicsek"
        elif "supernova" in name_lower and ("_vs_" in name_lower or name_lower.endswith("_vs") or "_VS_" in exp_name or exp_name.endswith("_VS")):
            regime = "supernova_VS"
        elif "supernova" in name_lower:
            regime = "supernova"
        elif "blackhole" in name_lower and ("_vs_" in name_lower or name_lower.endswith("_vs") or "_VS_" in exp_name or exp_name.endswith("_VS")):
            regime = "blackhole_VS"
        elif "blackhole" in name_lower:
            regime = "blackhole"
        elif "gas" in name_lower and ("_vs_" in name_lower or name_lower.endswith("_vs") or "_VS_" in exp_name or exp_name.endswith("_VS")):
            regime = "gas_VS"
        elif "gas" in name_lower:
            regime = "gas"

        results.append({
            "exp": exp_name,
            "regime": regime,
            "r2": r2_mean,
            "N": N,
            "mass_pp": mp,
            "path": csv_path,
        })

results.sort(key=lambda x: (x["regime"], -x["r2"]))

print("{:<15} {:<55} {:>8} {:>5} {:<15}".format("Regime", "Experiment", "R2", "N", "mass_pp"))
print("=" * 105)
current_regime = None
for r in results:
    if r["regime"] != current_regime:
        current_regime = r["regime"]
        thesis_val = THESIS_LSTM.get(current_regime)
        if thesis_val is not None:
            print("  [Thesis LSTM R2 = {}]".format(thesis_val))
        elif current_regime in THESIS_LSTM:
            print("  [Thesis LSTM R2 = ---]")
        else:
            print("  [Not a thesis regime]")
    tv = THESIS_LSTM.get(r["regime"])
    better = ""
    if tv is not None and r["r2"] > tv:
        better = " <-- BETTER"
    print("{:<15} {:<55} {:>8.4f} {:>5} {:<15}{}".format(
        r["regime"], r["exp"], r["r2"], str(r["N"]), str(r["mass_pp"]), better))
