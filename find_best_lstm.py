"""Find the best LSTM R² for each regime across ALL experiments."""
import csv, os, re

import sys

model = sys.argv[1] if len(sys.argv) > 1 else "LSTM"
base = os.environ.get("OSCAR_BASE", "oscar_output/systematics")

regime_map = {
    "gas": r"NDYN04.*gas(?!_VS)",
    "gas_VS": r"NDYN04.*gas_VS",
    "blackhole": r"NDYN05.*blackhole(?!_VS)",
    "blackhole_VS": r"NDYN05.*blackhole_VS",
    "supernova": r"NDYN06.*supernova(?!_VS)",
    "supernova_VS": r"NDYN06.*supernova_VS",
    "pure_vicsek": r"NDYN08.*pure_vicsek",
}

for regime, pat in regime_map.items():
    best_r2 = -9999
    best_exp = "none"
    for d in sorted(os.listdir(base)):
        if not re.search(pat, d):
            continue
        f = os.path.join(base, d, model, "test_results.csv")
        if not os.path.exists(f):
            continue
        try:
            with open(f) as fh:
                rows = list(csv.DictReader(fh))
            r2 = sum(float(r["r2_reconstructed"]) for r in rows) / len(rows)
            tag = "*" if r2 > best_r2 else " "
            if r2 > best_r2:
                best_r2 = r2
                best_exp = d
            print(f"  {tag} {d:55s} R2={r2:.4f}")
        except Exception:
            pass
    print(f">>> {regime:20s} BEST={best_r2:.4f} from {best_exp}")
    print()
