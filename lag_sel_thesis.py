import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
REGIMES = [
    "NDYN04_gas",
    "NDYN04_gas_VS",
    "NDYN05_blackhole",
    "NDYN05_blackhole_VS",
    "NDYN06_supernova",
    "NDYN06_supernova_VS",
    "NDYN08_pure_vicsek",
]
print("%-30s %5s %5s %5s" % ("Regime", "BIC", "HQIC", "AIC"))
print("-" * 50)
for reg in REGIMES:
    p = "oscar_output/%s_thesis_final/rom_common/latent_dataset.npz" % reg
    d = np.load(p, allow_pickle=True)
    # latent_dataset.npz uses keys: X_all (physical), Y_all (latent POD coords), lag
    Y = d["Y_all"]
    if Y.ndim == 3:
        # Use only first trajectory to keep VAR fitting fast
        Y = Y[0]
    try:
        res = VAR(Y).select_order(maxlags=30)
        b = res.selected_orders["bic"]
        h = res.selected_orders["hqic"]
        a = res.selected_orders["aic"]
    except Exception as e:
        b = h = a = "ERR"
        import sys; print("ERR %s: %s" % (reg, e), file=sys.stderr)
    print("%-30s %5s %5s %5s" % (reg, b, h, a))
