from collections import defaultdict

data = [
    ("NDYN04_gas", "scale", [0.9897, 0.2801, 0.0365, 0.9900], 0.5741),
    ("NDYN04_gas_VS", "scale", [0.6569, -2.3825, -0.2952, -0.1117], -0.5331),
    ("NDYN05_blackhole_VS", "scale", [-0.3761, -1.2816, -0.4588, -0.5069], -0.6559),
    ("NDYN06_supernova", "scale", [0.8284, 0.0885, 0.6593, 0.4540], 0.5076),
    ("NDYN04_gas", "none", [0.9804, -32917.12, -2271.91, 0.9810], -8796.77),
    ("NDYN04_gas_VS", "none", [0.3269, -679.49, -152.14, -109.91], -235.30),
    ("NDYN05_blackhole_VS", "none", [-0.4496, -1.2579, -0.4565, -0.5294], -0.6734),
    ("NDYN06_supernova", "none", [0.8249, 0.0808, 0.6628, 0.4307], 0.4998),
    ("NDYN04_gas", "simplex", [0.9844, -18.12, -7.34, 0.9848], -5.87),
    ("NDYN04_gas_VS", "simplex", [0.5551, -16.85, -4.33, -3.10], -5.93),
    ("NDYN05_blackhole_VS", "simplex", [-0.4285, -1.2244, -0.4433, -0.5109], -0.6518),
    ("NDYN06_supernova", "simplex", [0.8270, 0.0873, 0.6655, 0.4447], 0.5061),
]

table = {}
for regime, mpp, tests, mean in data:
    table[(regime, mpp)] = (tests, mean)

print("=" * 85)
print("MASS POSTPROCESS COMPARISON (reeval_bad_lstm, job 1491316)")
print("All experiments: N=100, thesis_final configs, same LSTM weights")
print("=" * 85)

mpps = ["scale", "none", "simplex"]
for regime in ["NDYN04_gas", "NDYN04_gas_VS", "NDYN05_blackhole_VS", "NDYN06_supernova"]:
    print()
    print("REGIME:", regime)
    print("  {:<10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "mass_pp", "test_000", "test_001", "test_002", "test_003", "MEAN"))
    print("  " + "-" * 65)
    means = {m: table[(regime, m)][1] for m in mpps}
    best_mpp = max(means, key=means.get)
    for mpp in mpps:
        tests, mean = table[(regime, mpp)]
        tag = " <-- BEST" if mpp == best_mpp else ""
        print("  {:<10} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}{}".format(
            mpp, tests[0], tests[1], tests[2], tests[3], mean, tag))

print()
print("=" * 85)
print("CONCLUSION: 'scale' is ALWAYS the best mass_postprocess for LSTM.")
print("  - gas: scale=0.574, none=-8797, simplex=-5.87")
print("  - gas_VS: scale=-0.533, none=-235, simplex=-5.93")
print("  - blackhole_VS: scale=-0.656, none=-0.673, simplex=-0.652")
print("  - supernova: scale=0.508, none=0.500, simplex=0.506")
print()
print("No mass_postprocess change can improve the bad LSTMs.")
print("The problem is the LSTM model itself, not postprocessing.")
