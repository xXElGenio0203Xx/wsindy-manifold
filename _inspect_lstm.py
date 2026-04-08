import torch, json, pathlib, csv

exp = pathlib.Path("oscar_output/NDYN05_blackhole_thesis_final")
ckpt_path = exp / "LSTM/lstm_state_dict.pt"
log_path = exp / "LSTM/training_log.csv"
test_path = exp / "LSTM/test_results.csv"

print("=== CHECKPOINT METADATA ===")
ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
print("type:", type(ckpt))
if isinstance(ckpt, dict):
    for k, v in ckpt.items():
        if k == "state_dict":
            total = sum(vv.numel() for vv in v.values())
            print(f"total_params: {total:,}")
            print("layers:", list(v.keys()))
        elif k in ("input_mean", "input_std"):
            import numpy as np
            arr = v if hasattr(v, "__len__") else [v]
            print(f"{k}: {arr}")
        else:
            print(f"{k}: {v}")

print("\n=== TRAINING LOG (last 5 epochs) ===")
rows = list(csv.reader(open(log_path)))
for r in rows[:1] + rows[-5:]:
    print(" ", r)

print("\n=== TEST RESULTS ===")
rows = list(csv.reader(open(test_path)))
for r in rows:
    print(" ", r)

sj = exp / "summary.json"
if sj.exists():
    d = json.load(open(sj))
    print("\n=== SUMMARY.JSON (LSTM section) ===")
    lstm = d.get("lstm", {})
    for k, v in lstm.items():
        print(f"  {k}: {v}")
    print("  n_train:", d.get("n_train"))
    print("  r_pod:", d.get("r_pod"))
    print("  lag:", d.get("lag"))
    print("  mass_postprocess:", d.get("mass_postprocess"))
