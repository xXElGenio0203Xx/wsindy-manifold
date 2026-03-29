#!/usr/bin/env python3
"""Scan local oscar_output for R² scores."""
import json, os

base = "oscar_output"
results = []

for d in sorted(os.listdir(base)):
    full = os.path.join(base, d)
    if not os.path.isdir(full):
        continue
    summary = os.path.join(full, "summary.json")
    if not os.path.isfile(summary):
        continue
    with open(summary) as f:
        s = json.load(f)

    row = {"exp": d}

    # MVAR results
    mvar = s.get("mvar", s.get("MVAR", {}))
    if isinstance(mvar, dict):
        row["mvar_r2"] = mvar.get("mean_r2_test", mvar.get("r2_mean", mvar.get("test", {}).get("r2_mean", "N/A")))
    else:
        row["mvar_r2"] = "N/A"

    # LSTM results
    lstm = s.get("lstm", s.get("LSTM", {}))
    if isinstance(lstm, dict):
        row["lstm_r2"] = lstm.get("mean_r2_test", lstm.get("r2_mean", lstm.get("test", {}).get("r2_mean", "N/A")))
        row["lstm_1step"] = lstm.get("mean_r2_1step_test", "N/A")
    else:
        row["lstm_r2"] = "N/A"
        row["lstm_1step"] = "N/A"

    row["total_min"] = s.get("total_time_minutes", "N/A")
    
    # Try to get speed mode from config
    cfg = s.get("config", {})
    if isinstance(cfg, dict):
        model = cfg.get("model", {})
        row["speed_mode"] = model.get("speed_mode", "?") if isinstance(model, dict) else "?"
        forces = cfg.get("forces", {})
        row["forces"] = "morse" if (isinstance(forces, dict) and forces.get("enabled", False)) else "off"
    else:
        row["speed_mode"] = "?"
        row["forces"] = "?"

    results.append(row)


def fmt(v):
    if isinstance(v, float):
        return f"{v:>8.4f}"
    return f"{str(v):>8}"


print(f"{'Experiment':<50} {'MVAR_R2':>8} {'LSTM_R2':>8} {'LSTM_1s':>8} {'Speed':>8} {'Forces':>6}")
print("─" * 94)
for r in results:
    print(f"{r['exp']:<50} {fmt(r['mvar_r2'])} {fmt(r['lstm_r2'])} {fmt(r.get('lstm_1step', 'N/A'))} {str(r.get('speed_mode', '?')):>8} {r.get('forces', '?'):>6}")

# Best MVAR
print()
print("=== TOP 15 EXPERIMENTS BY MVAR R² ===")
mvar_good = [r for r in results if isinstance(r["mvar_r2"], (int, float))]
mvar_good.sort(key=lambda x: x["mvar_r2"], reverse=True)
for r in mvar_good[:15]:
    print(f"  {r['exp']:<50} MVAR={r['mvar_r2']:.4f}  speed={r['speed_mode']}  forces={r['forces']}")

# Best LSTM
print()
print("=== TOP 15 EXPERIMENTS BY LSTM R² ===")
lstm_good = [r for r in results if isinstance(r["lstm_r2"], (int, float))]
lstm_good.sort(key=lambda x: x["lstm_r2"], reverse=True)
for r in lstm_good[:15]:
    print(f"  {r['exp']:<50} LSTM={r['lstm_r2']:.4f}  speed={r['speed_mode']}  forces={r['forces']}")

# Positive R²
print()
print("=== EXPERIMENTS WITH R² > 0.5 (either model) ===")
good = [r for r in results
        if (isinstance(r["mvar_r2"], (int, float)) and r["mvar_r2"] > 0.5)
        or (isinstance(r["lstm_r2"], (int, float)) and r["lstm_r2"] > 0.5)]
good.sort(key=lambda x: max(
    x["mvar_r2"] if isinstance(x["mvar_r2"], (int, float)) else -999,
    x["lstm_r2"] if isinstance(x["lstm_r2"], (int, float)) else -999
), reverse=True)
for r in good:
    m = r["mvar_r2"] if isinstance(r["mvar_r2"], (int, float)) else None
    l = r["lstm_r2"] if isinstance(r["lstm_r2"], (int, float)) else None
    m_str = f"MVAR={m:.4f}" if m is not None else "MVAR=N/A"
    l_str = f"LSTM={l:.4f}" if l is not None else "LSTM=N/A"
    print(f"  {r['exp']:<50} {m_str:<16} {l_str:<16} speed={r['speed_mode']}  forces={r['forces']}")
