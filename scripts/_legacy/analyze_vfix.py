#!/usr/bin/env python3
"""Extract and summarize VFIX experiment results from Oscar output."""
import json, os, re, statistics
from collections import defaultdict

BASE = os.path.expanduser("~/wsindy-manifold/oscar_output")

results = []
for d in sorted(os.listdir(BASE)):
    if not d.startswith("VFIX_"):
        continue
    summary = os.path.join(BASE, d, "summary.json")
    if not os.path.isfile(summary):
        continue
    with open(summary) as f:
        s = json.load(f)
    lstm = s.get("lstm", {})
    
    m = re.match(r"VFIX_(F\w+?)_(\d+)_(.+)", d)
    if m:
        variant, regime_num, regime = m.group(1), int(m.group(2)), m.group(3)
    else:
        variant, regime_num, regime = d, 0, "?"
    
    results.append({
        "exp": d,
        "variant": variant,
        "regime": regime,
        "regime_num": regime_num,
        "r2_rollout": lstm.get("mean_r2_test", "N/A"),
        "r2_1step": lstm.get("mean_r2_1step_test", "N/A"),
        "neg_frac": lstm.get("mean_negativity_frac", "N/A"),
        "val_loss": lstm.get("val_loss", "N/A"),
        "hidden": lstm.get("hidden_units", "?"),
        "layers": lstm.get("num_layers", "?"),
        "total_min": s.get("total_time_minutes", "N/A"),
    })

def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)

# ── Full CSV ──
print("experiment,variant,regime,hidden,layers,r2_rollout,r2_1step,neg_frac,val_loss,time_min")
for r in results:
    cols = [r["exp"], r["variant"], r["regime"], str(r["hidden"]), str(r["layers"]),
            fmt(r["r2_rollout"]), fmt(r["r2_1step"]), fmt(r["neg_frac"]),
            fmt(r["val_loss"]), fmt(r["total_min"])]
    print(",".join(cols))

# ── Mean R² by variant ──
print("\n=== MEAN R² BY VARIANT (across regimes) ===")
by_variant = defaultdict(list)
for r in results:
    if isinstance(r["r2_rollout"], (int, float)):
        by_variant[r["variant"]].append(r["r2_rollout"])

print(f"{'Variant':<10} {'H':>4} {'L':>2} {'Mean_R2':>8} {'Min_R2':>8} {'Max_R2':>8} {'N':>3}")
for var in ["F16", "F32", "F64", "F128", "F32x2", "F64x2", "FSS", "FSQRT"]:
    if var in by_variant:
        vals = by_variant[var]
        h = [r for r in results if r["variant"] == var][0]["hidden"]
        l = [r for r in results if r["variant"] == var][0]["layers"]
        print(f"{var:<10} {h:>4} {l:>2} {statistics.mean(vals):>8.4f} {min(vals):>8.4f} {max(vals):>8.4f} {len(vals):>3}")

# ── Mean R² by regime ──
print("\n=== MEAN R² BY REGIME (across variants) ===")
by_regime = defaultdict(list)
for r in results:
    if isinstance(r["r2_rollout"], (int, float)):
        by_regime[r["regime"]].append(r["r2_rollout"])

for regime in ["gentle", "hypervelocity", "hypernoisy", "blackhole", "supernova", "baseline", "pure_vicsek"]:
    if regime in by_regime:
        vals = by_regime[regime]
        print(f"  {regime:<16} mean={statistics.mean(vals):>8.4f}  min={min(vals):>8.4f}  max={max(vals):>8.4f}  n={len(vals)}")

# ── Best variant per regime ──
print("\n=== BEST VARIANT PER REGIME ===")
for regime in ["gentle", "hypervelocity", "hypernoisy", "blackhole", "supernova", "baseline", "pure_vicsek"]:
    best = None
    for r in results:
        if r["regime"] == regime and isinstance(r["r2_rollout"], (int, float)):
            if best is None or r["r2_rollout"] > best["r2_rollout"]:
                best = r
    if best:
        print(f"  {regime:<16} BEST={best['variant']:<8} R²={best['r2_rollout']:.4f}")

# ── Full grid: variant × regime ──
print("\n=== FULL GRID (R² rollout) ===")
regimes = ["gentle", "hypervelocity", "hypernoisy", "blackhole", "supernova", "baseline", "pure_vicsek"]
variants = ["F16", "F32", "F64", "F128", "F32x2", "F64x2", "FSS", "FSQRT"]
grid = {}
for r in results:
    grid[(r["variant"], r["regime"])] = r["r2_rollout"]

header = f"{'Variant':<10}" + "".join(f"{reg:>16}" for reg in regimes) + f"{'MEAN':>10}"
print(header)
for var in variants:
    row = f"{var:<10}"
    vals = []
    for reg in regimes:
        v = grid.get((var, reg), "—")
        if isinstance(v, (int, float)):
            row += f"{v:>16.4f}"
            vals.append(v)
        else:
            row += f"{'—':>16}"
    if vals:
        row += f"{statistics.mean(vals):>10.4f}"
    print(row)

# ── Compare with VDYN MVAR baselines ──
print("\n=== COMPARISON WITH VDYN MVAR BASELINES ===")
mvar_baselines = {
    "gentle": 0.904, "hypervelocity": 0.850, "hypernoisy": 0.783,
    "blackhole": 0.836, "supernova": 0.734, "baseline": 0.880,
    "pure_vicsek": 0.643
}
print(f"{'Regime':<16} {'MVAR_R2':>8} {'Best_LSTM':>10} {'Variant':>8} {'Delta':>8}")
for regime in regimes:
    best = None
    for r in results:
        if r["regime"] == regime and isinstance(r["r2_rollout"], (int, float)):
            if best is None or r["r2_rollout"] > best["r2_rollout"]:
                best = r
    mvar = mvar_baselines.get(regime, 0)
    if best:
        delta = best["r2_rollout"] - mvar
        print(f"  {regime:<14} {mvar:>8.3f} {best['r2_rollout']:>10.4f} {best['variant']:>8} {delta:>+8.4f}")
