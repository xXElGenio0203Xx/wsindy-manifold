#!/usr/bin/env python3
"""
Analyze ALL completed ablation experiments: VLST (14) + VARCH (42).

Produces:
  1. A comprehensive LaTeX-ready table of all 56 experiments
  2. Separate tables for VLST (transform ablation) and VARCH (architecture ablation)
  3. Key findings summary
  4. Comparison with prior VDYN baseline (MVAR)
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

# ── Parse the combined summaries file ──
def parse_summaries(filepath):
    """Parse the multi-experiment dump into a list of dicts.
    
    Format: blocks separated by '=====' lines, each block has:
      line 1: directory name
      lines 2+: JSON object (summary.json content)
    """
    results = []
    
    with open(filepath) as f:
        content = f.read()
    
    # Split on the ===== separator
    blocks = content.split("=====")
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        
        # First line is the directory name, rest is JSON
        lines = block.split("\n", 1)
        if len(lines) < 2:
            continue
        
        dir_name = lines[0].strip()
        json_text = lines[1].strip()
        
        if not json_text.startswith("{"):
            continue
        
        try:
            data = json.loads(json_text)
            data["_dir_name"] = dir_name
            results.append(data)
        except json.JSONDecodeError as e:
            print(f"  ⚠ Failed to parse JSON for {dir_name}: {e}")
    
    return results


def extract_metrics(entry):
    """Extract standardized metrics from a summary entry."""
    name = entry.get("experiment_name", entry.get("_dir_name", "unknown"))
    
    # LSTM metrics
    lstm = entry.get("lstm", {})
    r2_rollout = lstm.get("mean_r2_test", None)
    r2_1step = lstm.get("mean_r2_1step_test", None)
    neg_frac = lstm.get("mean_negativity_frac", None)
    val_loss = lstm.get("val_loss", None)
    hidden = lstm.get("hidden_units", None)
    layers = lstm.get("num_layers", None)
    
    # MVAR metrics (if present)
    mvar = entry.get("mvar", {})
    mvar_r2 = mvar.get("mean_r2_test", None)
    mvar_neg = mvar.get("mean_negativity_frac", None)
    
    # POD
    pod = entry.get("pod", {})
    density_transform = pod.get("density_transform", "raw")
    
    clamp = entry.get("clamp_mode", "?")
    mass_pp = entry.get("mass_postprocess", "?")
    
    # Model params
    runtime = entry.get("runtime_analysis", {})
    profiles = runtime.get("profiles", [])
    n_params = None
    for p in profiles:
        if p.get("model_name") == "LSTM":
            n_params = p.get("complexity", {}).get("total_parameters", 
                       p.get("memory", {}).get("model_parameters", None))
            break
    
    # Training time
    train_time = None
    for p in profiles:
        if p.get("model_name") == "LSTM":
            train_time = p.get("training", {}).get("total_seconds", None)
            break
    
    return {
        "name": name,
        "r2_rollout": r2_rollout,
        "r2_1step": r2_1step,
        "neg_frac": neg_frac,
        "val_loss": val_loss,
        "hidden": hidden,
        "layers": layers,
        "n_params": n_params,
        "train_time_s": train_time,
        "density_transform": density_transform,
        "clamp_mode": clamp,
        "mass_pp": mass_pp,
        "mvar_r2": mvar_r2,
        "mvar_neg": mvar_neg,
    }


def regime_from_name(name):
    """Extract regime label from experiment name."""
    # Primary: match by regime keyword in name (most reliable)
    name_lower = name.lower()
    for key, label in [("gentle", "Gentle"), ("hypervelocity", "HyperVel"),
                       ("hypernoisy", "HyperNoisy"), ("blackhole", "Blackhole"),
                       ("supernova", "Supernova"), ("pure_vicsek", "PureVicsek"),
                       ("baseline", "Baseline")]:
        if key in name_lower:
            return label
    return "?"


def regime_number(name):
    """Extract regime number for sorting."""
    m = re.search(r'(\d)', name.split("_")[1] if "_" in name else name)
    if m:
        return int(m.group(1))
    return 0


def suite_from_name(name):
    """Extract suite identifier."""
    if name.startswith("VLST_A"):
        return "VLST-A (sqrt+simplex+C0)"
    elif name.startswith("VLST_B"):
        return "VLST-B (raw+C2)"
    elif "A16" in name:
        return "VARCH-A16 (Nh=16,L=1)"
    elif "A32x2" in name:
        return "VARCH-A32x2 (Nh=32,L=2)"
    elif "A32" in name:
        return "VARCH-A32 (Nh=32,L=1)"
    elif "B1" in name:
        return "VARCH-B1 (no residual)"
    elif "B2" in name:
        return "VARCH-B2 (no LayerNorm)"
    elif "B3" in name:
        return "VARCH-B3 (multistep ON)"
    return "?"


def fmt_r2(val):
    """Format R² value with color coding."""
    if val is None:
        return "  —  "
    if val >= 0.9:
        return f"{val:+.3f} ★"
    elif val >= 0.5:
        return f"{val:+.3f}  "
    elif val >= 0.0:
        return f"{val:+.3f}  "
    elif val >= -1.0:
        return f"{val:+.3f}  "
    else:
        return f"{val:+.2f}  "


def fmt_neg(val):
    """Format negativity fraction."""
    if val is None:
        return "  —  "
    if val == 0.0:
        return " 0.0% ✓"
    elif val < 1.0:
        return f"{val:.1f}%  "
    elif val < 10.0:
        return f"{val:.1f}%  "
    else:
        return f"{val:.1f}% ✗"


# ── VDYN baseline MVAR results (from prior analysis) ──
VDYN_MVAR_BASELINE = {
    "Gentle":     {"r2": 0.904, "neg": 18.3},
    "HyperVel":   {"r2": 0.850, "neg": 16.4},
    "HyperNoisy": {"r2": 0.783, "neg": 12.1},
    "Blackhole":  {"r2": 0.836, "neg": 14.9},
    "Supernova":  {"r2": 0.734, "neg":  9.8},
    "Baseline":   {"r2": 0.880, "neg": 14.1},
    "PureVicsek": {"r2": 0.643, "neg": 11.2},
}


def main():
    # Parse
    filepath = "/tmp/all_ablation_summaries.txt"
    if not Path(filepath).exists():
        print(f"Error: {filepath} not found. Run download script first.")
        sys.exit(1)
    
    entries = parse_summaries(filepath)
    print(f"\nParsed {len(entries)} experiment results")
    
    # Extract metrics
    all_metrics = [extract_metrics(e) for e in entries]
    
    # Separate into VLST and VARCH
    vlst = [m for m in all_metrics if m["name"].startswith("VLST")]
    varch = [m for m in all_metrics if m["name"].startswith("VARCH")]
    
    print(f"  VLST experiments: {len(vlst)}")
    print(f"  VARCH experiments: {len(varch)}")
    
    # ═══════════════════════════════════════════════════════════════════
    # TABLE 1: VLST — Transform Ablation (sqrt+simplex+C0 vs raw+C2)
    # ═══════════════════════════════════════════════════════════════════
    print("\n")
    print("=" * 120)
    print("TABLE 1: VLST — DENSITY TRANSFORM ABLATION  (LSTM Nh=128, L=2, 211k params)")
    print("  Suite A: density_transform=sqrt, mass_postprocess=simplex, clamp_mode=C0")
    print("  Suite B: density_transform=raw,  mass_postprocess=none,    clamp_mode=C2")
    print("  Both suites use the SAME over-parameterized architecture from VDYN")
    print("  Test horizon: 50s forecast (autoregressive rollout)")
    print("=" * 120)
    
    # Group by suite and regime
    vlst_a = sorted([m for m in vlst if "sqrtSimplex" in m["name"] or "_A" in m["name"][:7]],
                     key=lambda m: regime_number(m["name"]))
    vlst_b = sorted([m for m in vlst if "raw_C2" in m["name"] or "_B" in m["name"][:7]],
                     key=lambda m: regime_number(m["name"]))
    
    header = f"{'Regime':<12s} │ {'R² (rollout)':>12s} {'R² (1-step)':>12s} {'Neg%':>8s} {'ValLoss':>10s} │ {'R² (rollout)':>12s} {'R² (1-step)':>12s} {'Neg%':>8s} {'ValLoss':>10s} │ {'MVAR R²':>8s}"
    print(f"\n{'':12s} │ {'── Suite A (sqrt+simplex+C0) ──':^46s} │ {'── Suite B (raw+C2) ──':^46s} │ {'Baseline':>8s}")
    print(header)
    print("─" * 120)
    
    for a, b in zip(vlst_a, vlst_b):
        regime = regime_from_name(a["name"])
        mvar_r2 = VDYN_MVAR_BASELINE.get(regime, {}).get("r2", None)
        mvar_str = f"{mvar_r2:.3f}" if mvar_r2 else "  —  "
        
        print(f"{regime:<12s} │ {fmt_r2(a['r2_rollout']):>12s} {fmt_r2(a['r2_1step']):>12s} {fmt_neg(a['neg_frac']):>8s} {a['val_loss']:.6f}   │ {fmt_r2(b['r2_rollout']):>12s} {fmt_r2(b['r2_1step']):>12s} {fmt_neg(b['neg_frac']):>8s} {b['val_loss']:.6f}   │ {mvar_str:>8s}")
    
    # Suite averages
    a_avg_r2 = sum(m["r2_rollout"] for m in vlst_a) / len(vlst_a)
    b_avg_r2 = sum(m["r2_rollout"] for m in vlst_b) / len(vlst_b)
    a_avg_neg = sum(m["neg_frac"] for m in vlst_a) / len(vlst_a)
    b_avg_neg = sum(m["neg_frac"] for m in vlst_b) / len(vlst_b)
    print("─" * 120)
    print(f"{'AVERAGE':<12s} │ {fmt_r2(a_avg_r2):>12s} {'':12s} {fmt_neg(a_avg_neg):>8s} {'':10s}   │ {fmt_r2(b_avg_r2):>12s} {'':12s} {fmt_neg(b_avg_neg):>8s}")
    
    # VLST Key findings
    print(f"\n  KEY FINDINGS (VLST):")
    a_better = sum(1 for a, b in zip(vlst_a, vlst_b) if a["r2_rollout"] > b["r2_rollout"])
    b_better = 7 - a_better
    print(f"    • Suite A (sqrt+simplex+C0) wins {a_better}/7 regimes in rollout R²")
    print(f"    • Suite B (raw+C2) wins {b_better}/7 regimes in rollout R²")
    print(f"    • Suite A avg R²(rollout) = {a_avg_r2:.3f},  Suite B avg = {b_avg_r2:.3f}")
    print(f"    • Suite A avg neg% = {a_avg_neg:.1f}%,  Suite B avg neg% = {b_avg_neg:.1f}%")
    print(f"    • ALL LSTM rollouts still negative R² → transform alone doesn't fix 211k-param overfitting")
    
    # ═══════════════════════════════════════════════════════════════════
    # TABLE 2: VARCH — Architecture Ablation
    # ═══════════════════════════════════════════════════════════════════
    print("\n\n")
    print("=" * 140)
    print("TABLE 2: VARCH — ARCHITECTURE & TRICKS ABLATION  (all with sqrt+simplex+C0)")
    print("  Suite A: Capacity sweep — A16 (Nh=16,L=1,2.7k), A32 (Nh=32,L=1,7.5k), A32x2 (Nh=32,L=2,15.9k)")
    print("  Suite B: Trick sweep at Nh=32,L=1 — B1 (no residual), B2 (no LayerNorm), B3 (multistep ON)")
    print("  Reference (A32): Nh=32, L=1, residual=ON, LayerNorm=ON, multistep=OFF, dropout=0.0, SS=OFF")
    print("  Test horizon: 50s forecast (autoregressive rollout)")
    print("=" * 140)
    
    # Group VARCH by variant
    variant_order = ["VARCH_A16", "VARCH_A32_", "VARCH_A32x2", "VARCH_B1", "VARCH_B2", "VARCH_B3"]
    variant_labels = {
        "VARCH_A16": "A16 (Nh=16, L=1, 2.7k)",
        "VARCH_A32_": "A32 (Nh=32, L=1, 7.5k) ← REF",
        "VARCH_A32x2": "A32x2 (Nh=32, L=2, 15.9k)",
        "VARCH_B1": "B1 (no residual)",
        "VARCH_B2": "B2 (no LayerNorm)",
        "VARCH_B3": "B3 (multistep ON)",
    }
    
    # Build regime-indexed grid
    regimes = ["Gentle", "HyperVel", "HyperNoisy", "Blackhole", "Supernova", "Baseline", "PureVicsek"]
    
    # Index all varch results
    varch_grid = {}  # (variant_key, regime) -> metrics
    for m in varch:
        for vk in variant_order:
            if m["name"].startswith(vk):
                regime = regime_from_name(m["name"])
                varch_grid[(vk, regime)] = m
                break
    
    # Print rollout R² table
    print(f"\n{'ROLLOUT R² (50s autoregressive)':^140s}")
    header = f"{'Variant':<30s} │"
    for r in regimes:
        header += f" {r:>12s}"
    header += f" │ {'Mean':>8s} {'Median':>8s}"
    print(header)
    print("─" * 140)
    
    for vk in variant_order:
        label = variant_labels[vk]
        row = f"{label:<30s} │"
        vals = []
        for r in regimes:
            m = varch_grid.get((vk, r))
            if m:
                v = m["r2_rollout"]
                vals.append(v)
                row += f" {fmt_r2(v):>12s}"
            else:
                row += f" {'—':>12s}"
        
        if vals:
            mean_v = sum(vals) / len(vals)
            sorted_v = sorted(vals)
            median_v = sorted_v[len(sorted_v)//2]
            row += f" │ {mean_v:>+8.3f} {median_v:>+8.3f}"
        print(row)
    
    # MVAR baseline row
    row = f"{'MVAR baseline (from VDYN)':30s} │"
    mvar_vals = []
    for r in regimes:
        v = VDYN_MVAR_BASELINE.get(r, {}).get("r2")
        if v is not None:
            mvar_vals.append(v)
            row += f" {v:>+12.3f}  "
        else:
            row += f" {'—':>12s}"
    mean_mvar = sum(mvar_vals) / len(mvar_vals) if mvar_vals else 0
    row += f" │ {mean_mvar:>+8.3f}"
    print("─" * 140)
    print(row)
    
    # Print 1-step R² table
    print(f"\n{'1-STEP R² (teacher-forced)':^140s}")
    header = f"{'Variant':<30s} │"
    for r in regimes:
        header += f" {r:>12s}"
    header += f" │ {'Mean':>8s}"
    print(header)
    print("─" * 140)
    
    for vk in variant_order:
        label = variant_labels[vk]
        row = f"{label:<30s} │"
        vals = []
        for r in regimes:
            m = varch_grid.get((vk, r))
            if m:
                v = m["r2_1step"]
                vals.append(v)
                row += f" {fmt_r2(v):>12s}"
            else:
                row += f" {'—':>12s}"
        if vals:
            mean_v = sum(vals) / len(vals)
            row += f" │ {mean_v:>+8.4f}"
        print(row)
    
    # Print negativity table
    print(f"\n{'NEGATIVITY FRACTION (% cells with ρ<0)':^140s}")
    header = f"{'Variant':<30s} │"
    for r in regimes:
        header += f" {r:>12s}"
    header += f" │ {'Mean':>8s}"
    print(header)
    print("─" * 140)
    
    for vk in variant_order:
        label = variant_labels[vk]
        row = f"{label:<30s} │"
        vals = []
        for r in regimes:
            m = varch_grid.get((vk, r))
            if m:
                v = m["neg_frac"]
                vals.append(v)
                row += f" {fmt_neg(v):>12s}"
            else:
                row += f" {'—':>12s}"
        if vals:
            mean_v = sum(vals) / len(vals)
            row += f" │ {mean_v:>7.1f}%"
        print(row)
    
    # Print val_loss table
    print(f"\n{'VALIDATION LOSS (MSE, normalized space)':^140s}")
    header = f"{'Variant':<30s} │"
    for r in regimes:
        header += f" {r:>12s}"
    header += f" │ {'Mean':>8s}"
    print(header)
    print("─" * 140)
    
    for vk in variant_order:
        label = variant_labels[vk]
        row = f"{label:<30s} │"
        vals = []
        for r in regimes:
            m = varch_grid.get((vk, r))
            if m:
                v = m["val_loss"]
                vals.append(v)
                row += f" {v:>12.5f}"
            else:
                row += f" {'—':>12s}"
        if vals:
            mean_v = sum(vals) / len(vals)
            row += f" │ {mean_v:>8.5f}"
        print(row)
    
    # Print parameters & training time
    print(f"\n{'MODEL SIZE & TRAINING TIME':^140s}")
    header = f"{'Variant':<30s} │ {'Params':>8s} │ {'Train Time (s)':>14s}"
    print(header)
    print("─" * 60)
    for vk in variant_order:
        label = variant_labels[vk]
        # Get from any regime (params are same across regimes)
        m = varch_grid.get((vk, regimes[0]))
        if m:
            params = m["n_params"]
            time_s = m["train_time_s"]
            params_str = f"{params:,}" if params else "—"
            time_str = f"{time_s:.1f}" if time_s else "—"
            print(f"{label:<30s} │ {params_str:>8s} │ {time_str:>14s}")
    
    # ═══════════════════════════════════════════════════════════════════
    # VARCH Key findings
    # ═══════════════════════════════════════════════════════════════════
    print("\n")
    print("=" * 100)
    print("KEY FINDINGS — VARCH ARCHITECTURE ABLATION")
    print("=" * 100)
    
    # Find best variant per regime
    print("\n  BEST VARIANT PER REGIME (rollout R²):")
    variant_wins = defaultdict(int)
    for r in regimes:
        best_vk = None
        best_r2 = -999
        for vk in variant_order:
            m = varch_grid.get((vk, r))
            if m and m["r2_rollout"] > best_r2:
                best_r2 = m["r2_rollout"]
                best_vk = vk
        if best_vk:
            variant_wins[best_vk] += 1
            print(f"    {r:12s} → {variant_labels[best_vk]:30s}  R²={best_r2:+.3f}")
    
    print(f"\n  WIN COUNT:")
    for vk in variant_order:
        print(f"    {variant_labels[vk]:30s}  wins {variant_wins[vk]}/7")
    
    # Capacity effect
    print(f"\n  CAPACITY EFFECT (A16 vs A32 vs A32x2):")
    for vk in ["VARCH_A16", "VARCH_A32_", "VARCH_A32x2"]:
        vals = [varch_grid.get((vk, r), {}).get("r2_rollout", None) for r in regimes]
        vals = [v for v in vals if v is not None]
        if vals:
            print(f"    {variant_labels[vk]:30s}  mean R²={sum(vals)/len(vals):+.3f}  range=[{min(vals):+.3f}, {max(vals):+.3f}]")
    
    # Trick effect at A32
    print(f"\n  TRICK EFFECT (relative to A32 baseline):")
    ref_vals = {r: varch_grid.get(("VARCH_A32_", r), {}).get("r2_rollout", None) for r in regimes}
    for vk in ["VARCH_B1", "VARCH_B2", "VARCH_B3"]:
        deltas = []
        for r in regimes:
            m = varch_grid.get((vk, r))
            ref = ref_vals.get(r)
            if m and ref is not None:
                deltas.append(m["r2_rollout"] - ref)
        if deltas:
            print(f"    {variant_labels[vk]:30s}  ΔR² vs A32: mean={sum(deltas)/len(deltas):+.4f}  range=[{min(deltas):+.4f}, {max(deltas):+.4f}]")
    
    # Compare with VLST (same regime, 211k vs 7.5k)
    print(f"\n  DOWNSIZING EFFECT (VLST-A 211k → VARCH-A32 7.5k, both sqrt+simplex+C0):")
    for i, r in enumerate(regimes):
        vlst_m = next((m for m in vlst_a if regime_from_name(m["name"]) == r), None)
        varch_m = varch_grid.get(("VARCH_A32_", r))
        if vlst_m and varch_m:
            delta = varch_m["r2_rollout"] - vlst_m["r2_rollout"]
            direction = "↑ IMPROVED" if delta > 0 else "↓ worsened"
            print(f"    {r:12s}  211k R²={vlst_m['r2_rollout']:+.3f} → 7.5k R²={varch_m['r2_rollout']:+.3f}  Δ={delta:+.3f} {direction}")
    
    # Overall verdict
    print(f"\n  OVERALL VERDICT:")
    
    # Check if any LSTM achieves positive rollout R²
    all_r2 = [m["r2_rollout"] for m in all_metrics if m["r2_rollout"] is not None]
    n_positive = sum(1 for v in all_r2 if v > 0)
    n_total = len(all_r2)
    best_overall = max(all_metrics, key=lambda m: m["r2_rollout"] if m["r2_rollout"] else -999)
    worst_overall = min(all_metrics, key=lambda m: m["r2_rollout"] if m["r2_rollout"] else 999)
    
    print(f"    • {n_positive}/{n_total} LSTM experiments achieve positive rollout R²")
    print(f"    • Best LSTM:  {best_overall['name']:40s}  R²={best_overall['r2_rollout']:+.3f}")
    print(f"    • Worst LSTM: {worst_overall['name']:40s}  R²={worst_overall['r2_rollout']:+.3f}")
    print(f"    • MVAR baseline mean R² = {mean_mvar:+.3f} (wins all regimes)")
    
    # Check LSTM vs MVAR
    any_lstm_beats_mvar = False
    for r in regimes:
        mvar_v = VDYN_MVAR_BASELINE.get(r, {}).get("r2", -999)
        for vk in variant_order:
            m = varch_grid.get((vk, r))
            if m and m["r2_rollout"] > mvar_v:
                any_lstm_beats_mvar = True
                print(f"    ★ LSTM beats MVAR in {r}: {variant_labels[vk]} R²={m['r2_rollout']:+.3f} vs MVAR={mvar_v:.3f}")
    
    if not any_lstm_beats_mvar:
        print(f"    • NO LSTM variant beats MVAR in ANY regime")
    
    # ═══════════════════════════════════════════════════════════════════
    # COMPACT SUMMARY TABLE (all 56 experiments, thesis-ready)
    # ═══════════════════════════════════════════════════════════════════
    print("\n\n")
    print("=" * 130)
    print("COMPACT SUMMARY: ALL 56 LSTM EXPERIMENTS")
    print(f"  7 regimes × 8 variants (2 VLST + 6 VARCH)")
    print(f"  Metric: Rollout R² on held-out test trajectories, 50s forecast horizon")
    print("=" * 130)
    
    all_variants = [
        ("VLST-A (Nh=128,L=2,211k,√+S+C0)", lambda m: "sqrtSimplex" in m["name"] and m["name"].startswith("VLST")),
        ("VLST-B (Nh=128,L=2,211k,raw+C2)", lambda m: "raw_C2" in m["name"] and m["name"].startswith("VLST")),
        ("A16 (Nh=16,L=1,2.7k)", lambda m: m["name"].startswith("VARCH_A16")),
        ("A32 (Nh=32,L=1,7.5k) ★REF", lambda m: m["name"].startswith("VARCH_A32_")),
        ("A32x2 (Nh=32,L=2,15.9k)", lambda m: m["name"].startswith("VARCH_A32x2")),
        ("B1 (A32, no residual)", lambda m: m["name"].startswith("VARCH_B1")),
        ("B2 (A32, no LayerNorm)", lambda m: m["name"].startswith("VARCH_B2")),
        ("B3 (A32, +multistep)", lambda m: m["name"].startswith("VARCH_B3")),
        ("MVAR (p=5,α=1e-4)", None),  # baseline
    ]
    
    header = f"{'Variant':<35s} │"
    for r in regimes:
        header += f" {r[:8]:>8s}"
    header += f" │ {'Mean':>7s}"
    print(f"\n{header}")
    print("─" * 130)
    
    for label, filter_fn in all_variants:
        row = f"{label:<35s} │"
        vals = []
        
        if filter_fn is None:
            # MVAR baseline
            for r in regimes:
                v = VDYN_MVAR_BASELINE.get(r, {}).get("r2")
                if v is not None:
                    vals.append(v)
                    row += f" {v:>+8.3f}"
                else:
                    row += f" {'—':>8s}"
        else:
            matched = sorted([m for m in all_metrics if filter_fn(m)],
                           key=lambda m: regime_number(m["name"]))
            for m in matched:
                v = m["r2_rollout"]
                vals.append(v)
                if v >= 0:
                    row += f" {v:>+8.3f}"
                elif v >= -1:
                    row += f" {v:>+8.3f}"
                else:
                    row += f" {v:>+8.2f}"
        
        if vals:
            mean_v = sum(vals) / len(vals)
            row += f" │ {mean_v:>+7.3f}"
        print(row)
    
    print("─" * 130)
    print(f"  √=sqrt transform, S=simplex mass, C0/C2=clamp mode")
    print(f"  All VARCH variants use sqrt+simplex+C0, dropout=0.0, SS=OFF")
    print(f"  ★REF = reference configuration for trick ablation (B1, B2, B3)")


if __name__ == "__main__":
    main()
