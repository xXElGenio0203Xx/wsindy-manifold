#!/usr/bin/env python3
"""
Deep analysis: Which experiments had good LSTM R² and why?
Compares architecture, dynamics, and pipeline settings.
"""
import csv, json, os, statistics, yaml
from collections import defaultdict

base = "oscar_output"

def load_config(exp_dir):
    """Load config from config_used.yaml or summary.json."""
    cfg_path = os.path.join(exp_dir, "config_used.yaml")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    sum_path = os.path.join(exp_dir, "summary.json")
    if os.path.isfile(sum_path):
        with open(sum_path) as f:
            s = json.load(f)
        cfg = s.get("config", {})
        if isinstance(cfg, dict):
            return cfg
    return {}

def get_r2_from_csv(csv_path, col="r2_reconstructed"):
    """Read R² values from test_results.csv."""
    if not os.path.isfile(csv_path):
        return []
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    vals = []
    for row in rows:
        v = row.get(col)
        if v is not None:
            try:
                vals.append(float(v))
            except ValueError:
                pass
    return vals

def get_r2_from_summary(summary_path, model="lstm"):
    """Read mean_r2_test from summary.json."""
    if not os.path.isfile(summary_path):
        return None
    with open(summary_path) as f:
        s = json.load(f)
    m = s.get(model, {})
    if isinstance(m, dict):
        return m.get("mean_r2_test")
    return None

def extract_features(cfg):
    """Extract key features from config dict."""
    model = cfg.get("model", {})
    if not isinstance(model, dict):
        model = {}
    
    rom = cfg.get("rom", {})
    if not isinstance(rom, dict):
        rom = {}
    
    lstm_cfg = rom.get("models", {}).get("lstm", {})
    if not isinstance(lstm_cfg, dict):
        lstm_cfg = {}
    
    forces = cfg.get("forces", {})
    if not isinstance(forces, dict):
        forces = {}
    
    force_params = forces.get("params", {})
    if not isinstance(force_params, dict):
        force_params = {}
    
    noise = cfg.get("noise", {})
    if not isinstance(noise, dict):
        noise = {}
    
    sim = cfg.get("sim", {})
    if not isinstance(sim, dict):
        sim = {}
    
    test_sim = cfg.get("test_sim", {})
    if not isinstance(test_sim, dict):
        test_sim = {}
    
    # Count training ICs
    train_ic = cfg.get("train_ic", {})
    n_train = 0
    if isinstance(train_ic, dict):
        for ic_type in ["uniform", "gaussian", "two_clusters", "ring"]:
            sub = train_ic.get(ic_type, {})
            if isinstance(sub, dict) and sub.get("enabled", False):
                n_train += sub.get("n_runs", 0) * sub.get("n_samples_per_config", 1) * sub.get("n_samples", 1)
    
    return {
        # Dynamics
        "speed_mode": model.get("speed_mode", "?"),
        "speed": model.get("speed", "?"),
        "forces_on": forces.get("enabled", False),
        "Ca": force_params.get("Ca", "?"),
        "Cr": force_params.get("Cr", "?"),
        "eta": noise.get("eta", "?"),
        "N_particles": sim.get("N", "?"),
        "T_train": sim.get("T", "?"),
        "T_test": test_sim.get("T", "?"),
        
        # ROM pipeline
        "density_transform": rom.get("density_transform", "?"),
        "mass_postprocess": rom.get("mass_postprocess", "?"),
        "shift_align": rom.get("shift_align", "?"),
        "fixed_modes": rom.get("fixed_modes", "?"),
        "subsample": rom.get("subsample", "?"),
        
        # LSTM architecture
        "hidden_units": lstm_cfg.get("hidden_units", "?"),
        "num_layers": lstm_cfg.get("num_layers", "?"),
        "residual": lstm_cfg.get("residual", "?"),
        "use_layer_norm": lstm_cfg.get("use_layer_norm", "?"),
        "dropout": lstm_cfg.get("dropout", "?"),
        "max_epochs": lstm_cfg.get("max_epochs", "?"),
        "lag": lstm_cfg.get("lag", "?"),
        "multistep_loss": lstm_cfg.get("multistep_loss", "?"),
        "scheduled_sampling": lstm_cfg.get("scheduled_sampling", "?"),
        "batch_size": lstm_cfg.get("batch_size", "?"),
        "lr": lstm_cfg.get("learning_rate", "?"),
        
        # Training data
        "n_train_ics": n_train if n_train > 0 else "?",
    }


# ── Gather all experiments ──
experiments = []

for d in sorted(os.listdir(base)):
    full = os.path.join(base, d)
    if not os.path.isdir(full):
        continue
    
    # Get LSTM R²
    lstm_r2_csv = get_r2_from_csv(os.path.join(full, "LSTM", "test_results.csv"))
    lstm_r2_summary = get_r2_from_summary(os.path.join(full, "summary.json"), "lstm")
    
    # Get MVAR R² for comparison
    mvar_r2_csv = get_r2_from_csv(os.path.join(full, "MVAR", "test_results.csv"))
    mvar_r2_summary = get_r2_from_summary(os.path.join(full, "summary.json"), "mvar")
    
    # Use CSV if available, else summary
    if lstm_r2_csv:
        lstm_mean = statistics.mean(lstm_r2_csv)
        lstm_min = min(lstm_r2_csv)
        lstm_max = max(lstm_r2_csv)
        lstm_n = len(lstm_r2_csv)
    elif lstm_r2_summary is not None:
        lstm_mean = lstm_r2_summary
        lstm_min = lstm_max = lstm_mean
        lstm_n = 1
    else:
        continue  # Skip experiments without LSTM results
    
    mvar_mean = statistics.mean(mvar_r2_csv) if mvar_r2_csv else (mvar_r2_summary if mvar_r2_summary is not None else None)
    
    cfg = load_config(full)
    features = extract_features(cfg)
    
    experiments.append({
        "name": d,
        "lstm_mean": lstm_mean,
        "lstm_min": lstm_min,
        "lstm_max": lstm_max,
        "lstm_n": lstm_n,
        "mvar_mean": mvar_mean,
        "features": features,
    })

# Sort by LSTM R²
experiments.sort(key=lambda x: x["lstm_mean"], reverse=True)

# ── Print detailed comparison table ──
print("=" * 120)
print("LSTM R² RANKING — ALL EXPERIMENTS WITH LSTM RESULTS")
print("=" * 120)

print(f"\n{'Rank':<5} {'Experiment':<40} {'LSTM_R2':>8} {'MVAR_R2':>8} {'speed_mode':<16} {'H':>4} {'L':>2} {'res':>4} {'transform':<8} {'Ca':>5} {'Cr':>5} {'eta':>5}")
print("-" * 120)

for i, e in enumerate(experiments):
    f = e["features"]
    mvar_str = f"{e['mvar_mean']:.4f}" if e["mvar_mean"] is not None else "N/A"
    res = "Y" if f["residual"] == True else ("N" if f["residual"] == False else "?")
    print(f"{i+1:<5} {e['name']:<40} {e['lstm_mean']:>8.4f} {mvar_str:>8} {str(f['speed_mode']):<16} {str(f['hidden_units']):>4} {str(f['num_layers']):>2} {res:>4} {str(f['density_transform']):<8} {str(f['Ca']):>5} {str(f['Cr']):>5} {str(f['eta']):>5}")

# ── Identify patterns ──
print("\n" + "=" * 120)
print("PATTERN ANALYSIS")
print("=" * 120)

good = [e for e in experiments if e["lstm_mean"] > 0.5]
bad = [e for e in experiments if e["lstm_mean"] < 0.0]
mediocre = [e for e in experiments if 0.0 <= e["lstm_mean"] <= 0.5]

print(f"\nGood (R² > 0.5):     {len(good)} experiments")
print(f"Mediocre (0 to 0.5): {len(mediocre)} experiments")
print(f"Bad (R² < 0):        {len(bad)} experiments")

# Factor analysis
for factor_name, factor_key in [
    ("Speed Mode", "speed_mode"),
    ("Residual Connection", "residual"),
    ("Hidden Units", "hidden_units"),
    ("Num Layers", "num_layers"),
    ("Density Transform", "density_transform"),
    ("Mass Postprocess", "mass_postprocess"),
    ("Shift Align", "shift_align"),
    ("Dropout", "dropout"),
    ("Max Epochs", "max_epochs"),
    ("Forces Ca", "Ca"),
    ("Forces Cr", "Cr"),
    ("Noise eta", "eta"),
    ("Multistep Loss", "multistep_loss"),
    ("Scheduled Sampling", "scheduled_sampling"),
]:
    by_val = defaultdict(list)
    for e in experiments:
        val = e["features"].get(factor_key, "?")
        by_val[str(val)].append(e["lstm_mean"])
    
    if len(by_val) > 1:
        print(f"\n  {factor_name}:")
        for val, r2s in sorted(by_val.items(), key=lambda x: statistics.mean(x[1]), reverse=True):
            mean_r2 = statistics.mean(r2s)
            n_pos = sum(1 for r in r2s if r > 0)
            print(f"    {str(val):<25} mean_R²={mean_r2:>8.4f}  n={len(r2s):>3}  positive={n_pos}/{len(r2s)}")

# ── Direct comparison: what's different between good and bad? ──
print("\n" + "=" * 120)
print("GOOD vs BAD — DIRECT COMPARISON OF FEATURES")
print("=" * 120)

if good:
    print("\n  GOOD LSTM experiments (R² > 0.5):")
    for e in good:
        f = e["features"]
        print(f"    {e['name']:<40} R²={e['lstm_mean']:.4f}")
        print(f"      speed_mode={f['speed_mode']}, Ca={f['Ca']}, Cr={f['Cr']}, eta={f['eta']}")
        print(f"      H={f['hidden_units']}, L={f['num_layers']}, residual={f['residual']}, dropout={f['dropout']}")
        print(f"      transform={f['density_transform']}, mass_pp={f['mass_postprocess']}, shift_align={f['shift_align']}")
        print(f"      epochs={f['max_epochs']}, lag={f['lag']}, batch={f['batch_size']}, lr={f['lr']}")
        print(f"      T_train={f['T_train']}, T_test={f['T_test']}, subsample={f['subsample']}")
        print()

if bad[:5]:
    print("\n  WORST 5 LSTM experiments:")
    for e in bad[-5:]:
        f = e["features"]
        print(f"    {e['name']:<40} R²={e['lstm_mean']:.4f}")
        print(f"      speed_mode={f['speed_mode']}, Ca={f['Ca']}, Cr={f['Cr']}, eta={f['eta']}")
        print(f"      H={f['hidden_units']}, L={f['num_layers']}, residual={f['residual']}, dropout={f['dropout']}")
        print(f"      transform={f['density_transform']}, mass_pp={f['mass_postprocess']}, shift_align={f['shift_align']}")
        print(f"      epochs={f['max_epochs']}, lag={f['lag']}, batch={f['batch_size']}, lr={f['lr']}")
        print(f"      T_train={f['T_train']}, T_test={f['T_test']}, subsample={f['subsample']}")
        print()

# Test horizon comparison
print("\n" + "=" * 120)
print("FORECAST HORIZON ANALYSIS")
print("=" * 120)
print(f"\n  {'Experiment':<40} {'LSTM_R2':>8} {'T_train':>8} {'T_test':>8} {'Ratio':>6}")
for e in experiments:
    f = e["features"]
    t_train = f["T_train"]
    t_test = f["T_test"]
    ratio = ""
    if isinstance(t_train, (int, float)) and isinstance(t_test, (int, float)) and t_train > 0:
        ratio = f"{t_test/t_train:.1f}x"
    print(f"  {e['name']:<40} {e['lstm_mean']:>8.4f} {str(t_train):>8} {str(t_test):>8} {ratio:>6}")
