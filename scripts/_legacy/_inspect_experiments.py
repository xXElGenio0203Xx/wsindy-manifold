#!/usr/bin/env python3
"""Print detailed summary of VDYN experiments."""
import json, os, glob, sys

pattern = sys.argv[1] if len(sys.argv) > 1 else "VDYN*"
for sf in sorted(glob.glob(f"oscar_output/{pattern}/summary.json")):
    s = json.load(open(sf))
    exp = os.path.basename(os.path.dirname(sf))
    m = s.get("mvar", {})
    l = s.get("lstm", {})
    p = s.get("pod", {})
    print(f"{'='*60}")
    print(f"  {exp}")
    print(f"{'='*60}")
    print(f"  train={s.get('n_train','?')}  test={s.get('n_test','?')}  d={p.get('r_pod','?')}  energy={p.get('energy_captured',0):.4f}")
    print(f"  transform={p.get('density_transform','?')}  latent_std={p.get('latent_standardize','?')}")
    print(f"  clamp={s.get('clamp_mode','?')}  mass_pp={s.get('mass_postprocess','?')}")
    if m:
        mr2 = m.get("mean_r2_test")
        mr2s = f"{mr2:.4f}" if mr2 is not None else "--"
        m1s = m.get("mean_r2_1step_test")
        m1ss = f"{m1s:.4f}" if m1s is not None else "--"
        neg = m.get("mean_negativity_frac")
        negs = f"{neg:.4f}" if neg is not None else "--"
        print(f"  MVAR:  R2_test={mr2s}  R2_1step={m1ss}  train_R2={m.get('r2_train',0):.4f}  lag={m.get('p_lag','?')}  alpha={m.get('ridge_alpha','?')}")
        print(f"         spectral_r={m.get('spectral_radius',0):.4f} -> {m.get('spectral_radius_after',0):.4f}  neg_frac={negs}")
    if l:
        lr2 = l.get("mean_r2_test")
        lr2s = f"{lr2:.4f}" if lr2 is not None else "--"
        print(f"  LSTM:  R2_test={lr2s}  val_loss={l.get('val_loss',0):.6f}  h={l.get('hidden_units','?')}  layers={l.get('num_layers','?')}")
    print(f"  time={s.get('total_time_minutes',0):.0f}m")
    # Check what config was used
    cfg_path = os.path.join(os.path.dirname(sf), "config_used.yaml")
    if os.path.exists(cfg_path):
        import yaml
        cfg = yaml.safe_load(open(cfg_path))
        sim = cfg.get("sim", {})
        model = cfg.get("model", {})
        noise = cfg.get("noise", {})
        forces = cfg.get("forces", {}).get("params", {})
        print(f"  CONFIG: N={sim.get('N')} T={sim.get('T')}s speed={model.get('speed')} speed_mode={model.get('speed_mode')}")
        print(f"          eta={noise.get('eta')} Ca={forces.get('Ca')} Cr={forces.get('Cr')} la={forces.get('la')} lr={forces.get('lr')}")
    print()
