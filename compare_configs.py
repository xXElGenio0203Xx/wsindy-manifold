#!/usr/bin/env python3
"""Compare physics & preprocessing between thesis_final and older experiments."""
import yaml, json, os, glob

configs = {}
for path in sorted(glob.glob('oscar_output/**/config_used.yaml', recursive=True)):
    try:
        with open(path) as f:
            c = yaml.safe_load(f)
        exp = c.get('experiment_name', os.path.basename(os.path.dirname(path)))
        configs[exp] = {
            'N': c.get('sim', {}).get('N'),
            'T_test': c.get('test_sim', {}).get('T'),
            'speed': c.get('model', {}).get('speed'),
            'speed_mode': c.get('model', {}).get('speed_mode'),
            'Ca': c.get('forces', {}).get('params', {}).get('Ca'),
            'Cr': c.get('forces', {}).get('params', {}).get('Cr'),
            'eta': c.get('noise', {}).get('eta'),
            'forces_enabled': c.get('forces', {}).get('enabled'),
            'density_transform': c.get('rom', {}).get('density_transform'),
            'mass_postprocess': c.get('rom', {}).get('mass_postprocess'),
            'shift_align': c.get('rom', {}).get('shift_align'),
            'mvar_lag': c.get('rom', {}).get('models', {}).get('mvar', {}).get('lag'),
        }
    except:
        pass

# Print thesis-final configs
thesis_finals = {k: v for k, v in configs.items() if 'thesis_final' in k}
fmt = "{:<48} {:>4} {:>6} {:>5} {:<22} {:>6} {:>6} {:>5} {:<6} {:<6} {:<6}"
hdr = fmt.format('Name', 'N', 'Ttest', 'spd', 'mode', 'Ca', 'Cr', 'eta', 'xform', 'mass', 'align')

print("THESIS-FINAL CONFIGS:")
print(hdr)
print("=" * len(hdr))
for name in sorted(thesis_finals):
    c = thesis_finals[name]
    print(fmt.format(name, c['N'] or 0, c['T_test'] or 0, c['speed'] or 0,
                     str(c['speed_mode'] or ''), c['Ca'] or 0, c['Cr'] or 0,
                     c['eta'] or 0, str(c['density_transform'] or ''),
                     str(c['mass_postprocess'] or ''), str(c['shift_align'] or '')))

print()
print("OTHER EXPERIMENTS WITH CONFIGS:")
print(hdr)
print("=" * len(hdr))
non_thesis = {k: v for k, v in configs.items() if 'thesis_final' not in k}
for name in sorted(non_thesis):
    c = non_thesis[name]
    print(fmt.format(name, c['N'] or 0, c['T_test'] or 0, c['speed'] or 0,
                     str(c['speed_mode'] or ''), c['Ca'] or 0, c['Cr'] or 0,
                     c['eta'] or 0, str(c['density_transform'] or ''),
                     str(c['mass_postprocess'] or ''), str(c['shift_align'] or '')))

# Now identify exact matches (same Ca, Cr, speed, speed_mode, eta)
print()
print("=" * 100)
print("EXACT PHYSICS MATCHES (same Ca, Cr, speed, speed_mode, eta):")
print("=" * 100)
for tf_name, tf_cfg in sorted(thesis_finals.items()):
    matches = []
    for name, cfg in sorted(non_thesis.items()):
        if (cfg['Ca'] == tf_cfg['Ca'] and cfg['Cr'] == tf_cfg['Cr'] and
            cfg['speed'] == tf_cfg['speed'] and cfg['speed_mode'] == tf_cfg['speed_mode'] and
            cfg['eta'] == tf_cfg['eta']):
            diffs = []
            if cfg['N'] != tf_cfg['N']:
                diffs.append("N={} vs {}".format(cfg['N'], tf_cfg['N']))
            if cfg['T_test'] != tf_cfg['T_test']:
                diffs.append("Ttest={} vs {}".format(cfg['T_test'], tf_cfg['T_test']))
            if cfg['density_transform'] != tf_cfg['density_transform']:
                diffs.append("xform={} vs {}".format(cfg['density_transform'], tf_cfg['density_transform']))
            if cfg['mass_postprocess'] != tf_cfg['mass_postprocess']:
                diffs.append("mass={} vs {}".format(cfg['mass_postprocess'], tf_cfg['mass_postprocess']))
            matches.append((name, diffs))
    if matches:
        print()
        print("  {} (thesis):".format(tf_name))
        for name, diffs in matches:
            diff_str = ", ".join(diffs) if diffs else "IDENTICAL CONFIG"
            print("    -> {} [{}]".format(name, diff_str))
