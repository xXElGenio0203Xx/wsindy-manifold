#!/usr/bin/env python3
"""Generate 16 lag-ablation configs (4 regimes x 4 IC criteria) for OSCAR."""
import yaml
import copy
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
phase_b_dir = base_dir / 'configs/systematic/thesis_final/phase_b'
out_dir = base_dir / 'configs/systematic/thesis_final/lag_ablation'
out_dir.mkdir(exist_ok=True)

# Lags from rom_hyperparameters/results_oscar_aligned/<regime>/summary.json
lag_table = {
    'NDYN04_gas':         {'bic': 2, 'hqic':  6, 'aic': 50, 'fpe': 42},
    'NDYN05_blackhole':   {'bic': 2, 'hqic': 10, 'aic': 50, 'fpe': 39},
    'NDYN06_supernova':   {'bic': 2, 'hqic':  4, 'aic': 50, 'fpe': 49},
    'NDYN08_pure_vicsek': {'bic': 2, 'hqic':  4, 'aic': 10, 'fpe': 10},
}

manifest_lines = []
for regime, lags in lag_table.items():
    base_path = phase_b_dir / f'{regime}.yaml'
    with open(base_path) as f:
        base_cfg = yaml.safe_load(f)
    for criterion, lag in lags.items():
        cfg = copy.deepcopy(base_cfg)
        cfg['experiment_name'] = f'{regime}_lag{lag}_{criterion}'
        cfg['rom']['models']['mvar']['lag'] = lag
        cfg['rom']['models']['lstm']['lag'] = lag
        # Disable WSINDy — ablation focuses on MVAR vs LSTM lag sensitivity
        if 'wsindy' in cfg.get('rom', {}).get('models', {}):
            cfg['rom']['models']['wsindy']['enabled'] = False
        fname = out_dir / f'{regime}_{criterion}.yaml'
        with open(fname, 'w') as f:
            f.write(f'---\n# {regime} -- lag ablation: {criterion.upper()}={lag}\n'
                    f'# Auto-generated from phase_b config\n\n')
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        rel = f'configs/systematic/thesis_final/lag_ablation/{regime}_{criterion}.yaml'
        manifest_lines.append(rel)
        print(f'  {fname.name}  lag={lag}')

manifest_path = out_dir / 'manifest.txt'
manifest_path.write_text('\n'.join(manifest_lines) + '\n')
print(f'\nWrote {len(manifest_lines)} configs to {out_dir}')
print(f'Manifest: {manifest_path}')
