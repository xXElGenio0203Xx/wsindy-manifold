#!/usr/bin/env python3
"""Generate 24 LSTM preprocessing ablation configs.

Design:
  4 regimes  x  2 density_transform  x  3 mass_postprocess  =  24 tasks

Regimes (constant speed + variable speed pairs):
  NDYN04_gas, NDYN04_gas_VS, NDYN05_blackhole, NDYN05_blackhole_VS

Density transforms: sqrt, raw
Mass postprocessing: none, simplex, scale

Changes vs base (main_regimes/):
  - WSINDy disabled
  - LSTM max_epochs 300->150, patience 40->25  (light / fast)
  - density_transform and mass_postprocess swept
"""
import yaml
import copy
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
src_dir  = base_dir / 'configs/systematic/main_regimes'
out_dir  = base_dir / 'configs/systematic/preproc_ablation'
out_dir.mkdir(exist_ok=True)

REGIMES = [
    'NDYN04_gas',
    'NDYN04_gas_VS',
    'NDYN05_blackhole',
    'NDYN05_blackhole_VS',
]

TRANSFORMS = ['sqrt', 'raw']
MASS_MODES   = ['none', 'simplex', 'scale']

manifest_lines = []

for regime in REGIMES:
    base_path = src_dir / f'{regime}.yaml'
    with open(base_path) as f:
        base_cfg = yaml.safe_load(f)

    for transform in TRANSFORMS:
        for mass in MASS_MODES:
            cfg = copy.deepcopy(base_cfg)

            # ----- preprocessing -----
            cfg['rom']['density_transform'] = transform
            cfg['rom']['mass_postprocess']  = mass

            # ----- lightweight LSTM -----
            cfg['rom']['models']['lstm']['max_epochs'] = 150
            cfg['rom']['models']['lstm']['patience']   = 25

            # ----- disable WSINDy and MVAR -----
            if 'wsindy' in cfg:
                cfg['wsindy']['enabled'] = False
            cfg['rom']['models']['mvar']['enabled'] = False

            # ----- experiment name -----
            name = f'{regime}_preproc_{transform}_{mass}'
            cfg['experiment_name'] = name

            fname = out_dir / f'{name}.yaml'
            with open(fname, 'w') as f:
                f.write(
                    f'---\n'
                    f'# {regime} -- preprocessing ablation\n'
                    f'# density_transform={transform}  mass_postprocess={mass}\n'
                    f'# Light: max_epochs=150, patience=25, WSINDy disabled\n\n'
                )
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

            rel = f'configs/systematic/preproc_ablation/{name}.yaml'
            manifest_lines.append(rel)
            print(f'  {fname.name}')

manifest_path = out_dir / 'manifest.txt'
manifest_path.write_text('\n'.join(manifest_lines) + '\n')
print(f'\nWrote {len(manifest_lines)} configs to {out_dir}')
print(f'Manifest: {manifest_path}')
