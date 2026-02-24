"""
Configuration Loading Module
=============================

Loads and parses YAML configuration files for the ROM-MVAR pipeline.
Supports multiple configuration formats:
- Mixed distributions (train_ic/test_ic with multiple types)
- Custom Gaussian experiments
- Flexible ROM parameters
"""

import yaml
from pathlib import Path


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str or Path
        Path to YAML configuration file
    
    Returns
    -------
    tuple
        (base_config, density_nx, density_ny, density_bandwidth,
         train_ic_config, test_ic_config, test_sim_config, rom_config, eval_config)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract simulation parameters
    sim_config = config.get('sim', {})
    base_config = {
        "sim": sim_config,
        "model": config.get('model', {}),
        "params": config.get('params', {}),
        "noise": config.get('noise', {}),
        "forces": config.get('forces', {}),
        "alignment": config.get('alignment', {}),
    }
    
    # Extract output/density parameters
    outputs = config.get('outputs', {})
    density_nx = outputs.get('density_resolution', 64)
    density_ny = outputs.get('density_resolution', 64)
    density_bandwidth = outputs.get('density_bandwidth', 2.0)
    
    # Extract ROM parameters
    rom_config = config.get('rom', {})
    
    # Extract IC configurations
    train_ic_config = config.get('train_ic', {})
    test_ic_config = config.get('test_ic', {})
    
    # Extract test simulation config (if separate from test_ic)
    test_sim_config = config.get('test_sim', {})
    
    # Extract evaluation config (optional, for time-resolved analysis)
    eval_config = config.get('evaluation', config.get('eval', {}))  # Support both names

    # Forward rom-level keys that the evaluator reads from eval_config.
    # mass_postprocess is logically a ROM postprocessing choice but the
    # evaluator historically reads it from eval_config.  If the user set it
    # under rom: (the documented location), copy it over so it takes effect.
    for _key in ('mass_postprocess', 'shift_align', 'shift_align_ref'):
        if _key in rom_config and _key not in eval_config:
            eval_config[_key] = rom_config[_key]
    
    return (base_config, density_nx, density_ny, density_bandwidth,
            train_ic_config, test_ic_config, test_sim_config, rom_config, eval_config)
