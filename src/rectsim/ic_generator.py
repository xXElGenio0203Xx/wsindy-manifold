"""
Initial Condition Generator Module
===================================

Generates training and test initial condition configurations.
Supports:
- Mixed distributions (gaussian, uniform, ring, two_clusters)
- Custom Gaussian experiments
- Interpolation/extrapolation tests
"""

import numpy as np


def generate_training_configs(train_ic_config, base_config):
    """
    Generate list of training run configurations.
    
    Parameters
    ----------
    train_ic_config : dict
        Training IC configuration from YAML
    base_config : dict
        Base simulation configuration
    
    Returns
    -------
    list
        List of configuration dictionaries with keys:
        - run_id: unique run identifier
        - distribution: IC type (gaussian_cluster, uniform, ring, two_clusters)
        - ic_params: parameters for IC generation
        - label: human-readable label
    """
    configs = []
    run_id = 0
    
    # Check if this is a simple Gaussian experiment (custom format)
    if 'center' in train_ic_config and 'variances' in train_ic_config:
        # Custom Gaussian format (from gaussians pipeline)
        center = train_ic_config['center']
        variances = train_ic_config['variances']
        n_samples = train_ic_config.get('n_samples_per_variance', 1)
        
        for var in variances:
            for sample in range(n_samples):
                configs.append({
                    'run_id': run_id,
                    'distribution': 'gaussian_cluster',
                    'ic_params': {
                        'center': (float(center[0]), float(center[1])),
                        'sigma': float(np.sqrt(var))
                    },
                    'label': f'gauss_center{center[0]:.1f},{center[1]:.1f}_var{var:.1f}_s{sample}'
                })
                run_id += 1
        return configs
    
    # Otherwise, use mixed distributions format
    ic_type = train_ic_config.get('type', 'mixed_comprehensive')
    
    # Gaussian configurations
    if train_ic_config.get('gaussian', {}).get('enabled', False):
        gauss_cfg = train_ic_config['gaussian']
        positions_x = gauss_cfg.get('positions_x', [])
        positions_y = gauss_cfg.get('positions_y', [])
        variances = gauss_cfg.get('variances', [])
        n_samples = gauss_cfg.get('n_samples_per_config', 1)
        
        for px in positions_x:
            for py in positions_y:
                for var in variances:
                    for sample in range(n_samples):
                        configs.append({
                            'run_id': run_id,
                            'distribution': 'gaussian_cluster',
                            'ic_params': {
                                'center': (float(px), float(py)),
                                'sigma': float(np.sqrt(var))
                            },
                            'label': f'gauss_x{px:.1f}_y{py:.1f}_var{var:.1f}_s{sample}'
                        })
                        run_id += 1
    
    # Uniform configurations
    if train_ic_config.get('uniform', {}).get('enabled', False):
        n_uniform = train_ic_config['uniform'].get('n_runs', 0)
        if n_uniform == 0:  # fallback to n_samples if n_runs not specified
            n_uniform = train_ic_config['uniform'].get('n_samples', 0)
        
        for i in range(n_uniform):
            configs.append({
                'run_id': run_id,
                'distribution': 'uniform',
                'ic_params': {},
                'label': f'uniform_s{i}'
            })
            run_id += 1
    
    # Ring configurations
    if train_ic_config.get('ring', {}).get('enabled', False):
        ring_cfg = train_ic_config['ring']
        radii = ring_cfg.get('radii', [])
        widths = ring_cfg.get('widths', [])
        n_samples = ring_cfg.get('n_samples_per_config', 1)
        
        for radius in radii:
            for width in widths:
                for sample in range(n_samples):
                    configs.append({
                        'run_id': run_id,
                        'distribution': 'ring',
                        'ic_params': {
                            'radius': float(radius),
                            'width': float(width)
                        },
                        'label': f'ring_r{radius:.1f}_w{width:.1f}_s{sample}'
                    })
                    run_id += 1
    
    # Two-cluster configurations
    if train_ic_config.get('two_clusters', {}).get('enabled', False):
        cluster_cfg = train_ic_config['two_clusters']
        separations = cluster_cfg.get('separations', [])
        sigmas = cluster_cfg.get('sigmas', [])
        n_samples = cluster_cfg.get('n_samples_per_config', 1)
        
        for sep in separations:
            for sigma in sigmas:
                for sample in range(n_samples):
                    configs.append({
                        'run_id': run_id,
                        'distribution': 'two_clusters',
                        'ic_params': {
                            'separation': float(sep),
                            'sigma': float(sigma)
                        },
                        'label': f'two_cluster_sep{sep:.1f}_sig{sigma:.1f}_s{sample}'
                    })
                    run_id += 1
    
    return configs


def generate_test_configs(test_ic_config, base_config):
    """
    Generate list of test run configurations.
    
    Parameters
    ----------
    test_ic_config : dict
        Test IC configuration from YAML
    base_config : dict
        Base simulation configuration
    
    Returns
    -------
    list
        List of test configuration dictionaries
    """
    configs = []
    run_id = 0
    
    # Check if this is a custom Gaussian experiment
    if 'centers' in test_ic_config and 'variance' in test_ic_config:
        # Custom Gaussian format (from gaussians pipeline)
        centers = test_ic_config['centers']
        variance = test_ic_config['variance']
        n_samples = test_ic_config.get('n_samples_per_center', 1)
        
        for center in centers:
            for sample in range(n_samples):
                configs.append({
                    'run_id': run_id,
                    'distribution': 'gaussian_cluster',
                    'ic_params': {
                        'center': (float(center[0]), float(center[1])),
                        'sigma': float(np.sqrt(variance))
                    },
                    'label': f'test_gauss_center{center[0]:.1f},{center[1]:.1f}_s{sample}'
                })
                run_id += 1
        return configs
    
    # Otherwise, use mixed distributions format
    
    # Gaussian test configurations
    if test_ic_config.get('gaussian', {}).get('enabled', False):
        gauss_cfg = test_ic_config['gaussian']
        
        # Interpolation tests
        test_positions_x = gauss_cfg.get('test_positions_x', [])
        test_positions_y = gauss_cfg.get('test_positions_y', [])
        test_variances = gauss_cfg.get('test_variances', [])
        
        for px in test_positions_x:
            for py in test_positions_y:
                for var in test_variances:
                    configs.append({
                        'run_id': run_id,
                        'distribution': 'gaussian_cluster',
                        'ic_params': {
                            'center': (float(px), float(py)),
                            'sigma': float(np.sqrt(var))
                        },
                        'label': f'test_gauss_interp_x{px:.1f}_y{py:.1f}_var{var:.1f}'
                    })
                    run_id += 1
        
        # Extrapolation tests
        extrap_positions = gauss_cfg.get('extrapolation_positions', [])
        extrap_variance = gauss_cfg.get('extrapolation_variance', [0.5])[0]
        
        for pos in extrap_positions:
            configs.append({
                'run_id': run_id,
                'distribution': 'gaussian_cluster',
                'ic_params': {
                    'center': (float(pos[0]), float(pos[1])),
                    'sigma': float(np.sqrt(extrap_variance))
                },
                'label': f'test_gauss_extrap_x{pos[0]:.1f}_y{pos[1]:.1f}'
            })
            run_id += 1
    
    # Uniform test configurations
    if test_ic_config.get('uniform', {}).get('enabled', False):
        n_uniform = test_ic_config['uniform'].get('n_runs', 0)
        if n_uniform == 0:
            n_uniform = test_ic_config['uniform'].get('n_samples', 0)
        
        for i in range(n_uniform):
            configs.append({
                'run_id': run_id,
                'distribution': 'uniform',
                'ic_params': {},
                'label': f'test_uniform_s{i}'
            })
            run_id += 1
    
    # Ring test configurations
    if test_ic_config.get('ring', {}).get('enabled', False):
        ring_cfg = test_ic_config['ring']
        test_radii = ring_cfg.get('test_radii', [])
        test_widths = ring_cfg.get('test_widths', [])
        n_samples = ring_cfg.get('n_samples_per_config', 1)
        
        for radius in test_radii:
            for width in test_widths:
                for sample in range(n_samples):
                    configs.append({
                        'run_id': run_id,
                        'distribution': 'ring',
                        'ic_params': {
                            'radius': float(radius),
                            'width': float(width)
                        },
                        'label': f'test_ring_r{radius:.1f}_w{width:.1f}_s{sample}'
                    })
                    run_id += 1
    
    # Two-cluster test configurations
    if test_ic_config.get('two_clusters', {}).get('enabled', False):
        cluster_cfg = test_ic_config['two_clusters']
        
        # Interpolation tests
        test_separations = cluster_cfg.get('test_separations', [])
        test_sigmas = cluster_cfg.get('test_sigmas', [])
        
        for sep in test_separations:
            for sigma in test_sigmas:
                configs.append({
                    'run_id': run_id,
                    'distribution': 'two_clusters',
                    'ic_params': {
                        'separation': float(sep),
                        'sigma': float(sigma)
                    },
                    'label': f'test_two_cluster_interp_sep{sep:.1f}_sig{sigma:.1f}'
                })
                run_id += 1
        
        # Extrapolation tests
        extrap_separations = cluster_cfg.get('extrapolation_separations', [])
        extrap_sigma = cluster_cfg.get('extrapolation_sigma', [1.0])[0]
        
        for sep in extrap_separations:
            configs.append({
                'run_id': run_id,
                'distribution': 'two_clusters',
                'ic_params': {
                    'separation': float(sep),
                    'sigma': float(extrap_sigma)
                },
                'label': f'test_two_cluster_extrap_sep{sep:.1f}'
            })
            run_id += 1
    
    return configs
