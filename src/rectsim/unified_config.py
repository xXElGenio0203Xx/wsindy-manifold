"""Unified configuration schema for all simulation types.

This module defines the standardized configuration structure that works
across all model types (discrete Vicsek, continuous D'Orsogna, hybrids).
"""

from typing import Dict, Any, List
import numpy as np


# Default values for all configuration parameters
DEFAULTS = {
    # Domain
    'domain': {
        'Lx': 20.0,
        'Ly': 20.0,
        'bc': 'periodic',  # 'periodic' or 'reflecting'
    },
    
    # Particles
    'particles': {
        'N': 100,
        'initial_distribution': 'uniform',  # 'uniform', 'gaussian_cluster', 'two_clusters', 'ring'
        'initial_speed': 0.5,
        'velocity_distribution': 'random',  # 'random', 'aligned', 'counter_rotating'
    },
    
    # Model type
    'model': {
        'type': 'discrete',  # 'discrete' or 'continuous'
        'speed_mode': 'constant',  # 'constant' or 'variable' (discrete only)
        # constant: Traditional Vicsek, fixed speed, forces only shift positions
        # variable: Forces affect velocity, speed can change
    },
    
    # Dynamics parameters
    'dynamics': {
        # Alignment (Vicsek-style)
        'alignment': {
            'enabled': True,
            'radius': 2.0,
            'rate': 1.0,  # Rotational mobility μ_r (continuous only, ignored in discrete)
        },
        
        # Forces (Morse potential)
        'forces': {
            'enabled': False,
            'Cr': 2.0,  # Repulsion strength
            'Ca': 1.0,  # Attraction strength
            'lr': 0.5,  # Repulsion length scale
            'la': 1.5,  # Attraction length scale
            'rcut_factor': 5.0,  # Cutoff radius = rcut_factor * max(lr, la)
            'mu_t': 0.5,  # Translational mobility (discrete only)
        },
        
        # Noise
        'noise': {
            'type': 'gaussian',  # 'gaussian' or 'uniform'
            'eta': 0.3,  # Noise strength (radians)
            'match_variance': True,  # Match variance between noise types
            'Dtheta': 0.001,  # Rotational diffusion (continuous only)
        },
        
        # Self-propulsion (continuous D'Orsogna only)
        'self_propulsion': {
            'alpha': 1.5,  # Self-propulsion magnitude
            'beta': 1.0,   # Friction coefficient (natural speed = alpha/beta)
        },
    },
    
    # Integration parameters
    'integration': {
        'T': 100.0,  # Total simulation time
        'dt': 0.01,  # Time step
        'save_every': 10,  # Save every N steps
        'neighbor_rebuild': 5,  # Rebuild neighbor lists every N steps
        'integrator': 'euler',  # 'euler' or 'rk4' (continuous only)
        'seed': 42,  # Random seed
    },
    
    # Ensemble simulation
    'ensemble': {
        'cases': 1,  # Number of simulations with different initial conditions (C)
        'outputs': 1,  # Number of cases to generate visualizations for (O ≤ C)
        # All C cases always save trajectories and densities
        # Only first O cases generate animations and order parameter plots
    },
    
    # Output configuration
    'outputs': {
        'directory': 'outputs/simulation',
        'order_parameters': True,  # Compute all 4 order parameters
        'animations': True,
        'save_csv': True,
        'fps': 20,
        'density_resolution': 50,
        'arrows': {
            'enabled': True,  # Show velocity arrows in trajectory animation
            'scale': 'speed',  # 'speed' (proportional to speed) or 'uniform' (fixed length)
            'scale_factor': 1.0,  # Scaling factor for arrow sizes
        },
    },
}


def apply_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply default values to configuration, filling in missing fields.
    
    Parameters
    ----------
    config : dict
        User configuration (may be incomplete)
        
    Returns
    -------
    config : dict
        Complete configuration with defaults applied
    """
    def _merge_dicts(defaults: dict, user: dict) -> dict:
        """Recursively merge user config into defaults."""
        result = defaults.copy()
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = _merge_dicts(result[key], value)
            else:
                result[key] = value
        return result
    
    return _merge_dicts(DEFAULTS, config)


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate configuration and return list of errors.
    
    Parameters
    ----------
    config : dict
        Configuration to validate
        
    Returns
    -------
    errors : list of str
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check domain
    if 'domain' not in config:
        errors.append("Missing 'domain' section")
    else:
        domain = config['domain']
        if domain['Lx'] <= 0 or domain['Ly'] <= 0:
            errors.append("Domain dimensions must be positive")
        if domain['bc'] not in ['periodic', 'reflecting']:
            errors.append(f"Invalid boundary condition: {domain['bc']}")
    
    # Check particles
    if 'particles' not in config:
        errors.append("Missing 'particles' section")
    else:
        particles = config['particles']
        if particles['N'] <= 0:
            errors.append("Number of particles must be positive")
        if particles['initial_speed'] < 0:
            errors.append("Initial speed must be non-negative")
        valid_distributions = ['uniform', 'gaussian_cluster', 'gaussian', 'two_clusters', 'bimodal', 'ring', 'annulus']
        if particles['initial_distribution'] not in valid_distributions:
            errors.append(f"Invalid initial_distribution: {particles['initial_distribution']}")
    
    # Check model
    if 'model' not in config:
        errors.append("Missing 'model' section")
    else:
        model = config['model']
        if model['type'] not in ['discrete', 'continuous']:
            errors.append(f"Invalid model type: {model['type']}")
    
    # Check dynamics
    if 'dynamics' in config:
        dynamics = config['dynamics']
        
        # Alignment
        if 'alignment' in dynamics:
            align = dynamics['alignment']
            if align['radius'] <= 0:
                errors.append("Alignment radius must be positive")
            if align['rate'] < 0:
                errors.append("Alignment rate must be non-negative")
        
        # Forces
        if 'forces' in dynamics and dynamics['forces']['enabled']:
            forces = dynamics['forces']
            if forces['Cr'] < 0 or forces['Ca'] < 0:
                errors.append("Force strengths must be non-negative")
            if forces['lr'] <= 0 or forces['la'] <= 0:
                errors.append("Force length scales must be positive")
        
        # Noise
        if 'noise' in dynamics:
            noise = dynamics['noise']
            if noise['type'] not in ['gaussian', 'uniform']:
                errors.append(f"Invalid noise type: {noise['type']}")
            if noise['eta'] < 0:
                errors.append("Noise strength must be non-negative")
        
        # Self-propulsion
        if 'self_propulsion' in dynamics:
            sp = dynamics['self_propulsion']
            if sp['alpha'] < 0 or sp['beta'] <= 0:
                errors.append("alpha must be non-negative, beta must be positive")
    
    # Check integration
    if 'integration' not in config:
        errors.append("Missing 'integration' section")
    else:
        integ = config['integration']
        if integ['T'] <= 0 or integ['dt'] <= 0:
            errors.append("Time and timestep must be positive")
        if integ['save_every'] <= 0 or integ['neighbor_rebuild'] <= 0:
            errors.append("save_every and neighbor_rebuild must be positive")
        if integ['integrator'] not in ['euler', 'rk4', 'euler_semiimplicit']:
            errors.append(f"Invalid integrator: {integ['integrator']}")
        
        # Check stability for discrete model
        if config.get('model', {}).get('type') == 'discrete':
            v0 = config.get('particles', {}).get('initial_speed', 0.5)
            dt = integ['dt']
            R = config.get('dynamics', {}).get('alignment', {}).get('radius', 2.0)
            if v0 * dt > 0.5 * R:
                errors.append(
                    f"Stability condition violated: v0*dt={v0*dt:.3f} > 0.5*R={0.5*R:.3f}. "
                    f"Reduce dt or increase alignment radius."
                )
    
    return errors


def convert_to_legacy_format(unified_config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """Convert unified config to legacy format for backward compatibility.
    
    Parameters
    ----------
    unified_config : dict
        Unified configuration
    model_type : str
        'discrete' or 'continuous'
        
    Returns
    -------
    legacy_config : dict
        Configuration in legacy format
    """
    uc = unified_config
    
    if model_type == 'discrete':
        # Legacy format for vicsek_discrete.simulate_backend
        legacy = {
            'sim': {
                'N': uc['particles']['N'],
                'Lx': uc['domain']['Lx'],
                'Ly': uc['domain']['Ly'],
                'bc': uc['domain']['bc'],
                'T': uc['integration']['T'],
                'dt': uc['integration']['dt'],
                'save_every': uc['integration']['save_every'],
                'neighbor_rebuild': uc['integration']['neighbor_rebuild'],
                'seed': uc['integration']['seed'],
            },
            'model': {
                'type': 'vicsek_discrete',
                'speed': uc['particles']['initial_speed'],
                'speed_mode': uc['model'].get('speed_mode', 'constant'),
            },
            'params': {
                'R': uc['dynamics']['alignment']['radius'],
            },
            'noise': {
                'kind': uc['dynamics']['noise']['type'],
                'eta': uc['dynamics']['noise']['eta'],
                'match_variance': uc['dynamics']['noise']['match_variance'],
            },
            'forces': {
                'enabled': uc['dynamics']['forces']['enabled'],
            },
        }
        
        if uc['dynamics']['forces']['enabled']:
            legacy['forces']['type'] = 'morse'
            legacy['forces']['params'] = {
                'Cr': uc['dynamics']['forces']['Cr'],
                'Ca': uc['dynamics']['forces']['Ca'],
                'lr': uc['dynamics']['forces']['lr'],
                'la': uc['dynamics']['forces']['la'],
                'rcut_factor': uc['dynamics']['forces']['rcut_factor'],
                'mu_t': uc['dynamics']['forces']['mu_t'],
            }
        
    else:  # continuous
        # Legacy format for dynamics.simulate_backend
        legacy = {
            'sim': {
                'N': uc['particles']['N'],
                'Lx': uc['domain']['Lx'],
                'Ly': uc['domain']['Ly'],
                'bc': uc['domain']['bc'],
                'T': uc['integration']['T'],
                'dt': uc['integration']['dt'],
                'save_every': uc['integration']['save_every'],
                'neighbor_rebuild': uc['integration']['neighbor_rebuild'],
                'integrator': uc['integration']['integrator'],
            },
            'model': {
                'type': 'dorsogna',
            },
            'params': {
                'alpha': uc['dynamics']['self_propulsion']['alpha'],
                'beta': uc['dynamics']['self_propulsion']['beta'],
                'Cr': uc['dynamics']['forces']['Cr'],
                'Ca': uc['dynamics']['forces']['Ca'],
                'lr': uc['dynamics']['forces']['lr'],
                'la': uc['dynamics']['forces']['la'],
                'rcut_factor': uc['dynamics']['forces']['rcut_factor'],
            },
        }
        
        # Add alignment if enabled
        if uc['dynamics']['alignment']['enabled']:
            legacy['params']['alignment'] = {
                'enabled': True,
                'radius': uc['dynamics']['alignment']['radius'],
                'rate': uc['dynamics']['alignment']['rate'],
                'Dtheta': uc['dynamics']['noise']['Dtheta'],
            }
    
    return legacy


__all__ = [
    'DEFAULTS',
    'apply_defaults',
    'validate_config',
    'convert_to_legacy_format',
]
