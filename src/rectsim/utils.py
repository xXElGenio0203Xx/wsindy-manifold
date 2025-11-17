"""Utility functions for rectsim package.

This module provides helper utilities that don't fit in other modules,
including model ID generation for ensemble organization.
"""

from __future__ import annotations

from typing import Any, Dict


def generate_model_id(config: Dict[str, Any]) -> str:
    """Generate a reproducible model identifier from configuration.

    The model ID is used to organize ensemble simulations in directories
    like `simulations/<model_id>/run_0001/`. It encodes the key parameters
    that define the model dynamics.

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys:
        - model: str, model type (e.g. "social_force", "vicsek_discrete")
        - sim: dict with N, T, etc.
        - params: dict with force/interaction parameters
        - vicsek: dict (for vicsek_discrete model)

    Returns
    -------
    model_id : str
        A compact string identifier encoding the model type and key parameters.

    Examples
    --------
    >>> config = {
    ...     "model": "social_force",
    ...     "sim": {"N": 200, "T": 100.0},
    ...     "params": {"alpha": 1.5, "beta": 0.5, "Cr": 2.0, "Ca": 1.0,
    ...                "lr": 0.9, "la": 1.0}
    ... }
    >>> generate_model_id(config)
    'social_force_N200_T100'

    Notes
    -----
    The function aims for a balance between uniqueness and readability:
    - Includes model name and basic simulation parameters (N, T)
    - Includes key force/interaction parameters that affect dynamics
    - Omits parameters that don't significantly change behavior (dt, Lx, Ly)
    - Formats floats to avoid excessive decimal places
    """
    model = config.get("model", "social_force")
    
    # Start with model name
    parts = [_sanitize_name(model)]
    
    # Add basic simulation parameters
    sim = config.get("sim", {})
    if "N" in sim:
        parts.append(f"N{sim['N']}")
    if "T" in sim:
        parts.append(f"T{_format_float(sim['T'])}")
    
    # Add model-specific parameters
    if model == "vicsek_discrete":
        vicsek = config.get("vicsek", {})
        if "v0" in vicsek:
            parts.append(f"v0{_format_float(vicsek['v0'])}")
        if "R" in vicsek:
            parts.append(f"R{_format_float(vicsek['R'])}")
        
        noise = vicsek.get("noise", {})
        if noise.get("kind") == "gaussian" and "sigma" in noise:
            parts.append(f"sigma{_format_float(noise['sigma'])}")
        elif "eta" in noise:
            parts.append(f"eta{_format_float(noise['eta'])}")
    
    else:  # social_force or similar
        params = config.get("params", {})
        
        # Core force parameters
        if "alpha" in params:
            parts.append(f"alpha{_format_float(params['alpha'])}")
        if "beta" in params:
            parts.append(f"beta{_format_float(params['beta'])}")
        if "Cr" in params:
            parts.append(f"Cr{_format_float(params['Cr'])}")
        if "Ca" in params:
            parts.append(f"Ca{_format_float(params['Ca'])}")
        if "lr" in params:
            parts.append(f"lr{_format_float(params['lr'])}")
        if "la" in params:
            parts.append(f"la{_format_float(params['la'])}")
        
        # Alignment parameters if enabled
        align = params.get("alignment", {})
        if align.get("enabled", False):
            if "radius" in align:
                parts.append(f"alignR{_format_float(align['radius'])}")
            if "rate" in align:
                parts.append(f"alignRate{_format_float(align['rate'])}")
    
    return "_".join(parts)


def _sanitize_name(name: str) -> str:
    """Convert a model name to a filesystem-safe string.
    
    Examples
    --------
    >>> _sanitize_name("vicsek_discrete")
    'vicsek_discrete'
    >>> _sanitize_name("social force")
    'social_force'
    """
    return name.replace(" ", "_").replace("-", "_").lower()


def _format_float(value: float) -> str:
    """Format a float for use in a model ID string.
    
    Removes trailing zeros and unnecessary decimal points for cleaner IDs.
    
    Examples
    --------
    >>> _format_float(1.0)
    '1'
    >>> _format_float(1.5)
    '1.5'
    >>> _format_float(0.05)
    '0.05'
    >>> _format_float(100.0)
    '100'
    """
    # Format with reasonable precision
    formatted = f"{value:.6f}".rstrip("0").rstrip(".")
    return formatted
