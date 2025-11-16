"""Top-level package for rectangular-domain collective motion simulations.

This package implements a lightweight particle-based simulator for
collective motion in a rectangular domain. It exposes two convenience
functions at package level:

- ``load_config(path, overrides)`` — load and validate a YAML configuration
    file (delegates to ``rectsim.config``).
- ``simulate(config)`` — run a simulation using a configuration dictionary
    (delegates to ``rectsim.dynamics``).

Why this file exists
--------------------
This module provides a minimal public API so callers (or tests) can do::

        from rectsim import load_config, simulate

without importing internal modules directly. It keeps the command-line
interface and tests decoupled from the package internals.

How it fits into the codebase
-----------------------------
The heavy lifting is done in the submodules under ``src/rectsim``; this
file only re-exports the commonly-used entry points.
"""

from __future__ import annotations

from typing import Any, Dict


def load_config(path: str, overrides=None) -> Dict[str, Any]:
    """Load a YAML configuration and apply optional overrides.

    How it works
    ------------
    This function is a thin wrapper that calls :func:`rectsim.config.load_config`.

    Why we need it
    --------------
    Exposing a package-level loader makes the public API simpler and
    mirrors common scientific-python patterns: a top-level import that
    delegates to a specialized implementation module.

    Returns
    -------
    dict
        A validated configuration dictionary ready to be passed to
        :func:`rectsim.simulate`.
    """

    from .config import load_config as _load_config

    return _load_config(path, overrides)


def simulate(config: Dict[str, Any]):
    """Run a simulation using a validated configuration dictionary (legacy API).

    How it works
    ------------
    This imports and calls :func:`rectsim.dynamics.simulate` which
    performs initialization (positions, velocities), integrates the
    equations of motion, and returns the trajectory and diagnostics.
    
    This is the legacy API that creates its own RNG from config["seed"].
    New code should use simulate_backend() with an explicit RNG for better control.

    Why we need it
    --------------
    A simple top-level function keeps examples and docs concise and
    focuses users on the high-level operation: load a config and simulate.

    Returns
    -------
    dict
        A dictionary containing trajectory arrays, times, and metadata.
    """

    from .dynamics import simulate as _simulate

    model = config.get("model", "social_force")
    if model == "vicsek_discrete":
        vicsek_cfg = config.get("vicsek")
        if vicsek_cfg is None:
            raise KeyError("Vicsek configuration expected under key 'vicsek'.")
        from .vicsek_discrete import simulate_vicsek

        return simulate_vicsek(vicsek_cfg)

    return _simulate(config)


def simulate_backend(config: Dict[str, Any], rng):
    """Unified backend interface for all model types (standardized API).
    
    This is the standardized API that accepts an explicit RNG and returns
    a consistent format across all model types (discrete/continuous).
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with model selection
    rng : np.random.Generator
        Random number generator instance
        
    Returns
    -------
    dict
        Standardized format: {"times", "traj", "vel", "meta"}
        - times: (T,) time points
        - traj: (T, N, 2) positions
        - vel: (T, N, 2) velocities  
        - meta: dict with config and diagnostics
    """
    model = config.get("model", {}).get("type", "dorsogna")
    
    if model == "vicsek_discrete":
        from .vicsek_discrete import simulate_backend as _vicsek_backend
        return _vicsek_backend(config, rng)
    elif model == "dorsogna":
        from .dynamics import simulate_backend as _dorsogna_backend
        return _dorsogna_backend(config, rng)
    else:
        raise ValueError(f"Unknown model type: {model}")


__all__ = ["load_config", "simulate", "simulate_backend"]
