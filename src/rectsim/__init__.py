"""Top-level package for rectangular domain collective motion simulations."""

from __future__ import annotations

from typing import Any, Dict


def load_config(path: str, overrides=None) -> Dict[str, Any]:
    """Load a simulation configuration file and optionally override entries."""

    from .config import load_config as _load_config

    return _load_config(path, overrides)


def simulate(config: Dict[str, Any]):
    """Run a simulation given a fully-formed configuration dictionary."""

    from .dynamics import simulate as _simulate

    return _simulate(config)


__all__ = ["load_config", "simulate"]
