"""Configuration utilities for rectangular collective motion simulations."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping

import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 0,
    "out_dir": "outputs/single",
    "device": "cpu",
    "sim": {
        "N": 200,
        "Lx": 20.0,
        "Ly": 20.0,
        "bc": "periodic",
        "T": 100.0,
        "dt": 0.01,
        "save_every": 10,
        "integrator": "rk4",
        "neighbor_rebuild": 10,
    },
    "params": {
        "alpha": 1.5,
        "beta": 0.5,
        "Cr": 2.0,
        "Ca": 1.0,
        "lr": 0.9,
        "la": 1.0,
        "rcut_factor": 3.0,
        "alignment": {
            "enabled": False,
            "radius": 1.5,
            "rate": 0.1,
        },
    },
    "outputs": {
        "save_npz": True,
        "save_csv": True,
        "plots": True,
        "animate": True,
        "grid_density": {
            "enabled": True,
            "nx": 128,
            "ny": 128,
            "bandwidth": 0.5,
        },
    },
}


class ConfigError(ValueError):
    """Raised when a configuration fails validation."""


def _deep_update(base: MutableMapping[str, Any], update: Mapping[str, Any]) -> None:
    """Recursively update a nested mapping."""

    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_update(base[key], value)  # type: ignore[index]
        else:
            base[key] = value


def _set_by_dotted_key(config: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    """Set a value in a nested mapping using a dotted key."""

    keys = dotted_key.split(".")
    target: MutableMapping[str, Any] = config
    for key in keys[:-1]:
        if key not in target or not isinstance(target[key], MutableMapping):
            target[key] = {}
        target = target[key]  # type: ignore[assignment]
    target[keys[-1]] = value


def _validate(config: Mapping[str, Any]) -> None:
    """Validate configuration values, raising :class:`ConfigError` if invalid."""

    sim = config["sim"]
    params = config["params"]

    if sim["N"] <= 0:
        raise ConfigError("Number of agents must be positive.")
    if sim["Lx"] <= 0 or sim["Ly"] <= 0:
        raise ConfigError("Domain dimensions must be positive.")
    if sim["dt"] <= 0:
        raise ConfigError("Time step must be positive.")
    if sim["T"] <= 0:
        raise ConfigError("Final time must be positive.")
    if sim["save_every"] <= 0:
        raise ConfigError("save_every must be positive.")
    if sim["neighbor_rebuild"] <= 0:
        raise ConfigError("neighbor_rebuild must be positive.")
    if sim["bc"] not in {"periodic", "reflecting"}:
        raise ConfigError("Boundary condition must be 'periodic' or 'reflecting'.")
    if sim["integrator"] not in {"rk4", "euler"}:
        raise ConfigError("Integrator must be 'rk4' or 'euler'.")

    if params["beta"] <= 0 or params["alpha"] <= 0:
        raise ConfigError("alpha and beta must be positive.")
    if params["Ca"] <= 0 or params["Cr"] <= 0:
        raise ConfigError("Ca and Cr must be positive.")
    if params["la"] <= 0 or params["lr"] <= 0:
        raise ConfigError("la and lr must be positive.")
    if params["rcut_factor"] <= 0:
        raise ConfigError("rcut_factor must be positive.")

    align = params.get("alignment", {})
    if align.get("enabled"):
        if align.get("radius", 0) <= 0:
            raise ConfigError("Alignment radius must be positive.")
        if align.get("rate", 0) < 0:
            raise ConfigError("Alignment rate must be non-negative.")

    if sim["dt"] > 0.05:
        warnings.warn(
            "Time step dt > 0.05 may lead to unstable integration.",
            RuntimeWarning,
            stacklevel=2,
        )


def load_config(path: str | Path, overrides: Iterable[tuple[str, Any]] | None = None) -> Dict[str, Any]:
    """Load a YAML configuration file and merge it with defaults.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.
    overrides:
        Optional iterable of ``(key, value)`` pairs where ``key`` is a dotted
        path specifying the field to override.

    Returns
    -------
    dict
        The merged and validated configuration dictionary.
    """

    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    config_path = Path(path)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            user_cfg = yaml.safe_load(fh) or {}
        if not isinstance(user_cfg, Mapping):
            raise ConfigError("Configuration file must define a mapping.")
        _deep_update(cfg, user_cfg)
    else:
        raise FileNotFoundError(config_path)

    if overrides:
        for key, value in overrides:
            _set_by_dotted_key(cfg, key, value)

    _validate(cfg)
    return cfg


__all__ = ["load_config", "DEFAULT_CONFIG", "ConfigError"]
