"""Configuration utilities for rectangular collective motion simulations.

This module centralizes loading, merging, and validating configuration
values for the rectsim package. It provides:

- A `DEFAULT_CONFIG` describing reasonable defaults for simulation
    parameters, force constants, output controls and plotting options.
- `load_config(path, overrides)` which reads a user YAML file, merges it
    deeply with the defaults, applies dotted-key CLI overrides and runs
    a validation pass.
- Small helpers used during merging and override application. These are
    intentionally lightweight to avoid pulling in heavy dependencies at
    import time and to make behavior explicit for unit tests.

Design notes
------------
We prefer a simple nested dictionary config rather than heavy schema
libraries for two reasons: (1) it keeps the dependency surface small
and (2) it is easy to override fields from the command-line using
``--some.path value`` style arguments. The module performs basic value
validation and emits a ``RuntimeWarning`` for suspicious but
non-fatal settings (for example an unusually large time-step).
"""

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
    "model": "social_force",
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
    "model_config": {
        "speed": 0.5,
        "speed_mode": "constant",
    },
    "params": {
        "alpha": 1.5,
        "beta": 0.5,
        "Cr": 2.0,
        "Ca": 1.0,
        "lr": 0.9,
        "la": 1.0,
        "rcut_factor": 3.0,
        "mu_t": 1.0,
        "R": 1.0,  # Alignment radius (for Vicsek models)
        "alignment": {
            "enabled": False,
            "radius": 1.5,
            "rate": 0.1,
        },
    },
    "noise": {
        "kind": "gaussian",
        "eta": 0.3,
        "sigma": 0.2,
        "match_variance": True,
    },
    "forces": {
        "enabled": True,
        "type": "morse",
        "params": {
            "Cr": 2.0,
            "Ca": 1.0,
            "lr": 0.9,
            "la": 1.0,
            "rcut_factor": 3.0,
            "mu_t": 1.0,
        },
    },
    "ic": {
        "type": "uniform",
    },
    "ensemble": {
        "n_runs": 20,
        "seeds": None,
        "base_seed": 0,
        "ic_types": ["gaussian", "uniform", "ring", "cluster"],
        "ic_weights": None,
    },
    "rom": {
        "enabled": False,
        "rank": 10,
        "mvar_order": 4,
        "mvar_horizon": 20,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
    },
    "vicsek": {
        "seed": 0,
        "N": 400,
        "Lx": 20.0,
        "Ly": 20.0,
        "bc": "periodic",
        "T": 1000.0,
        "dt": 1.0,
        "v0": 1.0,
        "R": 1.0,
        "noise": {
            "kind": "gaussian",
            "sigma": 0.2,
            "eta": 0.4,
        },
        "save_every": 1,
        "neighbor_rebuild": 5,
        "out_dir": "outputs/vicsek",
    },
    "outputs": {
        "save_npz": True,
        "save_csv": True,
        "plots": True,
        "animate": True,  # Legacy field; kept for backward compatibility
        # Video generation controls
        "animate_traj": False,
        "animate_density": False,
        "video_ics": 1,  # Number of ICs for which we generate videos
        # Order parameter plot controls
        "plot_order_params": True,
        "order_params_ics": 1,  # Number of ICs for which we generate order parameter plots
        "grid_density": {
            "enabled": True,
            "nx": 128,
            "ny": 128,
            "bandwidth": 0.5,
        },
        "efrom": {
            "rank": 10,
            "order": 4,
            "horizon": 20,
        },
        "plot_options": {
            "traj_marker_size": 20,
            "traj_quiver": True,
            "traj_quiver_scale": 3.0,
            "traj_quiver_width": 0.004,
            "traj_quiver_alpha": 0.8,
        },
    },
}


class ConfigError(ValueError):
    """Raised when a configuration fails validation."""


def _deep_update(base: MutableMapping[str, Any], update: Mapping[str, Any]) -> None:
    """Recursively merge ``update`` into ``base`` in-place.

    This behaves similarly to ``dict.update`` but performs a deep
    recursive merge for nested mappings. Lists and non-mapping values
    are replaced by the value from ``update``. The function mutates
    ``base`` and returns ``None``.

    Example
    -------
    >>> base = {"a": 1, "b": {"x": 2, "y": 3}}
    >>> _deep_update(base, {"b": {"y": 9, "z": 10}})
    >>> base
    {"a": 1, "b": {"x": 2, "y": 9, "z": 10}}
    """

    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_update(base[key], value)  # type: ignore[index]
        else:
            base[key] = value


def _set_by_dotted_key(config: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    """Assign ``value`` into ``config`` using a dotted path.

    The dotted key syntax allows CLI overrides such as
    ``sim.N`` or ``params.alignment.radius``. Intermediate mappings are
    created as needed. This mutates ``config`` in-place.

    Notes
    -----
    - The function is intentionally permissive: if an intermediate key
      already exists but is not a mapping, it will be replaced by a
      mapping to allow the assignment to succeed. This mirrors the
      common expectations for CLI override behavior.
    """

    keys = dotted_key.split(".")
    target: MutableMapping[str, Any] = config
    for key in keys[:-1]:
        if key not in target or not isinstance(target[key], MutableMapping):
            target[key] = {}
        target = target[key]  # type: ignore[assignment]
    target[keys[-1]] = value


def _validate(config: Mapping[str, Any]) -> None:
    """Perform sanity checks on the merged configuration.

    This function raises :class:`ConfigError` for invalid values that
    would definitely prevent a correct simulation (for example non
    positive sizes or invalid integrator names). For settings that are
    unusual but not strictly invalid (e.g. a large time step) a
    ``RuntimeWarning`` is emitted instead.

    The checks are intentionally conservative; they aim to catch common
    user mistakes early while leaving room for advanced users to tweak
    unusual parameters.
    """

    model = config.get("model", "social_force")
    # Handle both old string format and new dict format
    if isinstance(model, dict):
        model_type = model.get("type", "social_force")
    else:
        model_type = model
    
    if model_type not in {"social_force", "vicsek_discrete", "discrete"}:
        raise ConfigError("Model type must be 'social_force', 'vicsek_discrete', or 'discrete'.")

    if model_type in {"vicsek_discrete", "discrete"}:
        vicsek = config.get("vicsek")
        if not isinstance(vicsek, Mapping):
            raise ConfigError("Vicsek configuration must be provided when model is 'vicsek_discrete'.")

        required = ["N", "Lx", "Ly", "bc", "T", "dt", "v0", "R", "save_every", "neighbor_rebuild"]
        missing = [key for key in required if key not in vicsek]
        if missing:
            raise ConfigError(f"Vicsek configuration missing keys: {', '.join(missing)}")

        if vicsek["N"] <= 0:
            raise ConfigError("Number of agents must be positive.")
        if vicsek["Lx"] <= 0 or vicsek["Ly"] <= 0:
            raise ConfigError("Domain dimensions must be positive.")
        if vicsek["dt"] <= 0:
            raise ConfigError("Time step must be positive.")
        if vicsek["T"] <= 0:
            raise ConfigError("Final time must be positive.")
        if vicsek["save_every"] <= 0:
            raise ConfigError("save_every must be positive.")
        if vicsek["neighbor_rebuild"] <= 0:
            raise ConfigError("neighbor_rebuild must be positive.")
        if vicsek["v0"] <= 0:
            raise ConfigError("Self-propulsion speed v0 must be positive.")
        if vicsek["R"] < 0:
            raise ConfigError("Vicsek interaction radius R must be non-negative.")
        if vicsek["bc"] not in {"periodic", "reflecting"}:
            raise ConfigError("Boundary condition must be 'periodic' or 'reflecting'.")

        noise = vicsek.get("noise", {})
        if noise.get("kind", "gaussian") not in {"gaussian", "uniform"}:
            raise ConfigError("Vicsek noise kind must be 'gaussian' or 'uniform'.")
        if noise.get("sigma", 0.0) < 0.0:
            raise ConfigError("Vicsek Gaussian noise sigma must be non-negative.")
        if noise.get("eta", 0.0) < 0.0:
            raise ConfigError("Vicsek uniform noise eta must be non-negative.")

        return

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
    if params.get("mu_t", 1.0) <= 0:
        raise ConfigError("mu_t must be positive.")

    align = params.get("alignment", {})
    if align.get("enabled"):
        if align.get("radius", 0) <= 0:
            raise ConfigError("Alignment radius must be positive.")
        if align.get("rate", 0) < 0:
            raise ConfigError("Alignment rate must be non-negative.")

    efrom_cfg = config.get("outputs", {}).get("efrom", {})
    if efrom_cfg:
        if efrom_cfg.get("rank", 1) <= 0:
            raise ConfigError("EF-ROM rank must be positive.")
        if efrom_cfg.get("order", 1) <= 0:
            raise ConfigError("EF-ROM VAR order must be positive.")
        if efrom_cfg.get("horizon", 1) <= 0:
            raise ConfigError("EF-ROM forecast horizon must be positive.")

    # Validate new video/plot controls
    outputs = config.get("outputs", {})
    if outputs.get("video_ics", 1) < 0:
        raise ConfigError("video_ics must be non-negative.")
    if outputs.get("order_params_ics", 1) < 0:
        raise ConfigError("order_params_ics must be non-negative.")

    # Warn about suspicious but non-fatal settings
    if sim["dt"] > 0.05:
        warnings.warn(
            "Time step dt > 0.05 may lead to unstable integration.",
            RuntimeWarning,
            stacklevel=2,
        )


def _apply_backward_compatibility(config: MutableMapping[str, Any]) -> None:
    """Apply backward compatibility rules for legacy config fields.
    
    Ensures old configs continue to work with new video/plot control fields.
    Specifically handles the legacy 'animate' field:
    - If 'animate' is True and 'animate_density' is not set, set animate_density=True
    - New explicit fields (animate_traj, animate_density) always override 'animate'
    """
    outputs = config.get("outputs", {})
    
    # Backward compatibility: if animate=True but animate_density not set, enable animate_density
    if outputs.get("animate", False) and "animate_density" not in outputs:
        outputs["animate_density"] = True


def load_config(path: str | Path, overrides: Iterable[tuple[str, Any]] | None = None) -> Dict[str, Any]:
    """Load a YAML configuration file and merge it with defaults.

    This helper performs the following steps:

    1. Create a deep copy of :data:`DEFAULT_CONFIG` so callers receive a
       fresh mutable dict.
    2. If ``path`` exists, load the YAML and deep-merge it into the
       defaults using :func:`_deep_update`.
    3. Apply any dotted-key overrides supplied by the CLI via
       :func:`_set_by_dotted_key`.
    4. Apply backward compatibility rules via :func:`_apply_backward_compatibility`.
    5. Run :func:`_validate` to perform sanity checks.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.
    overrides:
        Optional iterable of ``(key, value)`` pairs where ``key`` is a dotted
        path specifying the field to override (e.g. ``sim.N``).

    Returns
    -------
    dict
        The merged and validated configuration dictionary. The returned
        object is safe to mutate by the caller.
    """

    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    config_path = Path(path)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            user_cfg = yaml.safe_load(fh) or {}
        if not isinstance(user_cfg, Mapping):
            raise ConfigError("Configuration file must define a mapping.")
        
        # Detect OLD schema and provide clear error message
        old_schema_keys = ["domain", "particles", "dynamics", "integration"]
        detected_old_keys = [key for key in old_schema_keys if key in user_cfg]
        if detected_old_keys:
            raise ConfigError(
                f"OLD config schema detected in {config_path}. "
                f"Found keys: {', '.join(detected_old_keys)}.\n"
                f"The config schema has been updated. Please migrate your config file.\n"
                f"See MIGRATION_GUIDE.md for instructions, or use:\n"
                f"  python migrate_config.py {config_path} {config_path.stem}_new.yaml\n"
                f"Key changes: domain → sim, particles.N → sim.N, "
                f"dynamics → params/forces/noise, integration → sim"
            )
        
        _deep_update(cfg, user_cfg)
    else:
        raise FileNotFoundError(config_path)

    if overrides:
        for key, value in overrides:
            _set_by_dotted_key(cfg, key, value)

    _apply_backward_compatibility(cfg)
    _validate(cfg)
    return cfg


__all__ = ["load_config", "DEFAULT_CONFIG", "ConfigError"]
