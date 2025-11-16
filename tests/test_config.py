import yaml
import json
from pathlib import Path

import pytest

from rectsim.config import load_config, ConfigError


def test_load_config_overrides(tmp_path: Path):
    # Write a minimal YAML that overrides N
    cfg_path = tmp_path / "small.yaml"
    cfg_path.write_text("""
sim:
  N: 10
""")

    cfg = load_config(str(cfg_path))
    assert cfg["sim"]["N"] == 10


def test_load_config_invalid_raises(tmp_path: Path):
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("""
sim:
  N: -5
""")

    with pytest.raises(ConfigError):
        load_config(str(cfg_path))


def test_vicsek_config_validation(tmp_path: Path):
    cfg_path = tmp_path / "vicsek.yaml"
    cfg_path.write_text(
        """
model: vicsek_discrete
vicsek:
  seed: 42
  N: 128
  Lx: 30.0
  Ly: 30.0
  bc: periodic
  T: 100.0
  dt: 1.0
  v0: 1.0
  R: 1.5
  noise:
    kind: gaussian
    sigma: 0.2
    eta: 0.4
  save_every: 10
  neighbor_rebuild: 2
  out_dir: outputs/test_vicsek
"""
    )

    cfg = load_config(str(cfg_path))
    assert cfg["model"] == "vicsek_discrete"
    assert cfg["vicsek"]["N"] == 128
    assert cfg["vicsek"]["noise"]["sigma"] == 0.2


def test_vicsek_invalid_radius(tmp_path: Path):
    cfg_path = tmp_path / "vicsek_bad.yaml"
    cfg_path.write_text(
        """
model: vicsek_discrete
vicsek:
  seed: 0
  N: 10
  Lx: 10.0
  Ly: 10.0
  bc: periodic
  T: 10.0
  dt: 1.0
  v0: 1.0
  R: -1.0
  noise:
    kind: gaussian
    sigma: 0.1
    eta: 0.4
  save_every: 1
  neighbor_rebuild: 1
  out_dir: outputs/test_vicsek
"""
    )

    with pytest.raises(ConfigError):
        load_config(str(cfg_path))
