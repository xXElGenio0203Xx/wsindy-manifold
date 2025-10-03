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
