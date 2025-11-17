"""Tests for utility functions."""

import pytest

from rectsim.utils import generate_model_id, _format_float, _sanitize_name


class TestFormatFloat:
    """Tests for float formatting utility."""

    def test_integer_removes_decimal(self):
        """Integers should be formatted without decimal point."""
        assert _format_float(1.0) == "1"
        assert _format_float(100.0) == "100"

    def test_preserves_decimals(self):
        """Non-zero decimals should be preserved."""
        assert _format_float(1.5) == "1.5"
        assert _format_float(0.05) == "0.05"
        assert _format_float(3.14159) == "3.14159"

    def test_removes_trailing_zeros(self):
        """Trailing zeros should be removed."""
        assert _format_float(1.50) == "1.5"
        assert _format_float(0.100) == "0.1"


class TestSanitizeName:
    """Tests for name sanitization utility."""

    def test_leaves_valid_names_unchanged(self):
        """Names with underscores should stay the same."""
        assert _sanitize_name("vicsek_discrete") == "vicsek_discrete"
        assert _sanitize_name("social_force") == "social_force"

    def test_replaces_spaces(self):
        """Spaces should be replaced with underscores."""
        assert _sanitize_name("social force") == "social_force"
        assert _sanitize_name("my model") == "my_model"

    def test_replaces_hyphens(self):
        """Hyphens should be replaced with underscores."""
        assert _sanitize_name("vicsek-discrete") == "vicsek_discrete"

    def test_converts_to_lowercase(self):
        """Names should be converted to lowercase."""
        assert _sanitize_name("SocialForce") == "socialforce"
        assert _sanitize_name("VICSEK") == "vicsek"


class TestGenerateModelID:
    """Tests for model ID generation."""

    def test_social_force_basic(self):
        """Social force model should include key parameters."""
        config = {
            "model": "social_force",
            "sim": {"N": 200, "T": 100.0},
            "params": {
                "alpha": 1.5,
                "beta": 0.5,
                "Cr": 2.0,
                "Ca": 1.0,
                "lr": 0.9,
                "la": 1.0,
            }
        }
        
        model_id = generate_model_id(config)
        
        # Should contain model name and key parameters
        assert "social_force" in model_id
        assert "N200" in model_id
        assert "T100" in model_id
        assert "alpha1.5" in model_id
        assert "beta0.5" in model_id
        assert "Cr2" in model_id
        assert "Ca1" in model_id
        assert "lr0.9" in model_id
        assert "la1" in model_id

    def test_vicsek_discrete_basic(self):
        """Vicsek discrete model should include noise parameters."""
        config = {
            "model": "vicsek_discrete",
            "sim": {"N": 400, "T": 1000.0},
            "vicsek": {
                "v0": 1.0,
                "R": 1.0,
                "noise": {
                    "kind": "gaussian",
                    "sigma": 0.2,
                }
            }
        }
        
        model_id = generate_model_id(config)
        
        assert "vicsek_discrete" in model_id
        assert "N400" in model_id
        assert "T1000" in model_id
        assert "v01" in model_id
        assert "R1" in model_id
        assert "sigma0.2" in model_id

    def test_vicsek_uniform_noise(self):
        """Vicsek with uniform noise should include eta parameter."""
        config = {
            "model": "vicsek_discrete",
            "sim": {"N": 400, "T": 1000.0},
            "vicsek": {
                "v0": 1.0,
                "R": 1.0,
                "noise": {
                    "kind": "uniform",
                    "eta": 0.4,
                }
            }
        }
        
        model_id = generate_model_id(config)
        
        assert "eta0.4" in model_id
        assert "sigma" not in model_id  # Should not include Gaussian param

    def test_with_alignment(self):
        """Model with alignment should include alignment parameters."""
        config = {
            "model": "social_force",
            "sim": {"N": 200, "T": 100.0},
            "params": {
                "alpha": 1.5,
                "beta": 0.5,
                "Cr": 2.0,
                "Ca": 1.0,
                "lr": 0.9,
                "la": 1.0,
                "alignment": {
                    "enabled": True,
                    "radius": 1.5,
                    "rate": 0.1,
                }
            }
        }
        
        model_id = generate_model_id(config)
        
        assert "alignR1.5" in model_id
        assert "alignRate0.1" in model_id

    def test_without_alignment(self):
        """Model without alignment should not include alignment parameters."""
        config = {
            "model": "social_force",
            "sim": {"N": 200, "T": 100.0},
            "params": {
                "alpha": 1.5,
                "beta": 0.5,
                "Cr": 2.0,
                "Ca": 1.0,
                "lr": 0.9,
                "la": 1.0,
                "alignment": {
                    "enabled": False,
                    "radius": 1.5,
                    "rate": 0.1,
                }
            }
        }
        
        model_id = generate_model_id(config)
        
        assert "alignR" not in model_id
        assert "alignRate" not in model_id

    def test_reproducible(self):
        """Same config should always produce same model ID."""
        config = {
            "model": "social_force",
            "sim": {"N": 200, "T": 100.0},
            "params": {
                "alpha": 1.5,
                "beta": 0.5,
                "Cr": 2.0,
                "Ca": 1.0,
                "lr": 0.9,
                "la": 1.0,
            }
        }
        
        model_id1 = generate_model_id(config)
        model_id2 = generate_model_id(config)
        
        assert model_id1 == model_id2

    def test_no_spaces_or_special_chars(self):
        """Model ID should be filesystem-safe (no spaces, etc)."""
        config = {
            "model": "social force",  # Space in name
            "sim": {"N": 200, "T": 100.0},
            "params": {
                "alpha": 1.5,
                "beta": 0.5,
                "Cr": 2.0,
                "Ca": 1.0,
                "lr": 0.9,
                "la": 1.0,
            }
        }
        
        model_id = generate_model_id(config)
        
        # Should not contain spaces or problematic characters
        assert " " not in model_id
        assert "/" not in model_id
        assert "\\" not in model_id

    def test_minimal_config(self):
        """Should work with minimal configuration."""
        config = {
            "model": "social_force",
            "sim": {},
            "params": {}
        }
        
        model_id = generate_model_id(config)
        
        # Should at least include model name
        assert "social_force" in model_id
        # Should not crash
        assert isinstance(model_id, str)
        assert len(model_id) > 0
