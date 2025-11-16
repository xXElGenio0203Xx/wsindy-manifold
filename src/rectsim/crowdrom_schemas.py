"""
JSON Schema definitions for CrowdROM outputs.

Defines validation schemas for:
- run.json: Complete run configuration and metadata
- non_stationarity_report.json: ADF test results
"""

# run.json schema
RUN_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["meta", "simulation", "domain_grid", "kde", "pod", "non_stationarity", "movies", "io"],
    "properties": {
        "meta": {
            "type": "object",
            "required": ["run_id", "timestamp_utc", "seed", "code_version", "env"],
            "properties": {
                "run_id": {"type": "string"},
                "timestamp_utc": {"type": "string", "format": "date-time"},
                "seed": {"type": "integer"},
                "code_version": {
                    "type": "object",
                    "required": ["git_commit", "repo"],
                    "properties": {
                        "git_commit": {"type": "string"},
                        "repo": {"type": "string"}
                    }
                },
                "env": {
                    "type": "object",
                    "required": ["python", "numpy", "platform"],
                    "properties": {
                        "python": {"type": "string"},
                        "numpy": {"type": "string"},
                        "scipy": {"type": "string"},
                        "platform": {"type": "string"}
                    }
                }
            }
        },
        "simulation": {
            "type": "object",
            "required": ["model", "C"],
            "properties": {
                "model": {"type": "string"},
                "C": {"type": "integer", "minimum": 1},
                "T": {"type": "number"},
                "dt_micro": {"type": "number", "exclusiveMinimum": 0},
                "dt_obs": {"type": "number", "exclusiveMinimum": 0},
                "boundary_conditions": {"type": "object"},
                "integrator": {"type": "string"},
                "num_particles": {"type": "integer", "minimum": 1},
                "speeds": {"type": "object"},
                "noise": {"type": "object"},
                "forces": {"type": "object"},
                "cutoff_radius": {"type": "number"}
            }
        },
        "domain_grid": {
            "type": "object",
            "required": ["domain", "nx", "ny"],
            "properties": {
                "domain": {
                    "type": "object",
                    "required": ["xmin", "xmax", "ymin", "ymax"],
                    "properties": {
                        "xmin": {"type": "number"},
                        "xmax": {"type": "number"},
                        "ymin": {"type": "number"},
                        "ymax": {"type": "number"}
                    }
                },
                "nx": {"type": "integer", "minimum": 1},
                "ny": {"type": "integer", "minimum": 1},
                "dx": {"type": "number", "exclusiveMinimum": 0},
                "dy": {"type": "number", "exclusiveMinimum": 0},
                "obstacles": {"type": "array"}
            }
        },
        "kde": {
            "type": "object",
            "properties": {
                "kernel": {"type": "string"},
                "bandwidth_mode": {"type": "string"},
                "H": {"type": "object"},
                "periodic_x": {"type": "boolean"},
                "periodic_extension_n": {"type": "integer"}
            }
        },
        "pod": {
            "type": "object",
            "required": ["energy_threshold", "chosen_d", "svd"],
            "properties": {
                "energy_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                "chosen_d": {"type": "integer", "minimum": 1},
                "svd": {
                    "type": "object",
                    "required": ["method", "randomized"],
                    "properties": {
                        "method": {"type": "string"},
                        "randomized": {"type": "boolean"}
                    }
                }
            }
        },
        "non_stationarity": {
            "type": "object",
            "required": ["adf_alpha", "adf_max_lags", "trend_policy"],
            "properties": {
                "adf_alpha": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                "adf_max_lags": {"oneOf": [{"type": "integer", "minimum": 0}, {"type": "string", "enum": ["auto"]}]},
                "trend_policy": {"type": "string"}
            }
        },
        "movies": {
            "type": "object",
            "required": ["fps", "max_frames", "make_for"],
            "properties": {
                "fps": {"type": "integer", "minimum": 1},
                "max_frames": {"type": "integer", "minimum": 1},
                "make_for": {"type": "array", "items": {"type": "integer", "minimum": 1}}
            }
        },
        "io": {
            "type": "object",
            "required": ["save_dtype"],
            "properties": {
                "save_dtype": {"type": "string", "enum": ["float32", "float64"]}
            }
        }
    }
}

# non_stationarity_report.json schema
NONSTATIONARITY_REPORT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["adf_alpha", "adf_max_lags", "d", "C", "per_simulation"],
    "properties": {
        "adf_alpha": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
        "adf_max_lags": {"oneOf": [{"type": "integer", "minimum": 0}, {"type": "string", "enum": ["auto"]}]},
        "d": {"type": "integer", "minimum": 1},
        "C": {"type": "integer", "minimum": 1},
        "per_simulation": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["sim_id", "K", "decisions"],
                "properties": {
                    "sim_id": {"type": "integer", "minimum": 1},
                    "K": {"type": "integer", "minimum": 1},
                    "decisions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["coord", "mode", "adf_variant", "p_value", "lag"],
                            "properties": {
                                "coord": {"type": "integer", "minimum": 1},
                                "mode": {"type": "string", "enum": ["raw", "diff", "detrend", "seasonal_diff"]},
                                "adf_variant": {"type": "string", "enum": ["const", "trend", "ct"]},
                                "p_value": {"type": "number", "minimum": 0, "maximum": 1},
                                "lag": {"type": "integer", "minimum": 0},
                                "init_level": {"type": "number"},
                                "notes": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }
    }
}


def validate_run_json(data: dict) -> tuple[bool, str]:
    """
    Validate run.json against schema.
    
    Parameters
    ----------
    data : dict
        Parsed run.json data
    
    Returns
    -------
    valid : bool
        True if valid, False otherwise
    message : str
        Error message if invalid, empty string if valid
    """
    try:
        import jsonschema
        jsonschema.validate(instance=data, schema=RUN_JSON_SCHEMA)
        return True, ""
    except jsonschema.ValidationError as e:
        return False, f"Schema validation failed: {e.message}"
    except ImportError:
        # jsonschema not available, skip validation
        return True, "Warning: jsonschema not installed, skipping validation"


def validate_nonstationarity_report(data: dict) -> tuple[bool, str]:
    """
    Validate non_stationarity_report.json against schema.
    
    Parameters
    ----------
    data : dict
        Parsed non_stationarity_report.json data
    
    Returns
    -------
    valid : bool
        True if valid, False otherwise
    message : str
        Error message if invalid, empty string if valid
    """
    try:
        import jsonschema
        jsonschema.validate(instance=data, schema=NONSTATIONARITY_REPORT_SCHEMA)
        return True, ""
    except jsonschema.ValidationError as e:
        return False, f"Schema validation failed: {e.message}"
    except ImportError:
        # jsonschema not available, skip validation
        return True, "Warning: jsonschema not installed, skipping validation"
