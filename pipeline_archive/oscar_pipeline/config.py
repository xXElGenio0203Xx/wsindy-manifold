"""
Shared configuration for Oscar pipeline

This ensures consistency across all Oscar scripts and with the local pipeline.
"""

# Fixed Vicsek-Morse discrete model configuration
BASE_CONFIG = {
    "sim": {
        "N": 40,
        "Lx": 15.0,
        "Ly": 15.0,
        "bc": "periodic",
        "T": 2.0,
        "dt": 0.1,
        "save_every": 1,
        "neighbor_rebuild": 5,
    },
    "model": {"speed": 1.0},
    "params": {"R": 2.0},
    "noise": {"kind": "gaussian", "eta": 0.3},
    "forces": {"enabled": False},
}

# IC types for stratified sampling
IC_TYPES = ["uniform", "gaussian_cluster", "ring", "two_clusters"]

# Training/test split
N_TRAIN = 100  # 25 per IC type
M_TEST = 20    # 5 per IC type

# Density estimation parameters
DENSITY_NX = 64
DENSITY_NY = 64
DENSITY_BANDWIDTH = 2.0

# POD/MVAR parameters
TARGET_ENERGY = 0.995  # 99.5% variance captured
P_LAG = 4
RIDGE_ALPHA = 1e-6
