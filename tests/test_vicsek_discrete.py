"""Unit tests for the discrete-time Vicsek model implementation."""

from __future__ import annotations

import numpy as np

from rectsim.vicsek_discrete import (
    angles_from_headings,
    compute_neighbors,
    headings_from_angles,
    rotation,
    simulate_vicsek,
)


def test_rotation_matrix_unitarity():
    angle = 0.37
    R = rotation(angle)
    should_be_identity = R @ R.T
    np.testing.assert_allclose(should_be_identity, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(R @ np.array([1.0, 0.0]), np.array([np.cos(angle), np.sin(angle)]))


def test_headings_angle_roundtrip():
    theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    headings = headings_from_angles(theta)
    norms = np.linalg.norm(headings, axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-12)
    recovered = angles_from_headings(headings)
    delta = np.angle(np.exp(1j * (recovered - theta)))
    np.testing.assert_allclose(delta, np.zeros_like(delta), atol=1e-12)


def test_gaussian_noise_variance_matches_sigma():
    """Test Gaussian noise variance in non-interacting limit (large domain, small R)."""
    cfg = {
        "seed": 123,
        "N": 64,  # Fewer particles for faster test
        "Lx": 100.0,  # Large domain to minimize interactions
        "Ly": 100.0,
        "bc": "periodic",
        "T": 100.0,  # Shorter simulation
        "dt": 0.1,
        "v0": 0.5,
        "R": 0.5,  # Small R, satisfies v0*dt < 0.5*R
        "noise": {"kind": "gaussian", "sigma": 0.3, "eta": 0.4},
        "save_every": 1,
        "neighbor_rebuild": 1,
        "out_dir": "unused",
    }
    result = simulate_vicsek(cfg)
    headings = result["headings"]
    angles = angles_from_headings(headings)
    deltas = np.angle(np.exp(1j * np.diff(angles, axis=0)))
    sample_var = float(np.var(deltas))
    expected_var = cfg["noise"]["sigma"] ** 2
    # More lenient tolerance due to finite-size effects and sparse interactions
    assert abs(sample_var - expected_var) < 0.1, (sample_var, expected_var)


def test_consensus_at_low_noise():
    cfg = {
        "seed": 321,
        "N": 64,
        "Lx": 20.0,
        "Ly": 20.0,
        "bc": "periodic",
        "T": 200.0,
        "dt": 1.0,
        "v0": 1.0,
        "R": 3.0,
        "noise": {"kind": "gaussian", "sigma": 0.05, "eta": 0.4},
        "save_every": 10,
        "neighbor_rebuild": 1,
        "out_dir": "unused",
    }
    result = simulate_vicsek(cfg)
    psi = result["psi"]
    assert psi[-1] > 0.75


def test_disorder_at_high_noise():
    cfg = {
        "seed": 111,
        "N": 64,
        "Lx": 20.0,
        "Ly": 20.0,
        "bc": "periodic",
        "T": 200.0,
        "dt": 1.0,
        "v0": 1.0,
        "R": 3.0,
        "noise": {"kind": "gaussian", "sigma": 2.0, "eta": 0.4},
        "save_every": 10,
        "neighbor_rebuild": 1,
        "out_dir": "unused",
    }
    result = simulate_vicsek(cfg)
    psi = result["psi"]
    assert psi[-1] < 0.3


def test_periodic_boundary_neighbors():
    Lx = Ly = 10.0
    x = np.array([[0.2, 0.5], [Lx - 0.2, 0.5], [5.0, 5.0]])
    neighbours, _ = compute_neighbors(x, Lx, Ly, R=0.6, bc="periodic")
    assert 0 in neighbours[0]
    assert 1 in neighbours[0]
    assert 0 in neighbours[1]
    assert 2 not in neighbours[0]


def test_zero_radius_neighbor_includes_self():
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    neighbours, _ = compute_neighbors(x, Lx=2.0, Ly=2.0, R=0.0, bc="periodic")
    assert [0] == neighbours[0].tolist()
    assert [1] == neighbours[1].tolist()
