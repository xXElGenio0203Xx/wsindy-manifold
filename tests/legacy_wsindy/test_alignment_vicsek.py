import numpy as np

from rectsim.dynamics import apply_alignment_step, vicsek_alignment_step
from wsindy_manifold.align_check import (
    angle_diff_mean,
    neighbor_finder_ball,
    order_parameter,
    step_yours_vs_gold,
)


def _random_headings(rng, n):
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return np.column_stack((np.cos(angles), np.sin(angles)))


def test_no_neighbors_no_noise_keeps_headings():
    rng = np.random.default_rng(0)
    x = np.array([[0.0, 0.0], [10.0, 10.0]])
    p = np.array([[1.0, 0.0], [0.0, 1.0]])
    p_new = apply_alignment_step(
        x,
        p,
        Lx=20.0,
        Ly=20.0,
        bc="periodic",
        mu_r=1.0,
        lV=0.1,
        Dtheta=0.0,
        dt=0.1,
        rng=rng,
    )
    np.testing.assert_allclose(p_new, p)


def test_zero_rate_gives_pure_noise():
    rng = np.random.default_rng(1)
    x = np.zeros((32, 2))
    p = _random_headings(rng, 32)
    p_new = apply_alignment_step(
        x,
        p,
        5.0,
        5.0,
        "reflecting",
        mu_r=0.0,
        lV=5.0,
        Dtheta=0.1,
        dt=0.05,
        rng=rng,
    )
    # Without drift the order parameter should not systematically increase
    psi_old = order_parameter(p)
    psi_new = order_parameter(p_new)
    assert abs(psi_new - psi_old) < 0.2


def test_global_consensus_without_noise():
    rng = np.random.default_rng(2)
    N = 64
    x = rng.uniform(0.0, 20.0, size=(N, 2))
    p = _random_headings(rng, N)
    dt = 0.02
    for _ in range(200):
        p = apply_alignment_step(
            x,
            p,
            20.0,
            20.0,
            "periodic",
            mu_r=1.0,
            lV=25.0,
            Dtheta=0.0,
            dt=dt,
            rng=rng,
        )
    assert order_parameter(p) > 0.97


def test_local_consensus_two_clusters():
    rng = np.random.default_rng(3)
    N = 40
    cluster_a = rng.normal(loc=[2.0, 2.0], scale=0.2, size=(N // 2, 2))
    cluster_b = rng.normal(loc=[8.0, 8.0], scale=0.2, size=(N // 2, 2))
    x = np.vstack([cluster_a, cluster_b])
    angles_a = rng.uniform(0, np.pi / 6, size=N // 2)
    angles_b = rng.uniform(np.pi / 2, np.pi / 2 + np.pi / 6, size=N // 2)
    p = np.vstack(
        [
            np.column_stack((np.cos(angles_a), np.sin(angles_a))),
            np.column_stack((np.cos(angles_b), np.sin(angles_b))),
        ]
    )
    dt = 0.05
    for _ in range(120):
        p = apply_alignment_step(
            x,
            p,
            10.0,
            10.0,
            "reflecting",
            mu_r=1.2,
            lV=1.0,
            Dtheta=0.0,
            dt=dt,
            rng=rng,
        )
    psi_global = order_parameter(p)
    psi_cluster_a = order_parameter(p[: N // 2])
    psi_cluster_b = order_parameter(p[N // 2 :])
    assert psi_cluster_a > 0.97
    assert psi_cluster_b > 0.97
    assert psi_global < 0.8


def test_alignment_matches_gold_reference():
    rng = np.random.default_rng(4)
    x = rng.uniform(0.0, 12.0, size=(30, 2))
    p = _random_headings(rng, 30)
    p_yours, p_gold = step_yours_vs_gold(
        x,
        p,
        Lx=12.0,
        Ly=12.0,
        bc="periodic",
        mu_r=0.8,
        lV=3.0,
        Dtheta=0.0,
        dt=0.05,
        seed=5,
    )
    assert angle_diff_mean(p_yours, p_gold) < 1e-3


def test_vicsek_step_matches_apply_when_deterministic():
    rng = np.random.default_rng(6)
    x = rng.uniform(0.0, 12.0, size=(40, 2))
    p = _random_headings(rng, 40)
    cell_list = None
    p_vec = vicsek_alignment_step(
        x,
        p,
        Lx=12.0,
        Ly=12.0,
        bc="periodic",
        lV=2.0,
        mu_r=0.9,
        Dtheta=0.0,
        dt=0.05,
        cell_list=cell_list,
        rng=rng,
    )
    p_apply = apply_alignment_step(
        x,
        p,
        12.0,
        12.0,
        "periodic",
        0.9,
        2.0,
        0.0,
        0.05,
        rng=rng,
    )
    assert angle_diff_mean(p_vec, p_apply) < 1e-3


def test_noise_variance_matches_expected_scale():
    rng = np.random.default_rng(5)
    N = 2000
    x = rng.uniform(0.0, 10.0, size=(N, 2))
    p = _random_headings(rng, N)
    dt = 0.01
    Dtheta = 0.2
    p_new = apply_alignment_step(
        x,
        p,
        10.0,
        10.0,
        "periodic",
        mu_r=0.0,
        lV=0.1,
        Dtheta=Dtheta,
        dt=dt,
        rng=rng,
    )
    angles_old = np.arctan2(p[:, 1], p[:, 0])
    angles_new = np.arctan2(p_new[:, 1], p_new[:, 0])
    delta = np.unwrap(angles_new - angles_old)
    var_emp = np.var(delta)
    var_theory = 2.0 * Dtheta * dt
    assert np.isclose(var_emp, var_theory, rtol=0.2)


def test_periodic_boundary_neighbor_detection():
    x = np.array([[0.1, 0.1], [9.9, 0.1]])
    neighbours = neighbor_finder_ball(x, Lx=10.0, Ly=10.0, bc="periodic", rcut=0.5)
    assert 1 in neighbours[0]
    assert 0 in neighbours[1]
