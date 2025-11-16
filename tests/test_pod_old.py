import numpy as np

from wsindy_manifold.latent.pod import fit_pod, restrict_movie, lift_pod


def _normalise(rho, dx, dy):
    rho = np.clip(rho, 0.0, None)
    rho /= rho.sum() * dx * dy
    return rho


def test_pod_reconstruction_and_mass():
    nx, ny = 4, 4
    dx = dy = 0.25
    nc = nx * ny
    T = 20
    base = np.ones(nc)
    mode1 = np.linspace(-0.3, 0.3, nc)
    mode2 = np.sin(np.linspace(0, np.pi, nc)) - 0.5

    snapshots = []
    for t in range(T):
        coeff1 = 0.1 * np.sin(2 * np.pi * t / T)
        coeff2 = 0.05 * np.cos(2 * np.pi * t / T)
        rho = base + coeff1 * mode1 + coeff2 * mode2
        snapshots.append(_normalise(rho, dx, dy))
    Rho = np.array(snapshots)

    model = fit_pod(Rho, energy_keep=0.999, dx=dx, dy=dy)
    assert model["energy_ratio"][model["Ud"].shape[1] - 1] >= 0.999

    Y = restrict_movie(Rho, model)
    recon = np.array([lift_pod(y, model) for y in Y])

    err = np.linalg.norm(recon - Rho, axis=1).mean()
    assert err < 1e-8
    masses = recon.sum(axis=1) * dx * dy
    assert np.allclose(masses, 1.0, atol=1e-8)
