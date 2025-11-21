"""import numpy as np

Tests for POD/SVD Restriction-Lifting (Alvarez et al., 2025)

from wsindy_manifold.latent.pod import fit_pod, restrict_movie, lift_pod

Validates:

- Mass normalization preservation through lifting

- Orthonormality of POD basisdef _normalise(rho, dx, dy):

- Reconstruction error monotonicity with d    rho = np.clip(rho, 0.0, None)

- Invariance of R/L under centering    rho /= rho.sum() * dx * dy

- Temporal covariance route equivalence to direct SVD    return rho

"""



import numpy as npdef test_pod_reconstruction_and_mass():

import pytest    nx, ny = 4, 4

from rectsim.pod import PODProjector    dx = dy = 0.25

    nc = nx * ny

    T = 20

def create_test_snapshots(n_t=100, n_x=40, n_y=20, n_modes=3, seed=42):    base = np.ones(nc)

    """    mode1 = np.linspace(-0.3, 0.3, nc)

    Create synthetic mass-normalized density snapshots with known structure.    mode2 = np.sin(np.linspace(0, np.pi, nc)) - 0.5

    

    Uses sum of Gaussian modes with time-varying amplitudes:    snapshots = []

    ρ(x, y, t) = Σ_i a_i(t) * exp(-[(x-μ_x_i)²/σ_x² + (y-μ_y_i)²/σ_y²])    for t in range(T):

            coeff1 = 0.1 * np.sin(2 * np.pi * t / T)

    Returns        coeff2 = 0.05 * np.cos(2 * np.pi * t / T)

    -------        rho = base + coeff1 * mode1 + coeff2 * mode2

    rho_array : ndarray of shape (n_t, n_x, n_y)        snapshots.append(_normalise(rho, dx, dy))

        Mass-normalized density snapshots    Rho = np.array(snapshots)

    delta_x, delta_y : float

        Grid spacings    model = fit_pod(Rho, energy_keep=0.999, dx=dx, dy=dy)

    """    assert model["energy_ratio"][model["Ud"].shape[1] - 1] >= 0.999

    rng = np.random.RandomState(seed)

        Y = restrict_movie(Rho, model)

    # Grid    recon = np.array([lift_pod(y, model) for y in Y])

    Lx, Ly = 20.0, 10.0

    delta_x = Lx / n_x    err = np.linalg.norm(recon - Rho, axis=1).mean()

    delta_y = Ly / n_y    assert err < 1e-8

    x = np.linspace(0, Lx, n_x, endpoint=False) + delta_x / 2    masses = recon.sum(axis=1) * dx * dy

    y = np.linspace(0, Ly, n_y, endpoint=False) + delta_y / 2    assert np.allclose(masses, 1.0, atol=1e-8)

    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Mode centers and widths
    centers_x = rng.uniform(Lx * 0.2, Lx * 0.8, n_modes)
    centers_y = rng.uniform(Ly * 0.2, Ly * 0.8, n_modes)
    sigma_x = Lx / 10
    sigma_y = Ly / 10
    
    # Time-varying amplitudes
    t = np.linspace(0, 2 * np.pi, n_t)
    amplitudes = np.zeros((n_modes, n_t))
    for i in range(n_modes):
        freq = (i + 1) * 0.5
        amplitudes[i, :] = 1.0 + 0.5 * np.sin(freq * t + rng.uniform(0, 2 * np.pi))
    
    # Build snapshots
    rho_array = np.zeros((n_t, n_x, n_y))
    for k in range(n_t):
        rho_k = np.zeros((n_x, n_y))
        for i in range(n_modes):
            gaussian = np.exp(
                -((X - centers_x[i])**2 / (2 * sigma_x**2) +
                  (Y - centers_y[i])**2 / (2 * sigma_y**2))
            )
            rho_k += amplitudes[i, k] * gaussian
        
        # Mass-normalize
        mass = np.sum(rho_k) * delta_x * delta_y
        rho_k /= mass
        rho_array[k, :, :] = rho_k
    
    return rho_array, delta_x, delta_y


class TestPODProjector:
    """Test suite for PODProjector class"""
    
    def setup_method(self):
        """Create test data for each test"""
        self.rho_array, self.delta_x, self.delta_y = create_test_snapshots(
            n_t=100, n_x=40, n_y=20, n_modes=3, seed=42
        )
        self.n_t, self.n_x, self.n_y = self.rho_array.shape
        self.n_c = self.n_x * self.n_y
    
    def test_mass_normalization_input(self):
        """Verify input snapshots are mass-normalized"""
        area = self.delta_x * self.delta_y
        masses = np.sum(self.rho_array.reshape(self.n_t, -1), axis=1) * area
        
        assert np.allclose(masses, 1.0, atol=1e-12), \
            f"Input not mass-normalized: masses range [{masses.min():.2e}, {masses.max():.2e}]"
    
    def test_mass_preservation_lifting(self):
        """
        Test that lifting preserves mass normalization:
        X_hat = U_d @ (U_d^T @ X_bar) + x_bar·1^T
        Assert: max |1^T X_hat - 1| ≤ tol_mass
        """
        projector = PODProjector(
            energy_threshold=0.99,
            tol_mass=1e-12,
            use_weighted_mass=True,
            delta_x=self.delta_x,
            delta_y=self.delta_y,
        )
        projector.fit(self.rho_array, self.delta_x, self.delta_y)
        
        # Transform and inverse transform
        Y_d = projector.transform(self.rho_array)
        X_hat = projector.inverse_transform(Y_d)
        
        # Check mass preservation
        mass_error = projector.mass_check(X_hat)
        
        assert mass_error <= 1e-12, \
            f"Mass not preserved: max error = {mass_error:.2e}"
        
        print(f"✓ Mass preservation: max error = {mass_error:.2e} (d={projector.d})")
    
    def test_orthonormality(self):
        """
        Test that POD basis is orthonormal:
        U_d^T @ U_d = I_d
        Assert: ||U_d^T @ U_d - I_d||_max ≤ 1e-12
        """
        projector = PODProjector(energy_threshold=0.95)
        projector.fit(self.rho_array, self.delta_x, self.delta_y)
        
        ortho_error = projector.orthonormality_check()
        
        assert ortho_error <= 1e-12, \
            f"Basis not orthonormal: max error = {ortho_error:.2e}"
        
        print(f"✓ Orthonormality: max error = {ortho_error:.2e}")
    
    def test_reconstruction_error_monotonicity(self):
        """
        Test that reconstruction error decreases monotonically with d.
        """
        X = self.rho_array.reshape(self.n_t, self.n_c).T  # (n_c, n_t)
        
        d_values = [1, 2, 3, 5, 10]
        mean_errors = []
        
        for d in d_values:
            projector = PODProjector(fixed_d=d)
            projector.fit(self.rho_array, self.delta_x, self.delta_y)
            
            e2_rec = projector.reconstruction_error(X)
            mean_error = np.mean(e2_rec)
            mean_errors.append(mean_error)
        
        # Check monotonic decrease
        for i in range(len(mean_errors) - 1):
            assert mean_errors[i] >= mean_errors[i + 1], \
                f"Reconstruction error not monotonic: d={d_values[i]} error={mean_errors[i]:.3e} >= " \
                f"d={d_values[i+1]} error={mean_errors[i+1]:.3e}"
        
        print(f"✓ Reconstruction error monotonicity:")
        for d, err in zip(d_values, mean_errors):
            print(f"  d={d:2d}: mean error = {err:.6e}")
    
    def test_centering_invariance(self):
        """
        Test that R/L are invariant under centering:
        y_k = U_d^T (x_k - x_bar) should equal column k of U_d^T X_bar
        """
        projector = PODProjector(energy_threshold=0.99)
        projector.fit(self.rho_array, self.delta_x, self.delta_y)
        
        # Method 1: Transform batch
        Y_d_batch = projector.transform(self.rho_array)  # (d, n_t)
        
        # Method 2: Transform one at a time
        Y_d_single = np.zeros((projector.d, self.n_t))
        for k in range(self.n_t):
            Y_d_single[:, k] = projector.restrict_one(self.rho_array[k, :, :])
        
        # Compare
        max_diff = np.max(np.abs(Y_d_batch - Y_d_single))
        
        assert max_diff <= 1e-12, \
            f"Centering invariance violated: max diff = {max_diff:.2e}"
        
        print(f"✓ Centering invariance: max diff = {max_diff:.2e}")
    
    def test_temporal_covariance_route(self):
        """
        Test that temporal covariance route gives equivalent results to direct SVD
        (up to sign flips in singular vectors).
        """
        # Direct SVD
        proj_direct = PODProjector(
            energy_threshold=0.95,
            temporal_cov=False,
        )
        proj_direct.fit(self.rho_array, self.delta_x, self.delta_y)
        
        # Temporal covariance route
        proj_cov = PODProjector(
            energy_threshold=0.95,
            temporal_cov=True,
        )
        proj_cov.fit(self.rho_array, self.delta_x, self.delta_y)
        
        # Same dimension chosen
        assert proj_direct.d == proj_cov.d, \
            f"Different dimensions: direct={proj_direct.d}, cov={proj_cov.d}"
        
        # Same singular values
        s_diff = np.max(np.abs(proj_direct.s - proj_cov.s))
        assert s_diff <= 1e-10, \
            f"Singular values differ: max diff = {s_diff:.2e}"
        
        # Same x_bar
        xbar_diff = np.max(np.abs(proj_direct.x_bar - proj_cov.x_bar))
        assert xbar_diff <= 1e-12, \
            f"Mean differs: max diff = {xbar_diff:.2e}"
        
        # U_d equivalent up to sign (check via reconstruction)
        X = self.rho_array.reshape(self.n_t, self.n_c).T
        e2_direct = proj_direct.reconstruction_error(X)
        e2_cov = proj_cov.reconstruction_error(X)
        
        err_diff = np.max(np.abs(e2_direct - e2_cov))
        assert err_diff <= 1e-10, \
            f"Reconstruction errors differ: max diff = {err_diff:.2e}"
        
        print(f"✓ Temporal covariance route: d={proj_cov.d}, singular value diff={s_diff:.2e}")
    
    def test_energy_threshold_selection(self):
        """Test that energy threshold correctly selects dimension"""
        thresholds = [0.90, 0.95, 0.99, 0.999]
        
        for tau in thresholds:
            projector = PODProjector(energy_threshold=tau)
            projector.fit(self.rho_array, self.delta_x, self.delta_y)
            
            energy_achieved = projector.energy_curve[projector.d - 1]
            
            # Check that achieved energy meets threshold
            assert energy_achieved >= tau, \
                f"Energy threshold {tau} not met: achieved {energy_achieved:.4f}"
            
            # Check that d-1 would not meet threshold (unless d=1)
            if projector.d > 1:
                energy_prev = projector.energy_curve[projector.d - 2]
                assert energy_prev < tau, \
                    f"Dimension d={projector.d} too large: d-1 already achieves {energy_prev:.4f} >= {tau}"
            
            print(f"✓ Energy threshold τ={tau:.3f}: d={projector.d}, achieved={energy_achieved:.4f}")
    
    def test_fixed_dimension(self):
        """Test that fixed_d overrides energy threshold"""
        fixed_d = 5
        
        projector = PODProjector(
            energy_threshold=0.99,  # Would normally choose different d
            fixed_d=fixed_d,
        )
        projector.fit(self.rho_array, self.delta_x, self.delta_y)
        
        assert projector.d == fixed_d, \
            f"Fixed dimension not used: expected {fixed_d}, got {projector.d}"
        
        print(f"✓ Fixed dimension: d={projector.d}")
    
    def test_single_snapshot_operations(self):
        """Test restrict_one and lift_one for single snapshots"""
        projector = PODProjector(energy_threshold=0.95)
        projector.fit(self.rho_array, self.delta_x, self.delta_y)
        
        # Pick a random snapshot
        k = 42
        rho_k = self.rho_array[k, :, :]
        
        # Restrict
        y_k = projector.restrict_one(rho_k)
        assert y_k.shape == (projector.d,), \
            f"Restrict output shape mismatch: expected ({projector.d},), got {y_k.shape}"
        
        # Lift
        x_k_hat = projector.lift_one(y_k)
        assert x_k_hat.shape == (self.n_c,), \
            f"Lift output shape mismatch: expected ({self.n_c},), got {x_k_hat.shape}"
        
        # Check reconstruction
        x_k = rho_k.ravel()
        error = np.linalg.norm(x_k - x_k_hat) / np.linalg.norm(x_k)
        
        print(f"✓ Single snapshot: y shape={y_k.shape}, reconstruction error={error:.6e}")
    
    def test_save_load(self, tmp_path):
        """Test saving and loading projector"""
        # Fit projector
        projector = PODProjector(energy_threshold=0.95)
        projector.fit(self.rho_array, self.delta_x, self.delta_y)
        
        # Save
        filepath = tmp_path / "pod_test.npz"
        projector.save(str(filepath))
        
        # Load
        projector_loaded = PODProjector.load(str(filepath))
        
        # Check metadata
        assert projector_loaded.n_x == projector.n_x
        assert projector_loaded.n_y == projector.n_y
        assert projector_loaded.d == projector.d
        assert np.allclose(projector_loaded.x_bar, projector.x_bar)
        assert np.allclose(projector_loaded.U_d, projector.U_d)
        
        # Check functionality
        Y_d = projector_loaded.transform(self.rho_array)
        X_hat = projector_loaded.inverse_transform(Y_d)
        mass_error = projector_loaded.mass_check(X_hat)
        
        assert mass_error <= 1e-12, \
            f"Loaded projector mass error: {mass_error:.2e}"
        
        print(f"✓ Save/load: d={projector_loaded.d}, mass error={mass_error:.2e}")
    
    def test_metadata(self):
        """Test metadata extraction"""
        projector = PODProjector(energy_threshold=0.99)
        projector.fit(self.rho_array, self.delta_x, self.delta_y)
        
        metadata = projector.get_metadata()
        
        assert metadata["n_x"] == self.n_x
        assert metadata["n_y"] == self.n_y
        assert metadata["n_c"] == self.n_c
        assert metadata["n_t"] == self.n_t
        assert metadata["d"] == projector.d
        assert metadata["energy_threshold"] == 0.99
        assert 0 <= metadata["energy_achieved"] <= 1.0
        assert len(metadata["singular_values"]) > 0
        
        print(f"✓ Metadata: d={metadata['d']}, energy={metadata['energy_achieved']:.4f}")


def test_paper_example():
    """
    Test with paper-like parameters: corridor domain 48m×12m, grid 80×20.
    Verify typical latent dimension d ≈ 13 for τ=0.99.
    """
    # Paper parameters
    Lx, Ly = 48.0, 12.0
    n_x, n_y = 80, 20
    delta_x = Lx / n_x  # 0.6m
    delta_y = Ly / n_y  # 0.6m
    
    # Create test data (50 cases × 22 time steps = 1100 snapshots)
    n_cases = 50
    n_steps = 22
    n_t = n_cases * n_steps
    
    rho_array, _, _ = create_test_snapshots(
        n_t=n_t, n_x=n_x, n_y=n_y, n_modes=5, seed=123
    )
    
    # Fit with paper energy threshold
    projector = PODProjector(energy_threshold=0.99)
    projector.fit(rho_array, delta_x, delta_y)
    
    print(f"\n{'='*60}")
    print(f"PAPER EXAMPLE TEST (Alvarez et al., 2025)")
    print(f"{'='*60}")
    print(f"Domain: {Lx}m × {Ly}m")
    print(f"Grid: {n_x} × {n_y} (δx={delta_x:.2f}m, δy={delta_y:.2f}m)")
    print(f"Snapshots: {n_t} ({n_cases} cases × {n_steps} steps)")
    print(f"Energy threshold: τ={projector.energy_threshold}")
    print(f"Chosen dimension: d={projector.d}")
    print(f"Energy achieved: {projector.energy_curve[projector.d-1]:.6f}")
    print(f"Top 5 singular values: {projector.s[:5]}")
    
    # Mass check
    X = rho_array.reshape(n_t, n_x * n_y).T
    Y_d = projector.transform(rho_array)
    X_hat = projector.inverse_transform(Y_d)
    mass_error = projector.mass_check(X_hat)
    ortho_error = projector.orthonormality_check()
    
    print(f"\nValidation:")
    print(f"  Mass preservation: {mass_error:.2e}")
    print(f"  Orthonormality: {ortho_error:.2e}")
    
    # Reconstruction error stats
    e2_rec = projector.reconstruction_error(X)
    print(f"  Reconstruction error: mean={np.mean(e2_rec):.6e}, "
          f"median={np.median(e2_rec):.6e}, p90={np.percentile(e2_rec, 90):.6e}")
    
    print(f"{'='*60}\n")
    
    # Paper typically gets d ≈ 13, but our synthetic data may differ
    assert 5 <= projector.d <= 30, \
        f"Dimension d={projector.d} outside expected range [5, 30] for τ=0.99"
    assert mass_error <= 1e-12
    assert ortho_error <= 1e-12


if __name__ == "__main__":
    print("="*80)
    print("POD/SVD RESTRICTION-LIFTING TESTS (Alvarez et al., 2025)")
    print("="*80)
    
    # Run all tests
    test_obj = TestPODProjector()
    test_obj.setup_method()
    
    print("\n1. Mass Normalization Input")
    test_obj.test_mass_normalization_input()
    
    print("\n2. Mass Preservation through Lifting")
    test_obj.test_mass_preservation_lifting()
    
    print("\n3. Orthonormality")
    test_obj.test_orthonormality()
    
    print("\n4. Reconstruction Error Monotonicity")
    test_obj.test_reconstruction_error_monotonicity()
    
    print("\n5. Centering Invariance")
    test_obj.test_centering_invariance()
    
    print("\n6. Temporal Covariance Route")
    test_obj.test_temporal_covariance_route()
    
    print("\n7. Energy Threshold Selection")
    test_obj.test_energy_threshold_selection()
    
    print("\n8. Fixed Dimension")
    test_obj.test_fixed_dimension()
    
    print("\n9. Single Snapshot Operations")
    test_obj.test_single_snapshot_operations()
    
    print("\n10. Metadata")
    test_obj.test_metadata()
    
    print("\n11. Paper Example")
    test_paper_example()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80)
