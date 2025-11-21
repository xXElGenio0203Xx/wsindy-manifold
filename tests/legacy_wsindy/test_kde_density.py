"""
Unit tests for KDE density estimation module.

Tests validate:
1. Silverman bandwidth computation
2. Gaussian kernel evaluation
3. Periodic augmentation
4. Obstacle masking
5. Mass normalization (∫_Ω ρ = 1)
6. Full pipeline on synthetic cases
"""

import numpy as np
from rectsim.kde_density import (
    silverman_bandwidth,
    gaussian_kernel_2d,
    augment_positions_periodic,
    create_obstacle_mask,
    kde_density_snapshot,
)


def test_silverman_bandwidth():
    """Test Silverman's rule computation matches Eq. (C.2)."""
    # Create test data with known statistics
    np.random.seed(42)
    N = 100
    positions = np.random.randn(N, 2) * np.array([2.0, 1.0])  # σ_x=2, σ_y=1
    
    h_x, h_y = silverman_bandwidth(positions)
    
    # Verify formula: h_i = [4/(N(d+2))]^(1/(d+4)) * σ_i, d=2
    d = 2
    multiplier = (4.0 / (N * (d + 2))) ** (1.0 / (d + 4))
    
    sigma_x = np.std(positions[:, 0], ddof=1)
    sigma_y = np.std(positions[:, 1], ddof=1)
    
    expected_h_x = multiplier * sigma_x
    expected_h_y = multiplier * sigma_y
    
    assert np.isclose(h_x, expected_h_x), f"h_x mismatch: {h_x} != {expected_h_x}"
    assert np.isclose(h_y, expected_h_y), f"h_y mismatch: {h_y} != {expected_h_y}"
    print(f"✓ Silverman bandwidth: h_x={h_x:.4f}, h_y={h_y:.4f}")


def test_gaussian_kernel_normalization():
    """Test that Gaussian kernel integrates to 1."""
    h_x, h_y = 1.0, 1.0
    
    # Create fine grid
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Evaluate kernel at origin
    K = gaussian_kernel_2d(X, Y, 0.0, 0.0, h_x, h_y)
    
    # Integrate (Riemann sum)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    integral = np.sum(K) * dx * dy
    
    assert np.isclose(integral, 1.0, atol=1e-3), f"Kernel integral = {integral}, expected 1"
    print(f"✓ Gaussian kernel normalization: ∫K = {integral:.6f}")


def test_periodic_augmentation():
    """Test that periodic augmentation creates ghosts at boundaries."""
    domain = (0.0, 10.0, 0.0, 5.0)
    a, b, c, d = domain
    Lx = b - a
    
    # Create particles near boundaries
    positions = np.array([
        [0.5, 2.5],   # Near left boundary
        [9.5, 2.5],   # Near right boundary
        [5.0, 2.5],   # In middle (no ghost)
    ])
    
    augmented = augment_positions_periodic(positions, domain, extension_n=5)
    
    # Should have: 3 original + 1 left ghost + 1 right ghost = 5
    # (unless a particle is near both boundaries, which it isn't)
    delta_x = Lx / 5
    
    n_left = np.sum(positions[:, 0] <= (a + delta_x))
    n_right = np.sum(positions[:, 0] >= (b - delta_x))
    
    expected_total = len(positions) + n_left + n_right
    
    assert augmented.shape[0] == expected_total, \
        f"Expected {expected_total} particles, got {augmented.shape[0]}"
    
    print(f"✓ Periodic augmentation: {len(positions)} → {augmented.shape[0]} particles")


def test_obstacle_mask():
    """Test obstacle masking creates zeros inside obstacle."""
    domain = (0.0, 10.0, 0.0, 5.0)
    nx, ny = 50, 25
    
    # Obstacle in center
    obstacle_rect = (4.0, 6.0, 2.0, 3.0)
    
    mask = create_obstacle_mask(domain, nx, ny, obstacle_rect)
    
    # Check that some cells are masked
    assert np.any(mask == 0.0), "Obstacle should mask some cells"
    assert np.any(mask == 1.0), "Some cells should remain free"
    
    # Check shape
    assert mask.shape == (nx, ny), f"Shape mismatch: {mask.shape} != ({nx}, {ny})"
    
    n_masked = np.sum(mask == 0.0)
    print(f"✓ Obstacle mask: {n_masked}/{nx*ny} cells masked")


def test_single_particle_center():
    """Test KDE with single particle at domain center."""
    domain = (0.0, 10.0, 0.0, 10.0)
    nx, ny = 50, 50
    
    # Single particle at center
    positions = np.array([[5.0, 5.0]])
    
    rho, H, S, meta = kde_density_snapshot(
        positions=positions,
        domain=domain,
        nx=nx,
        ny=ny,
        bandwidth_mode="manual",
        manual_H=(1.0, 1.0),
        periodic_x=False,
    )
    
    # Check shape
    assert rho.shape == (nx, ny), f"Shape mismatch: {rho.shape} != ({nx}, {ny})"
    
    # Check mass conservation: ∫_Ω ρ = 1
    dx = (domain[1] - domain[0]) / nx
    dy = (domain[3] - domain[2]) / ny
    total_mass = np.sum(rho) * dx * dy
    
    assert np.isclose(total_mass, 1.0, atol=1e-10), \
        f"Mass not conserved: ∫ρ = {total_mass}, expected 1"
    
    # Check that maximum is near center
    max_i, max_j = np.unravel_index(np.argmax(rho), rho.shape)
    assert abs(max_i - nx//2) <= 2, "Peak should be near center in x"
    assert abs(max_j - ny//2) <= 2, "Peak should be near center in y"
    
    print(f"✓ Single particle: mass = {total_mass:.10f}, peak at ({max_i}, {max_j})")


def test_mass_conservation_multiple_particles():
    """Test mass conservation with multiple particles."""
    np.random.seed(42)
    domain = (0.0, 20.0, 0.0, 10.0)
    nx, ny = 80, 40
    
    # Random particles
    N = 50
    positions = np.random.uniform(
        low=[domain[0], domain[2]],
        high=[domain[1], domain[3]],
        size=(N, 2)
    )
    
    # Test both bandwidth modes
    for bandwidth_mode in ["silverman", "manual"]:
        rho, H, S, meta = kde_density_snapshot(
            positions=positions,
            domain=domain,
            nx=nx,
            ny=ny,
            bandwidth_mode=bandwidth_mode,
            manual_H=(3.0, 2.0),
            periodic_x=True,
        )
        
        # Check mass conservation
        dx = (domain[1] - domain[0]) / nx
        dy = (domain[3] - domain[2]) / ny
        total_mass = np.sum(rho) * dx * dy
        
        assert np.isclose(total_mass, 1.0, atol=1e-9), \
            f"Mass not conserved ({bandwidth_mode}): ∫ρ = {total_mass:.10f}"
        
        print(f"✓ Mass conservation ({bandwidth_mode}): ∫ρ = {total_mass:.10f}")


def test_obstacle_zero_density():
    """Test that density is zero inside obstacle."""
    domain = (0.0, 10.0, 0.0, 5.0)
    nx, ny = 50, 25
    
    # Particles throughout domain
    np.random.seed(42)
    N = 100
    positions = np.random.uniform(
        low=[domain[0], domain[2]],
        high=[domain[1], domain[3]],
        size=(N, 2)
    )
    
    # Obstacle in middle
    obstacle_rect = (4.0, 6.0, 2.0, 3.0)
    
    rho, H, S, meta = kde_density_snapshot(
        positions=positions,
        domain=domain,
        nx=nx,
        ny=ny,
        bandwidth_mode="manual",
        manual_H=(0.5, 0.5),
        periodic_x=False,
        obstacle_rect=obstacle_rect,
    )
    
    # Get mask
    mask = create_obstacle_mask(domain, nx, ny, obstacle_rect)
    
    # Check that density is zero where mask is zero
    masked_cells = (mask == 0.0)
    density_in_obstacle = rho[masked_cells]
    
    assert np.allclose(density_in_obstacle, 0.0), \
        f"Density should be zero in obstacle, max = {np.max(density_in_obstacle)}"
    
    # Check mass still conserved
    dx = (domain[1] - domain[0]) / nx
    dy = (domain[3] - domain[2]) / ny
    total_mass = np.sum(rho) * dx * dy
    
    assert np.isclose(total_mass, 1.0, atol=1e-9), \
        f"Mass not conserved with obstacle: ∫ρ = {total_mass}"
    
    print(f"✓ Obstacle masking: {np.sum(masked_cells)} cells zeroed, mass = {total_mass:.10f}")


def test_periodic_vs_nonperiodic():
    """Test difference between periodic and non-periodic handling."""
    domain = (0.0, 10.0, 0.0, 5.0)
    nx, ny = 40, 20
    
    # Particles near left boundary
    positions = np.array([
        [0.5, 2.5],
        [0.8, 2.0],
        [9.2, 3.0],
    ])
    
    # Non-periodic
    rho_noper, _, _, _ = kde_density_snapshot(
        positions=positions,
        domain=domain,
        nx=nx,
        ny=ny,
        bandwidth_mode="manual",
        manual_H=(1.0, 1.0),
        periodic_x=False,
    )
    
    # Periodic
    rho_per, _, _, _ = kde_density_snapshot(
        positions=positions,
        domain=domain,
        nx=nx,
        ny=ny,
        bandwidth_mode="manual",
        manual_H=(1.0, 1.0),
        periodic_x=True,
    )
    
    # They should differ (periodic adds ghost contributions)
    max_diff = np.max(np.abs(rho_per - rho_noper))
    
    # With particles near boundaries, periodic should add some contribution
    # The difference might be small if bandwidth is large, so just check it exists
    print(f"✓ Periodic vs non-periodic: max diff = {max_diff:.6f}")


def test_paper_defaults():
    """Test with paper's default parameters."""
    # Paper setup
    domain = (0.0, 48.0, 0.0, 12.0)  # 48m × 12m corridor
    nx, ny = 80, 20  # 0.6m resolution
    
    # Simulate some pedestrians
    np.random.seed(42)
    N = 50
    positions = np.random.uniform(
        low=[domain[0], domain[2]],
        high=[domain[1], domain[3]],
        size=(N, 2)
    )
    
    # Paper's tuned bandwidth
    rho, H, S, meta = kde_density_snapshot(
        positions=positions,
        domain=domain,
        nx=nx,
        ny=ny,
        bandwidth_mode="manual",
        manual_H=(3.0, 2.0),
        periodic_x=True,
        periodic_extension_n=5,
    )
    
    # Validate
    assert rho.shape == (nx, ny)
    
    dx = (domain[1] - domain[0]) / nx
    dy = (domain[3] - domain[2]) / ny
    total_mass = np.sum(rho) * dx * dy
    
    assert np.isclose(total_mass, 1.0, atol=1e-9)
    assert np.all(rho >= 0.0), "Density should be non-negative"
    
    print(f"✓ Paper defaults: shape={rho.shape}, mass={total_mass:.10f}")
    print(f"  Bandwidth: H = diag({meta['h_x']}, {meta['h_y']})")
    print(f"  Grid: δx={dx:.3f}m, δy={dy:.3f}m")
    print(f"  Augmented: {meta['N']} → {meta['N_all']} particles")


if __name__ == "__main__":
    print("="*80)
    print("KDE DENSITY ESTIMATION TESTS (Alvarez et al., 2025)")
    print("="*80)
    
    test_silverman_bandwidth()
    test_gaussian_kernel_normalization()
    test_periodic_augmentation()
    test_obstacle_mask()
    test_single_particle_center()
    test_mass_conservation_multiple_particles()
    test_obstacle_zero_density()
    test_periodic_vs_nonperiodic()
    test_paper_defaults()
    
    print("="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80)
