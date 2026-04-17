"""
Figure 8.x — 2D test function footprint overlaid on a synthetic density field.

Generates a representative density snapshot (Gaussian-like cluster on a 64×64
periodic grid) and overlays the spatial support of one polynomial bump test
function with half-widths (ell_x*dx, ell_y*dy).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# ── synthetic density field (gas-like cluster) ──────────────────────
nx, ny = 64, 64
Lx, Ly = 25.0, 25.0
dx, dy = Lx / nx, Ly / ny
x = np.linspace(0, Lx, nx, endpoint=False)
y = np.linspace(0, Ly, ny, endpoint=False)
X, Y = np.meshgrid(x, y)

# Two overlapping Gaussian blobs to simulate a realistic density
rho = (
    80 * np.exp(-((X - 12) ** 2 + (Y - 14) ** 2) / (2 * 3.0**2))
    + 30 * np.exp(-((X - 16) ** 2 + (Y - 10) ** 2) / (2 * 2.5**2))
    + 5 * np.random.RandomState(42).randn(ny, nx)  # mild noise
)
rho = np.clip(rho, 0, None)

# ── test function parameters ────────────────────────────────────────
ell_x, ell_y = 10, 10  # half-widths in grid cells
centre_x, centre_y = 13.0, 12.5  # physical units

# ── figure ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.5))

im = ax.pcolormesh(x, y, rho, cmap="YlOrRd", shading="auto", rasterized=True)
cb = fig.colorbar(im, ax=ax, shrink=0.82, label=r"$\rho(x,y,t)$")

# Ellipse showing test function support
width = 2 * ell_x * dx   # physical diameter
height = 2 * ell_y * dy
ell_patch = Ellipse(
    (centre_x, centre_y),
    width, height,
    linewidth=2.0,
    edgecolor="royalblue",
    facecolor="cornflowerblue",
    alpha=0.25,
    linestyle="-",
    label=r"$\Psi_k$ support ($\ell_x=\ell_y=10$)",
)
ax.add_patch(ell_patch)

# Mark centre
ax.plot(centre_x, centre_y, "+", color="navy", ms=10, mew=2)

# Half-width annotation arrows (horizontal)
ax.annotate(
    "",
    xy=(centre_x + ell_x * dx, centre_y),
    xytext=(centre_x, centre_y),
    arrowprops=dict(arrowstyle="<->", color="navy", lw=1.5),
)
ax.text(
    centre_x + ell_x * dx / 2,
    centre_y + 0.6,
    rf"$\ell_x \Delta x$",
    ha="center",
    fontsize=10,
    color="navy",
)

# Half-width annotation arrows (vertical)
ax.annotate(
    "",
    xy=(centre_x, centre_y + ell_y * dy),
    xytext=(centre_x, centre_y),
    arrowprops=dict(arrowstyle="<->", color="navy", lw=1.5),
)
ax.text(
    centre_x + 0.7,
    centre_y + ell_y * dy / 2,
    rf"$\ell_y \Delta y$",
    ha="left",
    fontsize=10,
    color="navy",
)

ax.set_xlabel(r"$x$ [physical units]", fontsize=11)
ax.set_ylabel(r"$y$ [physical units]", fontsize=11)
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
ax.set_aspect("equal")
ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
ax.set_title("Test function support on density field", fontsize=10, pad=6)

fig.tight_layout()
out = "Thesis_Figures/fig8_test_footprint_2d.pdf"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
