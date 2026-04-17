"""
Figure 9.4 — KDE bandwidth limitation.

Plots the Morse interaction kernel W(r) = Ca*exp(-r/la) - Cr*exp(-r/lr)
for three regimes (gas, blackhole, supernova) and overlays the Gaussian
KDE bandwidth as a shaded region, showing that the repulsive core is
washed out by the smoothing scale.

KDE bandwidth:  sigma = 5 grid cells, dx = Lx/nx = 25/64
=> h = sigma * dx = 5 * (25/64) ≈ 1.953 physical units
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ── regime parameters ────────────────────────────────────────────────
regimes = {
    "Gas (CS)"       : dict(Ca=0.5,  la=1.5, Cr=0.2,  lr=0.5,  color="#1f77b4", ls="-"),
    "Blackhole (CS)" : dict(Ca=20.0, la=1.5, Cr=0.05, lr=0.3,  color="#ff7f0e", ls="--"),
    "Supernova (CS)" : dict(Ca=0.05, la=1.5, Cr=15.0, lr=0.5,  color="#2ca02c", ls="-."),
}

# KDE bandwidth in physical units
h_kde = 5.0 * (25.0 / 64.0)   # ≈ 1.953

r = np.linspace(0.0, 5.0, 2000)

fig, ax = plt.subplots(figsize=(6.5, 3.8))

# ── bandwidth shaded band ────────────────────────────────────────────
ax.axvspan(0, h_kde, color="grey", alpha=0.18, label=f"KDE bandwidth $h\\approx{h_kde:.2f}$")
ax.axvline(h_kde, color="grey", lw=1.0, ls=":")

# ── Morse kernels ────────────────────────────────────────────────────
for label, p in regimes.items():
    W = p["Ca"] * np.exp(-r / p["la"]) - p["Cr"] * np.exp(-r / p["lr"])
    # normalise so peak attraction lobe ≈ 1 for visual legibility
    peak = p["Ca"] * np.exp(-0.0 / p["la"]) - p["Cr"] * np.exp(-0.0 / p["lr"])
    W_norm = W / max(abs(peak), 1e-8)
    ax.plot(r, W_norm, color=p["color"], ls=p["ls"], lw=1.8, label=label)

# ── repulsion range annotations ──────────────────────────────────────
for p_name, p in regimes.items():
    lr = p["lr"]
    # small tick marker at lr
    ax.axvline(lr, color=p["color"], lw=0.6, ls=":", alpha=0.55)

# ── axis labels & legend ─────────────────────────────────────────────
ax.axhline(0, color="black", lw=0.7)
ax.set_xlabel(r"Distance $r$ [physical units]", fontsize=11)
ax.set_ylabel(r"Normalised $W(r)$", fontsize=11)
ax.set_xlim(0, 5.0)
ax.set_ylim(-2.2, 1.4)
ax.tick_params(labelsize=9)

# annotation arrow pointing into bandwidth region
ax.annotate(
    r"$\ell_r \lesssim h_{\rm KDE}$" "\n(washed out)",
    xy=(0.4, -0.5), xytext=(1.5, -1.5),
    fontsize=8.5, color="dimgrey",
    arrowprops=dict(arrowstyle="->", color="dimgrey", lw=0.9),
)

ax.legend(fontsize=8.5, loc="upper right", framealpha=0.85)
ax.set_title("KDE bandwidth vs. Morse interaction scales", fontsize=10, pad=6)

fig.tight_layout()

out = "Thesis_Figures/fig9_4_bandwidth_limitation.pdf"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
