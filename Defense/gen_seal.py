import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

BROWN = '#4E3629'
GOLD  = '#C09A6B'
WHITE = 'white'

fig, ax = plt.subplots(figsize=(4, 4), dpi=250)
fig.patch.set_facecolor('none')
fig.patch.set_alpha(0)
ax.set_facecolor('none')
ax.set_xlim(-1.20, 1.20)
ax.set_ylim(-1.20, 1.20)
ax.set_aspect('equal')
ax.axis('off')

# --- Outer brown ring ---
ax.add_patch(Circle((0, 0), 1.10, facecolor=BROWN, edgecolor='none', zorder=1))
ax.add_patch(Circle((0, 0), 1.04, facecolor='none', edgecolor=WHITE, lw=0.8, zorder=2))
ax.add_patch(Circle((0, 0), 0.89, facecolor=WHITE,  edgecolor=BROWN, lw=2.2, zorder=2))
ax.add_patch(Circle((0, 0), 0.93, facecolor='none', edgecolor=BROWN, lw=0.5, zorder=2))

# --- Arc text helpers ---
def arc_top(ax, text, r, a0, a1, **kw):
    """Top arc: character tops point outward."""
    for ch, a in zip(text, np.linspace(a0, a1, len(text))):
        rd = np.deg2rad(a)
        ax.text(r * np.cos(rd), r * np.sin(rd), ch,
                ha='center', va='center', rotation=a - 90, **kw)

def arc_bot(ax, text, r, a0, a1, **kw):
    """Bottom arc: character tops point inward (readable from outside)."""
    for ch, a in zip(text, np.linspace(a0, a1, len(text))):
        rd = np.deg2rad(a)
        ax.text(r * np.cos(rd), r * np.sin(rd), ch,
                ha='center', va='center', rotation=a + 90, **kw)

arc_top(ax, 'BROWN  UNIVERSITY', 0.965, 152, 28,
        color=WHITE, fontsize=9.5, fontweight='bold',
        fontfamily='serif', zorder=4)

arc_bot(ax, 'FOUNDED \u00b7 MDCCLXIV', 0.965, 210, 330,
        color=WHITE, fontsize=8.0, fontweight='bold',
        fontfamily='serif', zorder=4)

# Side diamond ornaments
for ang in [183, 357]:
    rd = np.deg2rad(ang)
    ax.text(0.965 * np.cos(rd), 0.965 * np.sin(rd), '\u25c6',
            ha='center', va='center', color=GOLD,
            fontsize=6, rotation=ang - 90, zorder=4)

# --- Inner central design ---
# 8-pointed star / sun
cx, cy = 0.0, 0.36
for i in range(8):
    a = np.deg2rad(i * 45)
    ax.plot([cx + 0.10 * np.cos(a), cx + 0.30 * np.cos(a)],
            [cy + 0.10 * np.sin(a), cy + 0.30 * np.sin(a)],
            color=BROWN, lw=2.5, solid_capstyle='round', zorder=3)
ax.add_patch(Circle((cx, cy), 0.115, facecolor=BROWN, edgecolor=BROWN, zorder=3))
ax.add_patch(Circle((cx, cy), 0.075, facecolor=WHITE, edgecolor=BROWN, lw=1.0, zorder=4))

# Double rule
for dy, lw in [(0.00, 1.4), (-0.025, 0.5)]:
    ax.plot([-0.50, 0.50], [dy, dy], color=BROWN, lw=lw, zorder=3)

# Motto
ax.text(0, -0.13, 'In Deo Speramus',
        ha='center', va='center', color=BROWN,
        fontsize=9.5, fontstyle='italic',
        fontfamily='serif', fontweight='bold', zorder=3)

# Three stars
for sx in [-0.28, 0.0, 0.28]:
    ax.text(sx, -0.32, '\u2605',
            ha='center', va='center', color=GOLD, fontsize=9, zorder=3)

# Open book
bx, by = 0.0, -0.57
# Left page
ax.fill([-0.38, -0.02, -0.02, -0.38],
        [by + 0.16, by + 0.22, by - 0.04, by - 0.04],
        facecolor=WHITE, edgecolor=BROWN, lw=1.2, zorder=3)
# Right page
ax.fill([0.02, 0.38, 0.38, 0.02],
        [by + 0.22, by + 0.16, by - 0.04, by - 0.04],
        facecolor=WHITE, edgecolor=BROWN, lw=1.2, zorder=3)
# Spine
ax.plot([bx, bx], [by + 0.22, by - 0.04],
        color=BROWN, lw=1.5, zorder=4)
# Page lines
for dy2 in [0.10, 0.04, -0.01]:
    ax.plot([-0.32, -0.07], [by + dy2, by + dy2],
            color=BROWN, lw=0.6, alpha=0.5, zorder=4)
    ax.plot([0.07, 0.32], [by + dy2, by + dy2],
            color=BROWN, lw=0.6, alpha=0.5, zorder=4)

plt.savefig('brown_seal.png', dpi=250, transparent=True,
            bbox_inches='tight', pad_inches=0.02)
print('brown_seal.png saved successfully')
