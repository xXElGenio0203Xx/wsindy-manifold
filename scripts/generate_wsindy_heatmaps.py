#!/usr/bin/env python3
"""Generate WSINDy regime heatmap figures for ch9."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("Thesis_Figures")

REGIMES_AVAILABLE = {
    'gas': 'oscar_output/wsindy_v3/NDYN04_gas_wsindy_v3/WSINDy/multifield_model.json',
    'BH': 'oscar_output/wsindy_v3/NDYN05_blackhole_wsindy_v3/WSINDy/multifield_model.json',
    'SN': 'oscar_output/NDYN06_supernova_wsindy_v3/WSINDy/multifield_model.json',
    'CR': 'oscar_output/NDYN07_crystal_wsindy_v3/WSINDy/multifield_model.json',
    'PV': 'oscar_output/wsindy_v3/NDYN08_pure_vicsek_wsindy_v3/WSINDy/multifield_model.json',
    'gas_VS': 'oscar_output/NDYN04_gas_VS_wsindy_v3/WSINDy/multifield_model.json',
    'BH_VS': 'oscar_output/wsindy_v3/NDYN05_blackhole_VS_wsindy_v3/WSINDy/multifield_model.json',
    'SN_VS': 'oscar_output/wsindy_v3/NDYN06_supernova_VS_wsindy_v3/WSINDy/multifield_model.json',
    'CR_VS': 'oscar_output/NDYN07_crystal_VS_wsindy_v3/WSINDy/multifield_model.json',
}

ALL_REGIMES = ['gas', 'BH', 'SN', 'CR', 'PV', 'gas_VS', 'BH_VS', 'SN_VS', 'CR_VS']

# Load models
models = {}
for regime, path in REGIMES_AVAILABLE.items():
    with open(path) as f:
        models[regime] = json.load(f)

# Print summary
for regime, m in models.items():
    print(f"=== {regime} ===")
    for eq in ['rho', 'px', 'py']:
        coeffs = m[eq]['coefficients']
        r2 = m[eq].get('r2_weak', 'N/A')
        if isinstance(coeffs, dict):
            items = list(coeffs.items())
        else:
            items = list(zip(m[eq]['active_terms'], coeffs))
        print(f"  {eq} (R2_wf={r2}): {[(t, round(float(c), 6)) for t, c in items]}")
    print()


def make_regime_heatmap(target_eq, title_eq, figname):
    """Create cross-regime term heatmap for one equation."""
    # Collect all unique terms
    all_terms = set()
    for regime, m in models.items():
        coeffs = m[target_eq]['coefficients']
        if isinstance(coeffs, dict):
            all_terms.update(coeffs.keys())
        else:
            all_terms.update(m[target_eq]['active_terms'])
    all_terms = sorted(all_terms)

    # Build coefficient matrix
    regimes_with_data = [r for r in ALL_REGIMES if r in models]
    coeff_matrix = np.zeros((len(regimes_with_data), len(all_terms)))

    for i, regime in enumerate(regimes_with_data):
        m = models[regime]
        coeffs = m[target_eq]['coefficients']
        if isinstance(coeffs, dict):
            for term_name, coeff_val in coeffs.items():
                if term_name in all_terms:
                    j = all_terms.index(term_name)
                    coeff_matrix[i, j] = abs(float(coeff_val))
        else:
            terms = m[target_eq]['active_terms']
            for t, c in zip(terms, coeffs):
                j = all_terms.index(t)
                coeff_matrix[i, j] = abs(float(c))

    # Compute dominant balance ratio (normalize each row by max)
    row_max = coeff_matrix.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1  # avoid division by zero
    balance_matrix = coeff_matrix / row_max

    fig, ax = plt.subplots(figsize=(max(6, len(all_terms)*0.9), max(3, len(regimes_with_data)*0.6)))
    im = ax.imshow(balance_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(all_terms)))
    ax.set_xticklabels([t.replace('_', ' ') for t in all_terms], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(regimes_with_data)))
    ax.set_yticklabels(regimes_with_data, fontsize=9)

    # Annotate cells
    for i in range(len(regimes_with_data)):
        for j in range(len(all_terms)):
            val = balance_matrix[i, j]
            if val > 0.01:
                color = 'white' if val > 0.6 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=color)
            else:
                ax.text(j, i, '—', ha='center', va='center',
                        fontsize=7, color='#bbbbbb')

    # Add vertical separator between CS+PV and VS
    cs_count = sum(1 for r in regimes_with_data if r in ['gas', 'BH', 'SN', 'CR', 'PV'])
    if cs_count < len(regimes_with_data):
        ax.axhline(cs_count - 0.5, color='black', linewidth=2)

    ax.set_title(f'Normalised dominant balance ratio — {title_eq}', fontsize=11)
    plt.colorbar(im, ax=ax, label='$\\tilde{\\Pi}_m$', shrink=0.8)
    plt.tight_layout()
    fig.savefig(OUT / f'{figname}.pdf', dpi=300)
    fig.savefig(OUT / f'{figname}.png', dpi=300)
    plt.close(fig)
    print(f"  -> {figname}.pdf")


print("\nGenerating heatmaps...")
make_regime_heatmap('rho', '$\\rho_t$ equation', 'thesis_regime_heatmap_rho')
make_regime_heatmap('px', '$(p_x)_t$ equation', 'thesis_regime_heatmap_px')
print("Done!")
