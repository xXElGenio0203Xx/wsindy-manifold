#!/usr/bin/env python3
"""Read all suite experiment results from oscar_output/."""
import json, os, glob

patterns = [
    'oscar_output/suite_S0_*/summary.json',
    'oscar_output/suite_S1_*/summary.json',
    'oscar_output/suite_A_*/summary.json',
]
results = []
for pat in patterns:
    for f in sorted(glob.glob(pat)):
        try:
            d = json.load(open(f))
            name = os.path.basename(os.path.dirname(f))
            r2r = d.get('r2_rollout', '?')
            r2s = d.get('r2_1step', '?')
            rho = d.get('mvar', {}).get('spectral_radius', d.get('spectral_radius', '?'))
            neg = d.get('neg_density_pct', '?')
            dim = d.get('pod', {}).get('modes', d.get('pod_modes', '?'))
            E = d.get('pod', {}).get('energy_captured', '?')
            lstd = d.get('pod', {}).get('latent_standardize', False)
            r2r_s = '{:.4f}'.format(r2r) if isinstance(r2r, (int, float)) and r2r == r2r else str(r2r)
            r2s_s = '{:.4f}'.format(r2s) if isinstance(r2s, (int, float)) and r2s == r2s else str(r2s)
            rho_s = '{:.3f}'.format(rho) if isinstance(rho, (int, float)) else str(rho)
            neg_s = '{:.1f}'.format(neg) if isinstance(neg, (int, float)) else str(neg)
            E_s = '{:.1f}%'.format(E) if isinstance(E, (int, float)) else str(E)
            results.append((name, r2r_s, r2s_s, rho_s, neg_s, dim, E_s, lstd))
        except Exception as e:
            results.append((os.path.basename(os.path.dirname(f)), str(e), '', '', '', '', '', ''))

hdr = '{:<55} {:>10} {:>12} {:>8} {:>6} {:>4} {:>8} {:>5}'.format(
    'Experiment', 'R2_roll', 'R2_1step', 'rho', 'neg%', 'd', 'E', 'std')
print(hdr)
print('-' * 110)
for r in results:
    print('{:<55} {:>10} {:>12} {:>8} {:>6} {:>4} {:>8} {:>5}'.format(
        r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]))
