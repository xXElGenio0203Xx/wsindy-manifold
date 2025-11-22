# Rectangular Collective Motion Simulator

Simulation toolkit for two-dimensional self-propelled agent swarms on a rectangular
periodic or reflecting domain. The implementation follows the D'Orsogna model with a
Morse attraction--repulsion potential, using parameter sweeps similar to Bhaskar &
Ziegelmeier, *Chaos* **29**, 123125 (2019).

## Model

For agents with positions $\mathbf{x}_i \in \mathbb{R}^2$ and velocities
$\mathbf{v}_i \in \mathbb{R}^2$, the equations of motion read

\begin{align}
\dot{\mathbf{x}}_i &= \mathbf{v}_i, \\
\dot{\mathbf{v}}_i &= (\alpha - \beta \lVert \mathbf{v}_i \rVert^2)\mathbf{v}_i
 - \sum_{j\neq i} f(r_{ij}) \hat{\mathbf{r}}_{ij}, \\
f(r) &= \frac{C_r}{\ell_r} e^{-r/\ell_r} - \frac{C_a}{\ell_a} e^{-r/\ell_a},
\end{align}

where $r_{ij} = \lVert \mathbf{x}_i - \mathbf{x}_j \rVert$ under the chosen boundary
conditions and $\hat{\mathbf{r}}_{ij} = (\mathbf{x}_j - \mathbf{x}_i)/r_{ij}$.
The interaction radius is truncated at $r_\mathrm{cut} = 3\max(\ell_r, \ell_a)$
and neighbour lists accelerate force evaluation.

Default parameters ($N=200$, $\alpha=1.5$, $\beta=0.5$, $L_x=L_y=20$) recover the
collective behaviours described in the literature. Optionally a Vicsek-style alignment
torque may be enabled.

### Discrete Vicsek Model

The package also includes an implementation of the discrete-time Vicsek model, where
particles move at constant speed and update their headings based on local alignment:

\begin{align}
\mathbf{x}_i(t + \Delta t) &= \mathbf{x}_i(t) + v_0 \mathbf{p}_i(t) \Delta t, \\
\theta_i(t + \Delta t) &= \text{Arg}\left[\sum_{j \in \mathcal{N}_i} e^{i\theta_j(t)}\right] + \eta_i(t),
\end{align}

where $\mathbf{p}_i = (\cos\theta_i, \sin\theta_i)$ is the unit heading vector,
$\mathcal{N}_i$ includes neighbors within radius $R$, and $\eta_i$ is either:
- Gaussian noise with standard deviation $\sigma$ (angular diffusion)
- Uniform noise in $[-\eta/2, \eta/2]$ (as in the original Vicsek paper)

Run a Vicsek simulation with:
```bash
python -m rectsim.cli single --config examples/configs/vicsek_gaussian.yaml
```

This produces similar outputs plus an order parameter plot showing the evolution of
the polarization $\psi(t) = \|\langle \mathbf{p}_i \rangle\|$.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quickstart

Run a single simulation using the provided configuration:

```bash
python -m rectsim.cli single --config examples/configs/single.yaml
```
- `density_anim.mp4`: Gaussian-smoothed density movie (if enabled).
- `density.npz`: gridded densities with coordinates and times.
- `run.json`: configuration, seed and Git commit hash.

To run a parameter sweep and collect a manifest:

```bash
python -m rectsim.cli grid --config examples/configs/grid.yaml
```

The grid runner creates per-combination output folders, a `manifest.csv` summary and
heatmaps of average order parameters versus $(C_r, \ell_r)$.

## KDE Heatmap Generation

Generate Gaussian kernel density estimation (KDE) heatmaps from particle trajectories:

```bash
# Quick demo with synthetic data
python examples/generate_kde_heatmaps.py --demo

# From simulation trajectories
python examples/generate_kde_heatmaps.py \
    --input simulation/trajectories.npz \
    --output kde_results/ \
    --Lx 20 --Ly 20 --nx 128 --ny 128 \
    --hx 0.6 --hy 0.6 --cmap magma
```

This generates:
- `kde_density.npz`: Density data with grid metadata
- `kde_snapshots_magma.png`: Snapshot grid visualization
- `kde_animation_magma.gif`: Animated density evolution

See [`docs/KDE_HEATMAP_GUIDE.md`](docs/KDE_HEATMAP_GUIDE.md) for detailed usage and API reference.

## Latent EF-ROM (rect 2D)

An equation-free reduced-order model pipeline is available under `wsindy_manifold.latent`.
It transforms raw agent trajectories into density forecasts via KDE movies, POD
restriction and a multivariate VAR model. Minimal dependencies (`numpy`, `scipy`,
`matplotlib`) keep scripts CLI-friendly and reproducible.

For the complete ROM/MVAR workflow:
1. Generate simulation trajectories
2. Convert to KDE heatmaps using `examples/generate_kde_heatmaps.py`
3. Train ROM/MVAR models using `scripts/rom_mvar_train.py`
4. Evaluate and visualize with `scripts/rom_mvar_eval.py` and `scripts/rom_mvar_visualize.py`

See ROM guides for details: [`ROM_MVAR_GUIDE.md`](ROM_MVAR_GUIDE.md), [`ROM_MVAR_IMPLEMENTATION.md`](ROM_MVAR_IMPLEMENTATION.md)

Use `scripts/compare_heatmaps.py` to produce side-by-side videos for arbitrary runs,
and `scripts/plot_latent.py` for quick density/latent visualisations.

## Reproducibility

All random number generation uses a configurable seed (default `0`). The run metadata
includes the full configuration and the current Git commit hash to facilitate later
analysis or WSINDy/EF-ROM system identification.

## License

MIT License. See `LICENSE` for details.

---

Rendering LaTeX in this README
--------------------------------

GitHub's Markdown renderer doesn't natively render LaTeX math in `README.md`. This
project includes a tiny renderer that converts LaTeX math blocks into SVG images and
produces `README.rendered.md` which references those SVGs. The script lives at
`scripts/render_readme_math.js` and is run automatically by a GitHub Action on pushes
to `main`.

Usage locally:

1. Install Node.js (>=18).
2. From the project root run:

```bash
cd scripts
npm install mathjax-node
node render_readme_math.js
```

This writes `README.rendered.md` and the generated SVGs under `assets/readme/`.
Open `README.rendered.md` in GitHub or a Markdown viewer to see equations rendered as
images.


Next steps : Read Kefriakedis paper with  the orsogna model,

Next Steps: Use model for the vicsek Model, use the simplest one from the oens professor is sharing with me. Try to do the vicsek as an euler step instead of integrating it, it should be a discrete system where I update the new directions every single time step, use the SFM code from professor Constantino and add the merge with repository of simple euler dfiscrete time Vicsek Alignment. Think about vicsek model and the papers and chooose the simplest one, see that everything works.  



