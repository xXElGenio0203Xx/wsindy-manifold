# Rectangular Collective Motion Simulator

Simulation toolkit for two-dimensional self-propelled agent swarms on a rectangular
periodic or reflecting domain. The implementation follows the D'Orsogna model with a
Morse attraction--repulsion potential, using parameter sweeps similar to Bhaskar &
Ziegelmeier, *Chaos* **29**, 123125 (2019).

## Model

For agents with positions ![](assets/readme/eq-f1ecc768.svg) and velocities
![](assets/readme/eq-40fe241d.svg), the equations of motion read

\begin{align}
\dot{\mathbf{x}}_i &= \mathbf{v}_i, \\
\dot{\mathbf{v}}_i &= (\alpha - \beta \lVert \mathbf{v}_i \rVert^2)\mathbf{v}_i
 - \sum_{j\neq i} f(r_{ij}) \hat{\mathbf{r}}_{ij}, \\
f(r) &= \frac{C_r}{\ell_r} e^{-r/\ell_r} - \frac{C_a}{\ell_a} e^{-r/\ell_a},
\end{align}

where ![](assets/readme/eq-e6f75d21.svg) under the chosen boundary
conditions and ![](assets/readme/eq-53c724eb.svg).
The interaction radius is truncated at ![](assets/readme/eq-4bf598db.svg)
and neighbour lists accelerate force evaluation.

Default parameters (![](assets/readme/eq-38df127a.svg), ![](assets/readme/eq-44550782.svg), ![](assets/readme/eq-ebe55909.svg), ![](assets/readme/eq-d806d56e.svg)) recover the
collective behaviours described in the literature. Optionally a Vicsek-style alignment
torque may be enabled.

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

Outputs are written to the directory specified in the configuration and include:

- `traj.npz`: trajectory, velocity and metadata arrays.
- `metrics.csv`: polarization, angular momentum and distance statistics.
- `traj_final.png`: final positions with velocity quivers.
- `order_params.png`: time series of order parameters.
- `density_anim.mp4`: Gaussian-smoothed density movie (if enabled).
- `density.npz`: gridded densities with coordinates and times.
- `run.json`: configuration, seed and Git commit hash.

To run a parameter sweep and collect a manifest:

```bash
python -m rectsim.cli grid --config examples/configs/grid.yaml
```

The grid runner creates per-combination output folders, a `manifest.csv` summary and
heatmaps of average order parameters versus ![](assets/readme/eq-3a44459a.svg).

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
