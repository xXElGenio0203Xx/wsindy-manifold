# WSINDy on Manifolds — Thesis Repo

**Weak-Form Sparse Cell-to-Cell Interaction PDE Discovery from Noisy Trajectories on Curved Manifolds**

This repo contains a modular, reproducible codebase to (1) discover interaction PDEs on curved Riemannian manifolds via a weak-form sparse regression (WSINDy), and (2) benchmark against an Equation‑Free Reduced‑Order Model (EF‑ROM).

## Features
- Sobolev/Galerkin weak formulation with compactly supported test functions
- Geodesic KDE for density & gradient estimation on manifolds
- Sparse regression with MSTLS (multi-step thresholded least squares)
- EF‑ROM baseline via POD/SVD + MVAR or LSTM
- Diagnostics: predictive error, Koopman/transfer spectra, persistent homology, UQ

## Quickstart
```bash
# clone your GitHub repo after creation, then cd into it and:
conda env create -f env.yml  # or: python -m venv .venv && source .venv/bin/activate && pip install -e .[dev]
conda activate wsindy-mfld
pre-commit install  # optional: enables ruff/black on commit
pytest -q           # run tests
```

## Layout
```
src/wsindy_manifold/         # core Python package
  geometry/                  # charts, differential ops, quadrature on M
  density/                   # geodesic KDE, restriction (traj -> density)
  wsindy/                    # weak system assembly + MSTLS
  efrom/                     # EF-ROM: POD/SVD, MVAR, LSTM
  diagnostics/               # metrics, Koopman, topology, UQ
notebooks/                   # experiments & demos
experiments/                 # YAML configs for sweeps
tests/                       # unit tests (pytest)
.github/workflows/ci.yml     # auto CI (lint+tests)
```

## Citation
Please cite this repository if it helps your research. See `CITATION.cff`.

## License
MIT — see `LICENSE`.
