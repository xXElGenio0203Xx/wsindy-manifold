"""Domain utilities for rectangular collective motion simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np

ArrayLike = np.ndarray


@dataclass
class CellList:
    """Linked-cell data structure for neighbor searches."""

    cells: Dict[Tuple[int, int], List[int]]
    cell_size: float
    ncellx: int
    ncelly: int


def apply_bc(x: ArrayLike, Lx: float, Ly: float, bc: str) -> Tuple[ArrayLike, ArrayLike]:
    """Apply boundary conditions to particle positions.

    Parameters
    ----------
    x:
        Positions with shape ``(N, 2)``.
    Lx, Ly:
        Domain side lengths.
    bc:
        ``"periodic"`` or ``"reflecting"``.

    Returns
    -------
    tuple of ndarray
        The adjusted positions and a mask of shape ``(N, 2)`` indicating
        velocity components that should be flipped (``True``) due to
        reflections. For periodic boundaries the mask is all ``False``.
    """

    if bc == "periodic":
        x[:, 0] = np.mod(x[:, 0], Lx)
        x[:, 1] = np.mod(x[:, 1], Ly)
        return x, np.zeros_like(x, dtype=bool)

    if bc == "reflecting":
        flips = np.zeros_like(x, dtype=bool)
        for dim, L in enumerate((Lx, Ly)):
            below = x[:, dim] < 0
            above = x[:, dim] > L
            if np.any(below):
                x[below, dim] = -x[below, dim]
                flips[below, dim] = True
            if np.any(above):
                x[above, dim] = 2 * L - x[above, dim]
                flips[above, dim] = True
            # Handle particles that might still be outside due to multiple crossings.
            x[:, dim] = np.clip(x[:, dim], 0.0, L)
        return x, flips

    raise ValueError(f"Unknown boundary condition '{bc}'")


def pair_displacements(
    x: ArrayLike,
    Lx: float,
    Ly: float,
    bc: str,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Compute pairwise displacements and unit vectors.

    Returns
    -------
    dx, dy, rij, unit_vectors
    """

    diff = x[None, :, :] - x[:, None, :]
    dx = diff[:, :, 0]
    dy = diff[:, :, 1]

    if bc == "periodic":
        dx -= Lx * np.round(dx / Lx)
        dy -= Ly * np.round(dy / Ly)
    elif bc != "reflecting":
        raise ValueError(f"Unknown boundary condition '{bc}'")

    rij = np.sqrt(dx**2 + dy**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ux = np.divide(dx, rij, out=np.zeros_like(dx), where=rij > 0)
        uy = np.divide(dy, rij, out=np.zeros_like(dy), where=rij > 0)
    unit_vectors = np.stack((ux, uy), axis=-1)
    return dx, dy, rij, unit_vectors


def build_cells(
    x: ArrayLike,
    Lx: float,
    Ly: float,
    rcut: float,
    bc: str,
) -> CellList:
    """Construct a linked-cell list for neighbor searches."""

    ncellx = max(1, int(np.ceil(Lx / rcut)))
    ncelly = max(1, int(np.ceil(Ly / rcut)))
    cell_size_x = Lx / ncellx
    cell_size_y = Ly / ncelly
    cell_size = float(max(cell_size_x, cell_size_y))

    cells: Dict[Tuple[int, int], List[int]] = {}

    if bc not in {"periodic", "reflecting"}:
        raise ValueError(f"Unknown boundary condition '{bc}'")

    for idx, (xi, yi) in enumerate(x):
        ix = int(np.floor(xi / cell_size_x))
        iy = int(np.floor(yi / cell_size_y))
        ix = np.clip(ix, 0, ncellx - 1)
        iy = np.clip(iy, 0, ncelly - 1)
        cells.setdefault((ix, iy), []).append(idx)

    return CellList(cells=cells, cell_size=cell_size, ncellx=ncellx, ncelly=ncelly)


def iter_neighbors(
    x: ArrayLike,
    cell_list: CellList,
    Lx: float,
    Ly: float,
    rcut: float,
    bc: str,
) -> Iterator[Tuple[int, int, float, float, float]]:
    """Iterate over particle pairs within the cutoff distance."""

    cells = cell_list.cells
    ncellx = cell_list.ncellx
    ncelly = cell_list.ncelly

    for (ix, iy), indices in cells.items():
        for i_local, i in enumerate(indices):
            for dx_cell in (-1, 0, 1):
                nx = ix + dx_cell
                if bc == "periodic":
                    nx %= ncellx
                elif nx < 0 or nx >= ncellx:
                    continue
                for dy_cell in (-1, 0, 1):
                    ny = iy + dy_cell
                    if bc == "periodic":
                        ny %= ncelly
                    elif ny < 0 or ny >= ncelly:
                        continue
                    neighbor_indices = cells.get((nx, ny), [])
                    for j in neighbor_indices:
                        if j <= i:
                            continue
                        dx = x[j, 0] - x[i, 0]
                        dy = x[j, 1] - x[i, 1]
                        if bc == "periodic":
                            dx -= Lx * np.round(dx / Lx)
                            dy -= Ly * np.round(dy / Ly)
                        rij = np.hypot(dx, dy)
                        if 0 < rij <= rcut:
                            yield i, j, dx, dy, rij

