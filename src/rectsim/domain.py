"""Domain utilities for rectangular collective motion simulations.

This module contains helpers for imposing boundary conditions, computing
pairwise displacements, and building/iterating linked-cell neighbour lists.

Functions here are deliberately small and documented because they are
performance-sensitive and reused by both the force evaluator and the
alignment routines.
"""

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
    """Compute pairwise displacements, distances and unit vectors.

    This constructs full NxN arrays of pairwise differences with the
    specified boundary conditions applied (periodic or reflecting). The
    returned arrays are useful for small-to-moderate N or testing. For
    larger N, callers should prefer the linked-cell iterator
    :func:`iter_neighbors` which avoids building NxN temporary arrays.

    Returns
    -------
    dx : ndarray (N, N)
        x_j - x_i for each pair (i, j) with boundary conditions applied.
    dy : ndarray (N, N)
        y_j - y_i for each pair (i, j).
    rij : ndarray (N, N)
        Euclidean distances for each pair.
    unit_vectors : ndarray (N, N, 2)
        Unit vectors pointing from particle i to j. Zeros on the diagonal.
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
    """Construct a linked-cell list for neighbour searches.

    The returned :class:`CellList` organizes particle indices into spatial
    cells of size roughly `rcut`. The linked-cell structure allows
    neighbour queries with complexity close to O(N) for uniform point
    distributions and is used by pairwise force routines.
    """

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
    """Yield particle pairs (i, j) whose separation is <= rcut.

    The iterator yields tuples (i, j, dx, dy, rij) where (i < j) and
    0 < rij <= rcut. It accounts for the specified boundary condition.
    Consumers should use this iterator when computing pairwise forces or
    building neighbour lists for other interactions.
    """

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


def neighbor_indices_from_celllist(
    x: ArrayLike,
    cell_list: CellList,
    Lx: float,
    Ly: float,
    rcut: float,
    bc: str,
) -> list:
    """Return neighbour index arrays for each particle using a CellList.

    This is a small helper that builds the per-particle neighbour lists by
    iterating over pairs from :func:`iter_neighbors` and collecting indices.
    It mirrors the format expected by alignment routines.
    """

    neighbours = [[] for _ in range(x.shape[0])]
    for i, j, dx, dy, rij in iter_neighbors(x, cell_list, Lx, Ly, rcut, bc):
        neighbours[i].append(j)
        neighbours[j].append(i)
    return [np.array(lst, dtype=int) for lst in neighbours]


class NeighborFinder:
    """Unified API for neighbor searches using linked-cell lists.
    
    This class wraps the existing linked-cell infrastructure to provide
    a consistent interface across all model backends. It automatically
    includes self in the neighbor list for each particle.
    
    Parameters
    ----------
    Lx, Ly : float
        Domain dimensions.
    rcut : float
        Interaction radius for neighbor searches.
    bc : {"periodic", "reflecting"}
        Boundary condition type.
        
    Attributes
    ----------
    Lx, Ly : float
        Domain dimensions.
    rcut : float
        Cutoff radius.
    bc : str
        Boundary condition.
    _cell_list : CellList or None
        Current cell list structure (None until rebuild() is called).
    _x : ndarray or None
        Cached particle positions from last rebuild.
        
    Examples
    --------
    >>> nf = NeighborFinder(Lx=10.0, Ly=10.0, rcut=1.0, bc="periodic")
    >>> x = np.random.uniform(0, 10, size=(100, 2))
    >>> nf.rebuild(x)
    >>> neighbors = nf.neighbors_of(x)
    >>> # neighbors[i] contains indices of all particles within rcut of i (including i)
    """
    
    def __init__(self, Lx: float, Ly: float, rcut: float, bc: str):
        self.Lx = Lx
        self.Ly = Ly
        self.rcut = rcut
        self.bc = bc
        self._cell_list: CellList | None = None
        self._x: ArrayLike | None = None
        
    def rebuild(self, x: ArrayLike) -> None:
        """Rebuild the cell list for new particle positions.
        
        Parameters
        ----------
        x : ndarray, shape (N, 2)
            Current particle positions.
        """
        self._cell_list = build_cells(x, self.Lx, self.Ly, self.rcut, self.bc)
        self._x = x.copy()
        
    def neighbors_of(self, x: ArrayLike) -> List[np.ndarray]:
        """Return neighbor indices for each particle.
        
        Parameters
        ----------
        x : ndarray, shape (N, 2)
            Particle positions (should match positions from last rebuild).
            
        Returns
        -------
        neighbors : list of ndarray
            List where neighbors[i] is a 1D array of neighbor indices for
            particle i within the cutoff radius. Each particle is included
            in its own neighbor list.
            
        Notes
        -----
        The cell list must be built/rebuilt via rebuild() before calling this.
        If positions have changed significantly since rebuild(), results may
        be inaccurate.
        """
        if self._cell_list is None:
            raise RuntimeError("Must call rebuild() before neighbors_of()")
            
        # Get neighbors from cell list (excludes self)
        neighbors = neighbor_indices_from_celllist(
            x, self._cell_list, self.Lx, self.Ly, self.rcut, self.bc
        )
        
        # Add self to each neighbor list
        for i in range(len(neighbors)):
            neighbors[i] = np.append(neighbors[i], i)
            
        return neighbors

