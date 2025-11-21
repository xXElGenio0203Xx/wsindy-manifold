"""Data loader for unseen test simulations for ROM/MVAR evaluation.

This module provides utilities to load simulation outputs (density movies,
trajectories, metadata) organized by initial condition type for evaluating
trained ROM/MVAR models.

Author: Maria
Date: November 2025
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np


@dataclass
class SimulationSample:
    """Container for a single test simulation.
    
    Attributes
    ----------
    ic_type : str
        Initial condition type (e.g., "ring", "gaussian", "uniform", "cluster2").
    name : str
        Simulation identifier (e.g., "sim_000", "run_0").
    density_true : np.ndarray
        Ground truth density movie, shape (T, Ny, Nx).
    traj_true : Optional[dict]
        Ground truth trajectories if available. Expected keys:
        - "x": positions, shape (T, N, 2)
        - "v": velocities, shape (T, N, 2)  
        - "times": time points, shape (T,)
    meta : dict
        Metadata from run.json or NPZ arrays (dt, N, Lx, Ly, etc.).
    path : Path
        Path to the simulation directory.
    """
    
    ic_type: str
    name: str
    density_true: np.ndarray
    traj_true: Optional[Dict[str, np.ndarray]]
    meta: Dict[str, Any]
    path: Path
    
    @property
    def T(self) -> int:
        """Number of time steps."""
        return self.density_true.shape[0]
    
    @property
    def grid_shape(self) -> tuple[int, int]:
        """Spatial grid shape (Ny, Nx)."""
        return self.density_true.shape[1:]
    
    @property
    def dt(self) -> float:
        """Time step (from metadata)."""
        return self.meta.get("dt", 0.01)
    
    @property
    def times(self) -> np.ndarray:
        """Time points array."""
        if self.traj_true is not None and "times" in self.traj_true:
            return self.traj_true["times"]
        # Reconstruct from dt if not available
        return np.arange(self.T) * self.dt


def load_unseen_simulations(
    root: Path,
    ic_types: Optional[List[str]] = None,
    require_density: bool = True,
    require_traj: bool = False,
) -> List[SimulationSample]:
    """Load test simulations organized by initial condition type.
    
    Expected directory structure:
        root/
        ├── ring/
        │   ├── sim_000/
        │   │   ├── density.npz
        │   │   ├── traj.npz (optional)
        │   │   ├── trajectories.npz (alternative name)
        │   │   └── run.json (optional metadata)
        │   └── sim_001/
        │       └── ...
        ├── gaussian/
        │   └── ...
        └── uniform/
            └── ...
    
    Parameters
    ----------
    root : Path
        Root directory containing IC type subdirectories.
    ic_types : Optional[List[str]]
        Specific IC types to load. If None, auto-detect all subdirectories.
    require_density : bool, default=True
        If True, skip simulations without density.npz.
    require_traj : bool, default=False
        If True, skip simulations without trajectory data.
        
    Returns
    -------
    samples : List[SimulationSample]
        Loaded simulation samples, sorted by IC type then name.
        
    Notes
    -----
    - Skips incomplete runs with a warning instead of crashing.
    - Density NPZ should contain "density" array with shape (T, Ny, Nx).
    - Trajectory NPZ can be named "traj.npz" or "trajectories.npz" and
      should contain "x" (positions), "v" (velocities), "times" arrays.
    - run.json is optional and can contain simulation metadata.
    """
    root = Path(root)
    
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")
    
    # Auto-detect IC types if not specified
    if ic_types is None:
        ic_types = sorted([
            d.name for d in root.iterdir() 
            if d.is_dir() and not d.name.startswith(".")
        ])
        if not ic_types:
            warnings.warn(f"No IC type directories found in {root}")
            return []
    
    samples = []
    
    for ic_type in ic_types:
        ic_dir = root / ic_type
        if not ic_dir.exists():
            warnings.warn(f"IC type directory not found: {ic_dir}, skipping")
            continue
        
        # Find all simulation subdirectories
        sim_dirs = sorted([
            d for d in ic_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        
        for sim_dir in sim_dirs:
            try:
                sample = _load_single_simulation(
                    sim_dir,
                    ic_type,
                    require_density=require_density,
                    require_traj=require_traj,
                )
                if sample is not None:
                    samples.append(sample)
            except Exception as e:
                warnings.warn(
                    f"Failed to load {sim_dir}: {e}. Skipping.",
                    stacklevel=2
                )
    
    # Sort by IC type, then name
    samples.sort(key=lambda s: (s.ic_type, s.name))
    
    return samples


def _load_single_simulation(
    sim_dir: Path,
    ic_type: str,
    require_density: bool = True,
    require_traj: bool = False,
) -> Optional[SimulationSample]:
    """Load a single simulation from directory.
    
    Parameters
    ----------
    sim_dir : Path
        Simulation directory containing NPZ files.
    ic_type : str
        IC type label.
    require_density : bool
        Skip if density.npz not found.
    require_traj : bool
        Skip if trajectory data not found.
        
    Returns
    -------
    sample : Optional[SimulationSample]
        Loaded sample, or None if requirements not met.
    """
    name = sim_dir.name
    
    # Load density
    density_path = sim_dir / "density.npz"
    if not density_path.exists():
        if require_density:
            warnings.warn(f"No density.npz in {sim_dir}, skipping")
            return None
        density_true = None
    else:
        density_data = np.load(density_path)
        if "density" in density_data:
            density_true = density_data["density"]
        elif "rho" in density_data:
            density_true = density_data["rho"]
        else:
            warnings.warn(
                f"density.npz in {sim_dir} has no 'density' or 'rho' array, skipping"
            )
            return None
    
    # Load trajectories (optional)
    traj_true = None
    traj_path = sim_dir / "traj.npz"
    if not traj_path.exists():
        traj_path = sim_dir / "trajectories.npz"
    
    if traj_path.exists():
        traj_data = np.load(traj_path)
        traj_true = {}
        if "x" in traj_data:
            traj_true["x"] = traj_data["x"]
        if "v" in traj_data:
            traj_true["v"] = traj_data["v"]
        if "times" in traj_data:
            traj_true["times"] = traj_data["times"]
        
        if not traj_true:
            traj_true = None  # No recognized arrays
    
    if require_traj and traj_true is None:
        warnings.warn(f"No trajectory data in {sim_dir}, skipping")
        return None
    
    # Load metadata
    meta = {}
    run_json_path = sim_dir / "run.json"
    if run_json_path.exists():
        try:
            with open(run_json_path, "r") as f:
                meta = json.load(f)
        except Exception as e:
            warnings.warn(f"Failed to parse {run_json_path}: {e}")
    
    # Extract common metadata from NPZ if not in run.json
    if density_path.exists() and density_data is not None:
        density_data = np.load(density_path)
        # Check for embedded metadata (common pattern in our outputs)
        for key in ["dt", "Nx", "Ny", "Lx", "Ly", "N", "T"]:
            if key in density_data and key not in meta:
                val = density_data[key]
                # Handle scalar arrays
                if hasattr(val, "item"):
                    meta[key] = val.item()
                else:
                    meta[key] = val
    
    # Extract from trajectory NPZ if available
    if traj_path.exists():
        traj_data = np.load(traj_path)
        # Look for JSON-encoded metadata (our standard format)
        for key in ["sim", "params"]:
            if key in traj_data:
                try:
                    val = traj_data[key].item()
                    if isinstance(val, str):
                        meta[key] = json.loads(val)
                    else:
                        meta[key] = val
                except:
                    pass
    
    return SimulationSample(
        ic_type=ic_type,
        name=name,
        density_true=density_true,
        traj_true=traj_true,
        meta=meta,
        path=sim_dir,
    )


def group_by_ic_type(samples: List[SimulationSample]) -> Dict[str, List[SimulationSample]]:
    """Group simulation samples by initial condition type.
    
    Parameters
    ----------
    samples : List[SimulationSample]
        Flat list of samples.
        
    Returns
    -------
    grouped : Dict[str, List[SimulationSample]]
        Dictionary mapping IC type to list of samples.
    """
    grouped = {}
    for sample in samples:
        if sample.ic_type not in grouped:
            grouped[sample.ic_type] = []
        grouped[sample.ic_type].append(sample)
    return grouped


def print_dataset_summary(samples: List[SimulationSample]) -> None:
    """Print a summary of loaded simulation dataset.
    
    Parameters
    ----------
    samples : List[SimulationSample]
        Loaded samples.
    """
    if not samples:
        print("No samples loaded.")
        return
    
    grouped = group_by_ic_type(samples)
    
    print(f"Loaded {len(samples)} simulations across {len(grouped)} IC types:")
    print()
    
    for ic_type, sims in sorted(grouped.items()):
        print(f"  {ic_type}: {len(sims)} simulations")
        if sims:
            example = sims[0]
            T, Ny, Nx = example.density_true.shape
            has_traj = example.traj_true is not None
            print(f"    Grid: ({Ny}, {Nx}), T={T}, trajectories={has_traj}")
    print()
