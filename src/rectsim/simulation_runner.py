"""
Simulation Runner Module
=========================

Parallel simulation execution for training and test runs.
"""

import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import json
import pandas as pd

from rectsim.vicsek_discrete import simulate_backend
from rectsim.legacy_functions import kde_density_movie


def simulate_single_run(args_tuple):
    """
    Worker function for parallel simulation execution.
    
    Parameters
    ----------
    args_tuple : tuple
        (cfg, BASE_CONFIG, OUTPUT_DIR, DENSITY_NX, DENSITY_NY, DENSITY_BANDWIDTH, is_test)
    
    Returns
    -------
    dict
        Metadata for the simulation run
    """
    (cfg, BASE_CONFIG, OUTPUT_DIR, DENSITY_NX, DENSITY_NY, DENSITY_BANDWIDTH, is_test) = args_tuple
    
    run_id = cfg['run_id']
    distribution = cfg['distribution']
    ic_params = cfg['ic_params']
    label = cfg['label']
    
    # Determine output directory
    if is_test:
        run_name = f"test_{run_id:03d}"
        run_dir = OUTPUT_DIR / "test" / run_name
    else:
        run_name = f"train_{run_id:03d}"
        run_dir = OUTPUT_DIR / "train" / run_name
    
    run_dir.mkdir(exist_ok=True, parents=True)
    
    # Set up configuration (keep nested structure for simulate_backend)
    config = BASE_CONFIG.copy()
    config["seed"] = run_id + 1000 * (1 if is_test else 0)
    config["initial_distribution"] = distribution
    config["ic_params"] = ic_params
    
    # Run simulation
    rng = np.random.default_rng(config["seed"])
    result = simulate_backend(config, rng)
    
    # Extract trajectories
    times = result["times"]
    traj = result["traj"]
    vel = result["vel"]
    
    # Save trajectories (REQUIRED for visualization pipeline)
    np.savez_compressed(
        run_dir / "trajectory.npz",
        traj=traj,
        vel=vel,
        times=times
    )
    
    # Compute density movies
    rho, meta = kde_density_movie(
        traj,
        Lx=config["sim"]["Lx"],
        Ly=config["sim"]["Ly"],
        nx=DENSITY_NX,
        ny=DENSITY_NY,
        bandwidth=DENSITY_BANDWIDTH,
        bc=config["sim"].get("bc", "periodic")
    )
    
    # Create spatial grids
    xgrid = np.linspace(0, config["sim"]["Lx"], DENSITY_NX, endpoint=False) + config["sim"]["Lx"]/(2*DENSITY_NX)
    ygrid = np.linspace(0, config["sim"]["Ly"], DENSITY_NY, endpoint=False) + config["sim"]["Ly"]/(2*DENSITY_NY)
    
    # Save density data (with xgrid, ygrid for compatibility)
    # Use different filename for test runs (visualization expects this)
    density_filename = "density_true.npz" if is_test else "density.npz"
    np.savez_compressed(
        run_dir / density_filename,
        rho=rho,
        xgrid=xgrid,
        ygrid=ygrid,
        times=times
    )
    
    # Create metadata
    metadata = {
        "run_id": run_id,
        "run_name": run_name,
        "label": label,
        "distribution": distribution,
        "ic_params": ic_params,
        "seed": config["seed"],
        "T": len(times)
    }
    
    return metadata


def run_simulations_parallel(configs, base_config, output_dir, density_nx, density_ny, 
                            density_bandwidth, is_test=False, n_workers=None):
    """
    Run simulations in parallel.
    
    Parameters
    ----------
    configs : list
        List of configuration dictionaries from generate_training_configs or generate_test_configs
    base_config : dict
        Base simulation configuration
    output_dir : Path
        Root output directory
    density_nx, density_ny : int
        Density grid resolution
    density_bandwidth : float
        KDE bandwidth
    is_test : bool
        Whether these are test runs
    n_workers : int, optional
        Number of parallel workers (default: min(cpu_count(), 16))
    
    Returns
    -------
    tuple
        (metadata_list, elapsed_time)
    """
    import time
    
    if n_workers is None:
        n_workers = min(cpu_count(), 16)
    
    # Prepare arguments
    args = [(cfg, base_config, output_dir, density_nx, density_ny, density_bandwidth, is_test)
            for cfg in configs]
    
    start_time = time.time()
    
    # Run in parallel
    with Pool(n_workers) as pool:
        metadata = list(tqdm(
            pool.imap(simulate_single_run, args),
            total=len(configs),
            desc="Test sims" if is_test else "Training sims"
        ))
    
    elapsed_time = time.time() - start_time
    
    # Save metadata
    subdir = "test" if is_test else "train"
    metadata_dir = output_dir / subdir
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    df = pd.DataFrame(metadata)
    df.to_csv(metadata_dir / "index_mapping.csv", index=False)
    
    return metadata, elapsed_time
