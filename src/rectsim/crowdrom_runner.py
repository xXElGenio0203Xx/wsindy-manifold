"""
CrowdROM End-to-End Runner API.

Orchestrates complete pipeline:
simulate -> KDE -> POD -> ADF non-stationarity -> CSVs -> order parameters -> JSONs -> movies

Exit codes:
    0: Success
    2: Mass conservation check failed
    3: Invalid configuration
    4: I/O error
"""

import json
import subprocess
import sys
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

from .pod import PODProjector
from .nonstationarity import NonStationarityProcessor
from .crowdrom_movies import create_trajectory_movie, create_density_movie, create_latent_movie
from .crowdrom_schemas import validate_run_json, validate_nonstationarity_report


class CrowdROMRunner:
    """
    End-to-end CrowdROM pipeline runner.
    
    Produces deterministic, versioned, validated outputs:
    - run.json (complete parameters + environment)
    - non_stationarity_report.json (ADF decisions per latent coordinate)
    - Per-simulation CSVs: trajectories, densities, latents, order_parameters
    - Movies (trajectory, density, latent) for selected simulations
    - Mass conservation validation
    """
    
    def __init__(
        self,
        cfg: dict,
        outdir: str,
        *,
        mass_tol: float = 1e-12,
        adf_alpha: float = 0.01,
        adf_max_lags: Optional[int] = None,
        save_dtype: str = "float64",
        movie_fps: int = 20,
        movie_max_frames: int = 500,
        movies_for: Tuple[int, ...] = (1,),
        quiet: bool = False
    ):
        """
        Initialize CrowdROM runner.
        
        Parameters
        ----------
        cfg : dict
            Configuration dictionary matching spec schema
        outdir : str
            Output directory path (created if missing)
        mass_tol : float, default=1e-12
            Tolerance for mass conservation checks
        adf_alpha : float, default=0.01
            ADF significance level
        adf_max_lags : int or None, default=None
            Max lags for ADF (None = auto selection)
        save_dtype : str, default="float64"
            CSV float precision ("float32" or "float64")
        movie_fps : int, default=20
            Frames per second for movies
        movie_max_frames : int, default=500
            Max frames in movies (subsample if needed)
        movies_for : tuple of int, default=(1,)
            1-based simulation indices to generate movies for
        quiet : bool, default=False
            Suppress non-error logs
        """
        self.cfg = cfg
        self.outdir = Path(outdir)
        self.mass_tol = mass_tol
        self.adf_alpha = adf_alpha
        self.adf_max_lags = adf_max_lags
        self.save_dtype = save_dtype
        self.movie_fps = movie_fps
        self.movie_max_frames = movie_max_frames
        self.movies_for = set(movies_for) if movies_for != ("none",) else set()
        self.quiet = quiet
        
        # Runtime state
        self.run_id = None
        self.seed = None
        self.trajectories = []  # List of (times, positions, velocities) per sim
        self.densities = []  # List of density snapshots per sim
        self.latents = []  # List of latent coordinates per sim
        self.order_params = []  # List of order parameter DataFrames per sim
        self.pod_projector = None
        self.nonstationarity_processor = None
        self.logs = []
        
    def log(self, msg: str, level: str = "INFO"):
        """Add timestamped log entry."""
        timestamp = datetime.now(timezone.utc).isoformat()
        entry = f"[{timestamp}] {level}: {msg}"
        self.logs.append(entry)
        if not self.quiet or level == "ERROR":
            print(entry)
    
    def run(self) -> int:
        """
        Execute complete pipeline.
        
        Returns
        -------
        exit_code : int
            0=success, 2=mass check failed, 3=invalid config, 4=I/O error
        """
        try:
            self.log("="*80)
            self.log("CrowdROM Pipeline Starting")
            self.log("="*80)
            
            # 1. Setup
            exit_code = self._setup()
            if exit_code != 0:
                return exit_code
            
            # 2. Simulate or load trajectories
            exit_code = self._simulate_trajectories()
            if exit_code != 0:
                return exit_code
            
            # 3. KDE -> densities
            exit_code = self._compute_densities()
            if exit_code != 0:
                return exit_code
            
            # 4. POD -> latents
            exit_code = self._compute_latents()
            if exit_code != 0:
                return exit_code
            
            # 5. ADF non-stationarity analysis
            exit_code = self._analyze_nonstationarity()
            if exit_code != 0:
                return exit_code
            
            # 6. Compute order parameters
            exit_code = self._compute_order_parameters()
            if exit_code != 0:
                return exit_code
            
            # 7. Save CSVs
            exit_code = self._save_csvs()
            if exit_code != 0:
                return exit_code
            
            # 8. Generate movies
            exit_code = self._generate_movies()
            if exit_code != 0:
                return exit_code
            
            # 9. Write JSONs
            exit_code = self._write_jsons()
            if exit_code != 0:
                return exit_code
            
            # 10. Final validation
            exit_code = self._final_validation()
            if exit_code != 0:
                return exit_code
            
            self.log("="*80)
            self.log("CrowdROM Pipeline Complete")
            self.log(f"Output directory: {self.outdir}")
            self.log("="*80)
            
            return 0
            
        except Exception as e:
            self.log(f"Pipeline failed with exception: {e}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "ERROR")
            return 4
    
    def _setup(self) -> int:
        """Setup: create output directory, validate config, initialize RNG."""
        try:
            self.log("Setting up pipeline...")
            
            # Create output directory
            self.outdir.mkdir(parents=True, exist_ok=True)
            self.log(f"  Output directory: {self.outdir}")
            
            # Extract seed
            self.seed = self.cfg.get("meta", {}).get("seed", 42)
            np.random.seed(self.seed)
            self.log(f"  Random seed: {self.seed}")
            
            # Generate run ID
            self.run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
            self.log(f"  Run ID: {self.run_id}")
            
            # Validate config (basic checks)
            if "simulation" not in self.cfg:
                self.log("Invalid config: missing 'simulation' section", "ERROR")
                return 3
            
            C = self.cfg["simulation"].get("C", 1)
            if C < 1:
                self.log(f"Invalid config: C={C} must be >= 1", "ERROR")
                return 3
            
            self.log(f"  Simulations (C): {C}")
            self.log("Setup complete")
            
            return 0
            
        except Exception as e:
            self.log(f"Setup failed: {e}", "ERROR")
            return 4
    
    def _simulate_trajectories(self) -> int:
        """Run simulations or load existing trajectories."""
        try:
            C = self.cfg["simulation"]["C"]
            self.log(f"Simulating {C} trajectories...")
            
            # TODO: Implement simulation using existing backends
            # For now, create placeholder
            self.log("  NOTE: Simulation implementation pending - creating placeholder")
            
            for c in range(1, C + 1):
                # Placeholder: random walk
                n_particles = self.cfg["simulation"].get("num_particles", 100)
                T = self.cfg["simulation"].get("T", 100)
                dt_obs = self.cfg["simulation"].get("dt_obs", 0.1)
                n_frames = int(T / dt_obs) + 1
                
                times = np.arange(n_frames) * dt_obs
                positions = np.random.randn(n_frames, n_particles, 2).cumsum(axis=0)
                velocities = np.diff(positions, axis=0, prepend=positions[:1])
                
                self.trajectories.append((times, positions, velocities))
                self.log(f"  Simulation {c}/{C}: {n_frames} frames, {n_particles} particles")
            
            return 0
            
        except Exception as e:
            self.log(f"Trajectory simulation failed: {e}", "ERROR")
            return 4
    
    def _compute_densities(self) -> int:
        """Compute KDE densities from trajectories."""
        try:
            self.log("Computing KDE densities...")
            
            # TODO: Implement using existing KDE module
            self.log("  NOTE: KDE implementation pending - creating placeholder")
            
            for c, (times, positions, _) in enumerate(self.trajectories, 1):
                # Placeholder: uniform density on grid
                nx = self.cfg.get("domain_grid", {}).get("nx", 40)
                ny = self.cfg.get("domain_grid", {}).get("ny", 40)
                n_frames = len(times)
                
                density_snapshots = np.random.rand(n_frames, nx, ny)
                # Normalize to mass = 1
                density_snapshots /= density_snapshots.sum(axis=(1, 2), keepdims=True)
                
                self.densities.append(density_snapshots)
                self.log(f"  Simulation {c}: density grid {nx}×{ny}, {n_frames} frames")
            
            return 0
            
        except Exception as e:
            self.log(f"Density computation failed: {e}", "ERROR")
            return 4
    
    def _compute_latents(self) -> int:
        """Apply POD to extract latent coordinates."""
        try:
            self.log("Computing POD latent coordinates...")
            
            # Stack all density snapshots
            all_snapshots = np.vstack(self.densities)
            n_total = len(all_snapshots)
            self.log(f"  Total snapshots: {n_total}")
            
            # Fit POD
            energy_threshold = self.cfg.get("pod", {}).get("energy_threshold", 0.99)
            dx = self.cfg.get("domain_grid", {}).get("dx", 1.0)
            dy = self.cfg.get("domain_grid", {}).get("dy", 1.0)
            
            self.pod_projector = PODProjector(
                energy_threshold=energy_threshold,
                use_weighted_mass=False
            )
            self.pod_projector.fit(all_snapshots, dx, dy)
            
            d = self.pod_projector.d
            self.log(f"  POD dimension: d={d}")
            self.log(f"  Energy captured: {self.pod_projector.energy_curve[d-1]:.4f}")
            
            # Project each simulation
            for c, density_snapshots in enumerate(self.densities, 1):
                Y = self.pod_projector.transform(density_snapshots)  # (d, n_t)
                self.latents.append(Y)
                self.log(f"  Simulation {c}: latent shape {Y.shape}")
            
            return 0
            
        except Exception as e:
            self.log(f"POD computation failed: {e}", "ERROR")
            return 4
    
    def _analyze_nonstationarity(self) -> int:
        """Run ADF tests on latent coordinates."""
        try:
            self.log("Analyzing non-stationarity (ADF tests)...")
            
            self.nonstationarity_processor = NonStationarityProcessor(
                adf_alpha=self.adf_alpha,
                verbose=not self.quiet
            )
            self.nonstationarity_processor.fit(self.latents)
            
            # Count stationary/non-stationary per simulation
            for c, case_meta in enumerate(self.nonstationarity_processor.case_meta, 1):
                stationary = sum(1 for d in case_meta.decisions if d.mode == 'raw')
                total = len(case_meta.decisions)
                self.log(f"  Simulation {c}: {stationary}/{total} stationary")
            
            return 0
            
        except Exception as e:
            self.log(f"Non-stationarity analysis failed: {e}", "ERROR")
            return 4
    
    def _compute_order_parameters(self) -> int:
        """Compute order parameters for each simulation."""
        try:
            self.log("Computing order parameters...")
            
            for c, ((times, positions, velocities), density_snapshots) in enumerate(
                zip(self.trajectories, self.densities), 1
            ):
                n_frames = len(times)
                dx = self.cfg.get("domain_grid", {}).get("dx", 1.0)
                dy = self.cfg.get("domain_grid", {}).get("dy", 1.0)
                dA = dx * dy
                
                order_params = []
                
                for t_idx in range(n_frames):
                    rho = density_snapshots[t_idx]
                    
                    # Mass conservation
                    mass_unweighted = float(rho.sum())
                    mass_weighted = float(rho.sum() * dA)
                    mass_err_unweighted = abs(mass_unweighted - 1.0)
                    mass_err_weighted = abs(mass_weighted - 1.0)
                    
                    # Check mass tolerance
                    if mass_err_weighted > self.mass_tol:
                        self.log(
                            f"Mass conservation violation at sim={c}, t={times[t_idx]:.3f}: "
                            f"error={mass_err_weighted:.2e} > tol={self.mass_tol:.2e}",
                            "ERROR"
                        )
                        return 2
                    
                    # Polarization and mean speed (if velocities available)
                    v = velocities[t_idx]  # (n_particles, 2)
                    v_norms = np.linalg.norm(v, axis=1)
                    polarization = float(np.linalg.norm(v.sum(axis=0)) / (v_norms.sum() + 1e-12))
                    mean_speed = float(v_norms.mean())
                    
                    # Density variance and entropy
                    density_variance = float(rho.var())
                    rho_flat = rho.flatten()
                    rho_safe = rho_flat + 1e-12
                    density_entropy = float(-np.sum(rho_flat * np.log(rho_safe)))
                    
                    order_params.append({
                        "sim_id": c,
                        "time_step": t_idx,
                        "time_sec": float(times[t_idx]),
                        "mass_unweighted": mass_unweighted,
                        "mass_weighted": mass_weighted,
                        "mass_err_unweighted": mass_err_unweighted,
                        "mass_err_weighted": mass_err_weighted,
                        "polarization": polarization,
                        "mean_speed": mean_speed,
                        "density_variance": density_variance,
                        "density_entropy": density_entropy
                    })
                
                df = pd.DataFrame(order_params)
                self.order_params.append(df)
                
                max_err = df["mass_err_weighted"].max()
                self.log(f"  Simulation {c}: max mass error = {max_err:.2e}")
            
            self.log("Order parameters computed successfully")
            return 0
            
        except Exception as e:
            self.log(f"Order parameter computation failed: {e}", "ERROR")
            return 4
    
    def _save_csvs(self) -> int:
        """Save all CSVs (trajectories, densities, latents, order_parameters)."""
        try:
            self.log("Saving CSV files...")
            
            C = len(self.trajectories)
            
            for c in range(1, C + 1):
                sim_dir = self.outdir / f"sim_{c:04d}"
                sim_dir.mkdir(exist_ok=True)
                
                # 1. trajectories.csv
                times, positions, velocities = self.trajectories[c - 1]
                traj_data = []
                for t_idx, t in enumerate(times):
                    for agent_id in range(positions.shape[1]):
                        traj_data.append({
                            "sim_id": c,
                            "time_step": t_idx,
                            "time_sec": float(t),
                            "agent_id": agent_id,
                            "x": float(positions[t_idx, agent_id, 0]),
                            "y": float(positions[t_idx, agent_id, 1]),
                            "vx": float(velocities[t_idx, agent_id, 0]),
                            "vy": float(velocities[t_idx, agent_id, 1])
                        })
                
                df_traj = pd.DataFrame(traj_data)
                traj_path = sim_dir / "trajectories.csv"
                df_traj.to_csv(traj_path, index=False, float_format=f"%.{6 if self.save_dtype == 'float32' else 12}g")
                
                # 2. densities.csv
                density_snapshots = self.densities[c - 1]
                n_frames, nx, ny = density_snapshots.shape
                xmin = self.cfg.get("domain_grid", {}).get("domain", {}).get("xmin", 0.0)
                xmax = self.cfg.get("domain_grid", {}).get("domain", {}).get("xmax", float(nx))
                ymin = self.cfg.get("domain_grid", {}).get("domain", {}).get("ymin", 0.0)
                ymax = self.cfg.get("domain_grid", {}).get("domain", {}).get("ymax", float(ny))
                
                x_cents = np.linspace(xmin, xmax, nx, endpoint=False) + (xmax - xmin) / (2 * nx)
                y_cents = np.linspace(ymin, ymax, ny, endpoint=False) + (ymax - ymin) / (2 * ny)
                
                density_data = []
                for t_idx in range(n_frames):
                    for i in range(nx):
                        for j in range(ny):
                            density_data.append({
                                "sim_id": c,
                                "time_step": t_idx,
                                "time_sec": float(times[t_idx]),
                                "i": i,
                                "j": j,
                                "x_centroid": float(x_cents[i]),
                                "y_centroid": float(y_cents[j]),
                                "rho": float(density_snapshots[t_idx, i, j])
                            })
                
                df_dens = pd.DataFrame(density_data)
                dens_path = sim_dir / "densities.csv"
                df_dens.to_csv(dens_path, index=False, float_format=f"%.{6 if self.save_dtype == 'float32' else 12}g")
                
                # 3. latents.csv
                Y = self.latents[c - 1]  # (d, n_t)
                d, n_t = Y.shape
                latent_data = []
                for t_idx in range(n_t):
                    row = {
                        "sim_id": c,
                        "time_step": t_idx,
                        "time_sec": float(times[t_idx])
                    }
                    for dim in range(d):
                        row[f"y{dim+1}"] = float(Y[dim, t_idx])
                    latent_data.append(row)
                
                df_latent = pd.DataFrame(latent_data)
                latent_path = sim_dir / "latents.csv"
                df_latent.to_csv(latent_path, index=False, float_format=f"%.{6 if self.save_dtype == 'float32' else 12}g")
                
                # 4. order_parameters.csv
                df_order = self.order_params[c - 1]
                order_path = sim_dir / "order_parameters.csv"
                df_order.to_csv(order_path, index=False, float_format=f"%.{6 if self.save_dtype == 'float32' else 12}g")
                
                self.log(f"  Saved CSVs for simulation {c}")
            
            return 0
            
        except Exception as e:
            self.log(f"CSV saving failed: {e}", "ERROR")
            return 4
    
    def _generate_movies(self) -> int:
        """Generate movies for selected simulations."""
        try:
            if not self.movies_for:
                self.log("Skipping movie generation (movies_for=none)")
                return 0
            
            self.log(f"Generating movies for simulations: {sorted(self.movies_for)}")
            
            domain = self.cfg.get("domain_grid", {}).get("domain", {})
            obstacles = self.cfg.get("domain_grid", {}).get("obstacles", [])
            
            for c in sorted(self.movies_for):
                if c < 1 or c > len(self.trajectories):
                    self.log(f"  Skipping sim {c}: out of range (1-{len(self.trajectories)})", "WARN")
                    continue
                
                sim_dir = self.outdir / f"sim_{c:04d}"
                
                times, positions, velocities = self.trajectories[c - 1]
                density_snapshots = self.densities[c - 1]
                latents = self.latents[c - 1]
                
                self.log(f"  Generating movies for simulation {c}...")
                
                # 1. Trajectory movie
                try:
                    traj_movie_path = sim_dir / "movie_trajectory.mp4"
                    create_trajectory_movie(
                        times=times,
                        positions=positions,
                        velocities=velocities,
                        domain=domain,
                        output_path=traj_movie_path,
                        fps=self.movie_fps,
                        max_frames=self.movie_max_frames,
                        obstacles=obstacles
                    )
                    self.log(f"    ✓ {traj_movie_path.name}")
                except Exception as e:
                    self.log(f"    ✗ Trajectory movie failed: {e}", "WARN")
                
                # 2. Density movie
                try:
                    dens_movie_path = sim_dir / "movie_density.mp4"
                    create_density_movie(
                        times=times,
                        densities=density_snapshots,
                        domain=domain,
                        output_path=dens_movie_path,
                        fps=self.movie_fps,
                        max_frames=self.movie_max_frames,
                        show_mass=True
                    )
                    self.log(f"    ✓ {dens_movie_path.name}")
                except Exception as e:
                    self.log(f"    ✗ Density movie failed: {e}", "WARN")
                
                # 3. Latent movie
                try:
                    latent_movie_path = sim_dir / "movie_latent.mp4"
                    create_latent_movie(
                        times=times,
                        latents=latents,
                        output_path=latent_movie_path,
                        fps=self.movie_fps,
                        max_frames=self.movie_max_frames,
                        mode="timeseries"  # or "embedding"
                    )
                    self.log(f"    ✓ {latent_movie_path.name}")
                except Exception as e:
                    self.log(f"    ✗ Latent movie failed: {e}", "WARN")
            
            return 0
            
        except Exception as e:
            self.log(f"Movie generation failed: {e}", "ERROR")
            return 4
    
    def _write_jsons(self) -> int:
        """Write run.json and non_stationarity_report.json."""
        try:
            self.log("Writing JSON artifacts...")
            
            # 1. run.json
            run_json = self._build_run_json()
            
            # Validate run.json
            valid, msg = validate_run_json(run_json)
            if not valid:
                self.log(f"run.json validation failed: {msg}", "ERROR")
                return 3
            if msg:  # Warning message
                self.log(f"  {msg}", "WARN")
            
            run_path = self.outdir / "run.json"
            with open(run_path, 'w') as f:
                json.dump(run_json, f, indent=2)
            self.log(f"  Saved {run_path.name}")
            
            # 2. non_stationarity_report.json
            nonstationarity_json = self._build_nonstationarity_json()
            
            # Validate non_stationarity_report.json
            valid, msg = validate_nonstationarity_report(nonstationarity_json)
            if not valid:
                self.log(f"non_stationarity_report.json validation failed: {msg}", "ERROR")
                return 3
            if msg:  # Warning message
                self.log(f"  {msg}", "WARN")
            
            nonstat_path = self.outdir / "non_stationarity_report.json"
            with open(nonstat_path, 'w') as f:
                json.dump(nonstationarity_json, f, indent=2)
            self.log(f"  Saved {nonstat_path.name}")
            
            # 3. logs.txt
            logs_path = self.outdir / "logs.txt"
            with open(logs_path, 'w') as f:
                f.write('\n'.join(self.logs))
            self.log(f"  Saved {logs_path.name}")
            
            return 0
            
        except Exception as e:
            self.log(f"JSON writing failed: {e}", "ERROR")
            return 4
    
    def _build_run_json(self) -> dict:
        """Build complete run.json matching spec schema."""
        # Get git commit
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).parent,
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except:
            git_commit = "unknown"
        
        # Get package versions
        import numpy as np
        try:
            import scipy
            scipy_version = scipy.__version__
        except:
            scipy_version = "not installed"
        
        run_json = {
            "meta": {
                "run_id": self.run_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "seed": int(self.seed),
                "code_version": {
                    "git_commit": git_commit,
                    "repo": "wsindy-manifold"
                },
                "env": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "scipy": scipy_version,
                    "platform": platform.platform()
                }
            },
            "simulation": self.cfg.get("simulation", {}),
            "domain_grid": self.cfg.get("domain_grid", {}),
            "kde": self.cfg.get("kde", {}),
            "pod": {
                "energy_threshold": self.cfg.get("pod", {}).get("energy_threshold", 0.99),
                "chosen_d": int(self.pod_projector.d) if self.pod_projector else 0,
                "svd": {"method": "economy", "randomized": False}
            },
            "non_stationarity": {
                "adf_alpha": float(self.adf_alpha),
                "adf_max_lags": self.adf_max_lags if self.adf_max_lags is not None else "auto",
                "trend_policy": "auto"
            },
            "movies": {
                "fps": int(self.movie_fps),
                "max_frames": int(self.movie_max_frames),
                "make_for": sorted(list(self.movies_for))
            },
            "io": {"save_dtype": self.save_dtype}
        }
        
        return run_json
    
    def _build_nonstationarity_json(self) -> dict:
        """Build non_stationarity_report.json matching spec schema."""
        if not self.nonstationarity_processor:
            return {}
        
        d = self.pod_projector.d if self.pod_projector else 0
        C = len(self.latents)
        
        report = {
            "adf_alpha": float(self.adf_alpha),
            "adf_max_lags": self.adf_max_lags if self.adf_max_lags is not None else "auto",
            "d": int(d),
            "C": int(C),
            "per_simulation": []
        }
        
        for c, (case_meta, Y) in enumerate(zip(self.nonstationarity_processor.case_meta, self.latents), 1):
            K = Y.shape[1]  # number of time steps
            
            decisions_list = []
            for coord_idx, decision in enumerate(case_meta.decisions):
                decisions_list.append({
                    "coord": int(coord_idx + 1),  # 1-based
                    "mode": decision.mode,
                    "adf_variant": decision.adf_variant,
                    "p_value": float(decision.adf_pvalue),
                    "lag": int(decision.adf_lag),
                    "notes": decision.notes
                })
            
            report["per_simulation"].append({
                "sim_id": c,
                "K": int(K),
                "decisions": decisions_list
            })
        
        return report
    
    def _final_validation(self) -> int:
        """Final checks: file existence, schema validation."""
        try:
            self.log("Running final validation...")
            
            # Check required files exist
            C = len(self.trajectories)
            for c in range(1, C + 1):
                sim_dir = self.outdir / f"sim_{c:04d}"
                required_files = [
                    "trajectories.csv",
                    "densities.csv",
                    "latents.csv",
                    "order_parameters.csv"
                ]
                for fname in required_files:
                    fpath = sim_dir / fname
                    if not fpath.exists():
                        self.log(f"Missing required file: {fpath}", "ERROR")
                        return 4
            
            # Check JSONs exist
            if not (self.outdir / "run.json").exists():
                self.log("Missing run.json", "ERROR")
                return 4
            
            if not (self.outdir / "non_stationarity_report.json").exists():
                self.log("Missing non_stationarity_report.json", "ERROR")
                return 4
            
            self.log("Validation passed")
            return 0
            
        except Exception as e:
            self.log(f"Final validation failed: {e}", "ERROR")
            return 4
