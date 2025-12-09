"""
Summary JSON Generation Module
===============================

Creates comprehensive pipeline summary with metrics and metadata.
"""

import json
import time
from pathlib import Path


def generate_summary_json(
    metrics_df,
    ic_metrics,
    ic_types,
    pod_data,
    train_metadata,
    test_metadata,
    base_config_sim,
    degradation_info,
    output_dir
):
    """
    Generate comprehensive summary JSON with all pipeline results.
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        Test metrics for all runs
    ic_metrics : dict
        Aggregated metrics by IC type
    ic_types : list
        List of IC types
    pod_data : dict
        POD model data (singular values, energy, etc.)
    train_metadata : list
        Training run metadata
    test_metadata : list
        Test run metadata
    base_config_sim : dict
        Base simulation configuration
    degradation_info : dict
        Time-resolved degradation statistics
    output_dir : Path
        Directory to save summary JSON
    
    Returns
    -------
    summary : dict
        Complete summary dictionary
    """
    
    output_dir = Path(output_dir)
    
    # Determine IC key - metrics_df should already have the correct column name
    ic_key = 'ic_type' if 'ic_type' in metrics_df.columns else 'distribution'
    
    # Extract POD parameters
    R_POD = len(pod_data['singular_values'])
    actual_energy = pod_data['cumulative_energy'][R_POD - 1]
    n_space = 64 * 64  # DENSITY_NX * DENSITY_NY
    compression_ratio = (1 - R_POD / n_space) * 100
    
    # Extract MVAR parameters
    P_LAG = int(pod_data.get('p_lag', 1))
    train_r2 = float(pod_data.get('train_r2', 0.0))
    train_rmse = float(pod_data.get('train_rmse', 0.0))
    
    # Rank IC types by performance
    ic_ranking = []
    for ic_type in ic_types:
        if ic_type in ic_metrics:
            ic_data = metrics_df[metrics_df[ic_key] == ic_type]
            ic_ranking.append({
                "ic_type": ic_type,
                "mean_r2": float(ic_data["r2"].mean()),
                "std_r2": float(ic_data["r2"].std()),
                "mean_l2_error": float(ic_data["median_e2"].mean()),
                "best_r2": float(ic_metrics[ic_type]["best_r2"]),
                "best_run": ic_metrics[ic_type]["best_run"]
            })
    
    ic_ranking.sort(key=lambda x: x["mean_r2"], reverse=True)
    
    # Training ensemble distribution
    n_train = len(train_metadata)
    ic_distribution = {
        ic: sum(1 for m in train_metadata if m.get(ic_key, m.get("ic_type")) == ic)
        for ic in ic_types
    }
    
    # Test ensemble distribution
    n_test = len(test_metadata)
    runs_per_ic = n_test // len(ic_types) if ic_types else 0
    
    # Standard configuration values
    DENSITY_NX = 64
    DENSITY_NY = 64
    DENSITY_BANDWIDTH = 2.0
    TARGET_ENERGY = 0.995
    RIDGE_ALPHA = 1e-6
    
    # Build comprehensive summary
    summary = {
        "model_parameters": {
            "simulation": {
                "N_particles": 40,
                "domain_size": {
                    "Lx": float(base_config_sim["Lx"]),
                    "Ly": float(base_config_sim["Ly"])
                },
                "boundary_conditions": "periodic",
                "time": {
                    "T_total": 2.0,
                    "dt": 0.1,
                    "save_every": 1
                },
                "model": "vicsek_morse_discrete",
                "speed": 1.0,
                "interaction_radius": 2.0,
                "noise": {
                    "kind": "gaussian",
                    "eta": 0.3
                },
                "forces_enabled": False
            },
            "density_estimation": {
                "method": "KDE",
                "resolution": {"nx": int(DENSITY_NX), "ny": int(DENSITY_NY)},
                "bandwidth": float(DENSITY_BANDWIDTH)
            },
            "initial_conditions": {
                "types": ic_types,
                "description": "Stratified sampling across IC types"
            }
        },
        "training_metrics": {
            "ensemble": {
                "n_simulations": int(n_train),
                "ic_distribution": ic_distribution
            },
            "pod": {
                "target_energy": float(TARGET_ENERGY),
                "n_modes_selected": int(R_POD),
                "actual_energy_captured": float(actual_energy),
                "spatial_dof": int(n_space),
                "compression_ratio_percent": float(compression_ratio),
                "compression_description": f"{R_POD}/{n_space} modes ({compression_ratio:.2f}% compression)"
            },
            "mvar": {
                "latent_dimension": int(R_POD),
                "lag_order": int(P_LAG),
                "ridge_alpha": float(RIDGE_ALPHA),
                "training_r2": float(train_r2),
                "training_rmse": float(train_rmse)
            }
        },
        "test_results": {
            "ensemble": {
                "n_test_runs": int(n_test),
                "stratification": "Equal distribution across IC types",
                "runs_per_ic": int(runs_per_ic)
            },
            "overall_performance": {
                "mean_r2": float(metrics_df["r2"].mean()),
                "std_r2": float(metrics_df["r2"].std()),
                "median_r2": float(metrics_df["r2"].median()),
                "min_r2": float(metrics_df["r2"].min()),
                "max_r2": float(metrics_df["r2"].max())
            },
            "error_metrics": {
                "mean_l2_error": float(metrics_df["median_e2"].mean()),
                "std_l2_error": float(metrics_df["median_e2"].std()),
                "p10_l2_error": float(metrics_df["p10_e2"].mean()),
                "p90_l2_error": float(metrics_df["p90_e2"].mean()),
                "mean_mass_error": float(metrics_df["mean_mass_error"].mean()),
                "max_mass_error": float(metrics_df["max_mass_error"].max()),
                "mean_tau_tolerance": float(metrics_df["tau_tol"].mean())
            },
            "ic_performance_ranking": ic_ranking,
            "best_overall_run": {
                "run_name": metrics_df.loc[metrics_df["r2"].idxmax(), "run_name"],
                "ic_type": metrics_df.loc[metrics_df["r2"].idxmax(), ic_key],
                "r2": float(metrics_df["r2"].max())
            },
            "worst_overall_run": {
                "run_name": metrics_df.loc[metrics_df["r2"].idxmin(), "run_name"],
                "ic_type": metrics_df.loc[metrics_df["r2"].idxmin(), ic_key],
                "r2": float(metrics_df["r2"].min())
            }
        },
        "pipeline_metadata": {
            "output_directory": str(output_dir),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "1.0",
            "pipeline_type": "unified"
        }
    }
    
    # Add time-resolved degradation info if available
    if degradation_info:
        summary["time_resolved_analysis"] = degradation_info
    
    # Save summary JSON
    summary_path = output_dir / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print("✓ Comprehensive summary JSON saved")
    
    return summary
