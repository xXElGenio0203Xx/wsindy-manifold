#!/usr/bin/env python3
"""
N-Convergence WSINDy Runner
============================

Runs particle simulation + WSINDy PDE discovery for the pure Vicsek regime
at multiple particle counts N ∈ {10, 20, 50, 100, 200, 300}.

Designed for fast local execution with reduced training runs and bootstrap.
Outputs:
  - R²_wf for each N (for the thesis table)
  - Coefficient convergence data (for the thesis plot)
  - A generated PDF convergence figure
"""

import sys
import json
import time
import yaml
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rectsim.config_loader import load_config
from rectsim.ic_generator import generate_training_configs
from rectsim.simulation_runner import run_simulations_parallel
from wsindy.fields import build_field_data, FieldData
from wsindy.multifield import (
    build_default_library,
    resolve_regime_aware_library_settings,
    model_selection_multifield,
    bootstrap_multifield,
)
from wsindy.select import default_ell_grid
from wsindy.grid import GridSpec
from wsindy.pretty import to_text

# ── Configuration ──────────────────────────────────────────────────
N_VALUES = [10, 20, 50, 100, 200, 300]
BASE_CONFIG_PATH = Path("configs/n_convergence/NDYN08_pure_vicsek_N0050.yaml")
OUTPUT_ROOT = Path("oscar_output")

# Lean settings for fast local runs
TRAIN_RUNS_PER_IC = 5        # 5 per IC type (vs 80), generates ~70 configs
MAX_WSINDY_TRAJS = 40        # cap WSINDy trajectories (vs 340)
N_ELL = 8                    # model selection ell configs (vs 20)
N_LAMBDA = 40                # lambda grid points (vs 60)
BOOTSTRAP_B = 30             # bootstrap replicates (vs 200)
SUBSAMPLE = 3                # temporal subsample factor


def create_config_for_n(base_yaml_path, n_particles):
    """Load the base config and override N and training runs."""
    with open(base_yaml_path) as f:
        cfg = yaml.safe_load(f)

    cfg["sim"]["N"] = n_particles
    cfg["experiment_name"] = f"NDYN08_pure_vicsek_N{n_particles:04d}"

    # Reduce training runs for speed
    for ic_type in ["gaussian", "ring", "two_clusters", "uniform"]:
        ic_cfg = cfg.get("train_ic", {}).get(ic_type, {})
        if ic_cfg:
            ic_cfg["n_runs"] = TRAIN_RUNS_PER_IC

    return cfg


def run_single_n(n_particles):
    """Run simulation + WSINDy for a single particle count."""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  N = {n_particles} particles")
    print(f"{sep}")

    # ── Load / create config ────────────────────────────────────
    raw_cfg = create_config_for_n(BASE_CONFIG_PATH, n_particles)
    exp_name = raw_cfg["experiment_name"]
    out_dir = OUTPUT_ROOT / exp_name
    train_dir = out_dir / "train"
    wsindy_dir = out_dir / "WSINDy"
    out_dir.mkdir(parents=True, exist_ok=True)
    wsindy_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out_dir / "config_used.yaml", "w") as f:
        yaml.dump(raw_cfg, f, default_flow_style=False)

    # ── Parse config ────────────────────────────────────────────
    (base_config, density_nx, density_ny, density_bw,
     train_ic_config, _test_ic, _test_sim,
     rom_config, _eval_config) = load_config(out_dir / "config_used.yaml")

    # ── Step 1: Generate training simulations ───────────────────
    train_configs = generate_training_configs(train_ic_config, base_config)
    n_train = len(train_configs)
    print(f"  Simulating {n_train} training runs with N={n_particles} particles...")

    t0 = time.time()
    train_metadata, sim_time = run_simulations_parallel(
        configs=train_configs,
        base_config=base_config,
        output_dir=out_dir,
        density_nx=density_nx,
        density_ny=density_ny,
        density_bandwidth=density_bw,
        is_test=False,
    )
    print(f"  Simulations done in {sim_time:.1f}s")

    # Save metadata
    serializable_meta = []
    for m in train_metadata:
        sm = dict(m)
        for k, v in sm.items():
            if isinstance(v, np.generic):
                sm[k] = v.item()
            elif isinstance(v, np.ndarray):
                sm[k] = v.tolist()
            elif isinstance(v, dict):
                sm[k] = {kk: (vv.item() if isinstance(vv, np.generic) else vv) for kk, vv in v.items()}
        serializable_meta.append(sm)
    with open(train_dir / "metadata.json", "w") as f:
        json.dump(serializable_meta, f, indent=2)

    # ── Step 2: Load training densities ─────────────────────────
    print(f"  Loading training densities...")
    rng = np.random.default_rng(42)
    # Cap to MAX_WSINDY_TRAJS for speed
    if len(train_metadata) > MAX_WSINDY_TRAJS:
        indices = rng.choice(len(train_metadata), MAX_WSINDY_TRAJS, replace=False)
        use_meta = [train_metadata[i] for i in sorted(indices)]
    else:
        use_meta = list(train_metadata)

    train_densities = []
    selected_meta = []
    skipped = 0
    for m in use_meta:
        run_dir = train_dir / m["run_name"]
        dpath = run_dir / "density.npz"
        try:
            d = np.load(dpath)
            rho = d["rho"][::SUBSAMPLE]
            if rho.size == 0:
                raise ValueError("empty rho")
            train_densities.append(rho)
            selected_meta.append(m)
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"    SKIP {m['run_name']}: {e}")
    if skipped:
        print(f"    Skipped {skipped} bad density files")

    T_sub, nx, ny = train_densities[0].shape
    print(f"  Loaded {len(train_densities)} trajectories, shape ({T_sub}, {nx}, {ny})")

    # ── Step 3: Build field data ────────────────────────────────
    ref = np.load(train_dir / selected_meta[0]["run_name"] / "density.npz")
    xgrid = ref["xgrid"]
    ygrid = ref["ygrid"]
    dx = float(xgrid[1] - xgrid[0])
    dy = float(ygrid[1] - ygrid[0])
    Lx = float(xgrid[-1] - xgrid[0]) + dx
    Ly = float(ygrid[-1] - ygrid[0]) + dy
    dt_base = float(ref["times"][1] - ref["times"][0])
    dt = dt_base * SUBSAMPLE

    # Pure Vicsek: no Morse forces
    regime_settings = resolve_regime_aware_library_settings(
        forces_enabled=False, Ca=0.0, Cr=0.0,
        morse_requested=False, regime_class="auto",
    )
    regime_class = regime_settings["regime_class"]

    print(f"  Building field data (regime: {regime_class})...")
    field_data_list = []
    good_densities = []
    for i, m in enumerate(selected_meta):
        run_dir = train_dir / m["run_name"]
        rho_i = train_densities[i]
        traj_path = run_dir / "trajectory.npz"
        try:
            td = np.load(traj_path)
            traj_i = td["traj"][::SUBSAMPLE]
            vel_i = td["vel"][::SUBSAMPLE]
            fd = build_field_data(
                rho_i, traj_i, vel_i,
                xgrid, ygrid, Lx, Ly, dt,
                bandwidth=5.0,
                morse_params=None,
                center_flux=True,
            )
            field_data_list.append(fd)
            good_densities.append(rho_i)
        except Exception as e:
            print(f"    SKIP field data {m['run_name']}: {e}")
    print(f"  Built {len(field_data_list)} FieldData objects")

    # ── Step 4: WSINDy model selection ──────────────────────────
    mf_library = build_default_library(
        morse=False, rich=True,
        rho_strategy="continuity_first",
        regime_class=regime_class,
    )
    n_lib = sum(len(v) for v in mf_library.values())
    print(f"  Library: {n_lib} terms")

    ell_grid = default_ell_grid(T_sub, nx, ny, n_points=N_ELL)
    lambdas = np.logspace(-5, 2, N_LAMBDA)

    print(f"  Model selection: {N_ELL} ℓ configs × {len(field_data_list)} trajectories")
    t_ms = time.perf_counter()
    mf_result, best_ell = model_selection_multifield(
        field_data_list, mf_library, ell_grid,
        p=(3, 5, 5), stride=(2, 2, 2), lambdas=lambdas,
        rho_strategy="continuity_first",
        morse_params=None,
    )
    ms_time = time.perf_counter() - t_ms
    print(f"  Model selection done in {ms_time:.1f}s")
    print(f"  Best ℓ = {best_ell}")

    # ── Step 5: Bootstrap UQ ────────────────────────────────────
    print(f"  Bootstrap: {BOOTSTRAP_B} replicates...")
    t_boot = time.perf_counter()
    boot_result = bootstrap_multifield(
        field_data_list, mf_library,
        ell=best_ell, p=(3, 5, 5), stride=(2, 2, 2),
        lambdas=lambdas, B=BOOTSTRAP_B, seed=42,
    )
    boot_time = time.perf_counter() - t_boot
    print(f"  Bootstrap done in {boot_time:.1f}s")

    # ── Extract results ─────────────────────────────────────────
    result = {"N": n_particles}
    total_time = time.time() - t0

    for eq_name in ["rho", "px", "py"]:
        mdl = getattr(mf_result, f"{eq_name}_model")
        r2 = float(mdl.diagnostics.get("r2", 0))
        result[f"r2_wf_{eq_name}"] = r2
        result[f"n_active_{eq_name}"] = mdl.n_active
        result[f"pde_{eq_name}"] = to_text(mdl)
        result[f"coefficients_{eq_name}"] = {
            n: float(mdl.w[mdl.col_names.index(n)])
            for n in mdl.active_terms
        }
        # Bootstrap CIs
        if boot_result and eq_name in boot_result:
            ci = boot_result[eq_name]
            result[f"bootstrap_{eq_name}"] = {
                term: {"mean": float(ci["mean"].get(term, 0)),
                       "lo": float(ci["ci_lo"].get(term, 0)),
                       "hi": float(ci["ci_hi"].get(term, 0))}
                for term in mdl.active_terms
                if term in ci.get("mean", {})
            }

    result["total_time_s"] = round(total_time, 1)
    result["ms_time_s"] = round(ms_time, 1)
    result["boot_time_s"] = round(boot_time, 1)

    print(f"\n  Results for N={n_particles}:")
    print(f"    R²_wf(ρ) = {result['r2_wf_rho']:.5f}")
    print(f"    Active terms(ρ): {result['n_active_rho']}")
    print(f"    PDE(ρ): {result['pde_rho']}")
    print(f"    Total time: {total_time:.1f}s")

    # Save per-N results
    with open(wsindy_dir / "n_convergence_result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def generate_convergence_plot(all_results, output_path):
    """Generate the coefficient convergence plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [r["N"] for r in all_results]

    # Collect all unique rho coefficient names across all N
    all_terms = set()
    for r in all_results:
        all_terms.update(r.get("coefficients_rho", {}).keys())
    all_terms = sorted(all_terms)

    if not all_terms:
        print("  WARNING: No rho coefficients found, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2, 1]})

    # Left panel: coefficient values vs N
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_terms), 1)))
    for i, term in enumerate(all_terms):
        vals = []
        ns_for_term = []
        ci_lo_list = []
        ci_hi_list = []
        for r in all_results:
            c = r.get("coefficients_rho", {}).get(term)
            if c is not None:
                vals.append(c)
                ns_for_term.append(r["N"])
                boot = r.get(f"bootstrap_rho", {}).get(term, {})
                ci_lo_list.append(boot.get("lo", c))
                ci_hi_list.append(boot.get("hi", c))
        if vals:
            ax.plot(ns_for_term, vals, "o-", color=colors[i], label=term, markersize=5)
            if any(lo != hi for lo, hi in zip(ci_lo_list, ci_hi_list)):
                ax.fill_between(ns_for_term, ci_lo_list, ci_hi_list,
                                alpha=0.15, color=colors[i])
    ax.set_xlabel("Number of particles $N$")
    ax.set_ylabel("WSINDy coefficient value")
    ax.set_title("Coefficient convergence with $N$")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.grid(True, alpha=0.3)

    # Right panel: R²_wf(ρ) vs N
    ax2 = axes[1]
    r2_vals = [r["r2_wf_rho"] for r in all_results]
    ax2.plot(ns, r2_vals, "s-", color="steelblue", markersize=7)
    ax2.set_xlabel("Number of particles $N$")
    ax2.set_ylabel(r"$R^2_{\mathrm{wf}}(\rho)$")
    ax2.set_title("Weak-form fit quality")
    ax2.set_xscale("log")
    ax2.set_xticks(ns)
    ax2.set_xticklabels([str(n) for n in ns])
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(min(0.9, min(r2_vals) - 0.02), 1.001)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(f"{output_path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}.pdf / .png")


def main():
    print("=" * 60)
    print("  N-CONVERGENCE WSINDy RUNNER")
    print("=" * 60)
    print(f"  N values: {N_VALUES}")
    print(f"  Training runs per IC: {TRAIN_RUNS_PER_IC}")
    print(f"  Bootstrap: {BOOTSTRAP_B}")
    print(f"  N_ell: {N_ELL}")

    all_results = []
    for n in N_VALUES:
        result = run_single_n(n)
        all_results.append(result)

    # ── Summary table ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'N':>6} | {'R²_wf(ρ)':>10} | {'|A|':>4} | {'Time':>8}")
    print(f"  {'─'*6}-+-{'─'*10}-+-{'─'*4}-+-{'─'*8}")
    for r in all_results:
        print(f"  {r['N']:>6} | {r['r2_wf_rho']:>10.5f} | {r['n_active_rho']:>4} | {r['total_time_s']:>7.1f}s")

    # ── Save combined results ───────────────────────────────────
    results_path = Path("oscar_output/n_convergence_wsindy_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Combined results: {results_path}")

    # ── Generate plot ───────────────────────────────────────────
    plot_path = Path("Thesis_Figures/n_convergence_coefficients")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    generate_convergence_plot(all_results, plot_path)

    # ── Print LaTeX table rows ──────────────────────────────────
    print("\n  LaTeX table rows (WSINDy column):")
    for r in all_results:
        v = r["r2_wf_rho"]
        print(f"    {r['N']:>4} & ... & {v:.3f} \\\\")


if __name__ == "__main__":
    main()
