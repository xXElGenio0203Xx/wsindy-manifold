"""
Deep Analysis of Completed ROM Runs
====================================
Analyzes POD reconstruction, latent dynamics, order parameters,
and R² degradation over time for V2.2 (and optionally V2).

Run on OSCAR: python scripts/deep_analysis.py
"""

import numpy as np
import json
from pathlib import Path


def analyze_run(run_name):
    """Full diagnostic analysis of a completed run."""
    base = Path(f"oscar_output/{run_name}")
    if not base.exists():
        print(f"⚠️  {run_name} not found, skipping")
        return
    
    print(f"\n{'='*80}")
    print(f"DEEP ANALYSIS: {run_name}")
    print(f"{'='*80}")
    
    # =========================================================================
    # 1. POD BASIS ANALYSIS
    # =========================================================================
    print(f"\n--- 1. POD BASIS ANALYSIS ---")
    pod_file = base / "rom_common" / "pod_basis.npz"
    if pod_file.exists():
        pod = np.load(pod_file)
        print(f"   Files in pod_basis.npz: {list(pod.keys())}")
        
        if 'singular_values' in pod:
            sv = pod['singular_values']
            energy = sv**2 / np.sum(sv**2)
            cum_energy = np.cumsum(energy)
            print(f"   Total singular values: {len(sv)}")
            print(f"   Top 20 energies: {[f'{e:.4f}' for e in energy[:20]]}")
            print(f"   Cumulative energy:")
            for d in [5, 8, 10, 14, 19, 25, 30, 40, 50]:
                if d <= len(cum_energy):
                    print(f"     d={d:3d}: {cum_energy[d-1]*100:.1f}%")
            print(f"   Condition number (σ₁/σ_d): {sv[0]/sv[min(9, len(sv)-1)]:.2f}")
        
        if 'U_r' in pod:
            U_r = pod['U_r']
            print(f"   U_r shape: {U_r.shape} (n_pixels × d)")
        
        if 'X_mean' in pod:
            X_mean = pod['X_mean']
            print(f"   X_mean shape: {X_mean.shape}")
            print(f"   X_mean range: [{X_mean.min():.4f}, {X_mean.max():.4f}]")
            print(f"   X_mean total mass: {X_mean.sum():.2f}")
    else:
        print(f"   ⚠️  pod_basis.npz not found")
    
    # =========================================================================
    # 2. LATENT DATASET ANALYSIS
    # =========================================================================
    print(f"\n--- 2. LATENT DATASET ANALYSIS ---")
    latent_file = base / "rom_common" / "latent_dataset.npz"
    if latent_file.exists():
        lat = np.load(latent_file)
        print(f"   Files in latent_dataset.npz: {list(lat.keys())}")
        
        if 'X_latent' in lat:
            X_lat = lat['X_latent']
            print(f"   X_latent shape: {X_lat.shape}")
            print(f"   Latent stats per mode:")
            n_modes = X_lat.shape[1] if X_lat.ndim == 2 else X_lat.shape[-1]
            for i in range(min(n_modes, 10)):
                col = X_lat[:, i] if X_lat.ndim == 2 else X_lat.reshape(-1, n_modes)[:, i]
                print(f"     Mode {i}: mean={col.mean():.4f}, std={col.std():.4f}, "
                      f"range=[{col.min():.4f}, {col.max():.4f}]")
    else:
        print(f"   ⚠️  latent_dataset.npz not found")
    
    # =========================================================================
    # 3. TEST-BY-TEST ANALYSIS
    # =========================================================================
    print(f"\n--- 3. TEST CASE ANALYSIS ---")
    test_dir = base / "test"
    
    # Read MVAR test results
    mvar_results_file = base / "MVAR" / "test_results.csv"
    if mvar_results_file.exists():
        import csv
        with open(mvar_results_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        print(f"\n   MVAR Test Results ({len(rows)} test cases):")
        print(f"   {'Case':>6s}  {'R²_recon':>10s}  {'R²_latent':>10s}  {'R²_POD':>10s}")
        r2_recons = []
        r2_latents = []
        r2_pods = []
        for row in rows:
            r2r = float(row.get('r2_reconstructed', 'nan'))
            r2l = float(row.get('r2_latent', 'nan'))
            r2p = float(row.get('r2_pod', 'nan'))
            r2_recons.append(r2r)
            r2_latents.append(r2l)
            r2_pods.append(r2p)
            idx = row.get('test_idx', row.get('test_run', '?'))
            print(f"   {idx:>6s}  {r2r:>10.4f}  {r2l:>10.4f}  {r2p:>10.4f}")
        print(f"   {'MEAN':>6s}  {np.mean(r2_recons):>10.4f}  {np.mean(r2_latents):>10.4f}  {np.mean(r2_pods):>10.4f}")
        print(f"   {'STD':>6s}  {np.std(r2_recons):>10.4f}  {np.std(r2_latents):>10.4f}  {np.std(r2_pods):>10.4f}")
        print(f"   {'MIN':>6s}  {np.min(r2_recons):>10.4f}  {np.min(r2_latents):>10.4f}  {np.min(r2_pods):>10.4f}")
        print(f"   {'MAX':>6s}  {np.max(r2_recons):>10.4f}  {np.max(r2_latents):>10.4f}  {np.max(r2_pods):>10.4f}")
    
    # Read LSTM test results
    lstm_results_file = base / "LSTM" / "test_results.csv"
    if lstm_results_file.exists():
        import csv
        with open(lstm_results_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        r2_recons_lstm = [float(r.get('r2_reconstructed', 'nan')) for r in rows]
        print(f"\n   LSTM Test Results: mean R²_recon = {np.mean(r2_recons_lstm):.4f}, "
              f"std = {np.std(r2_recons_lstm):.4f}")
    
    # =========================================================================
    # 4. DETAILED ANALYSIS OF BEST AND WORST TEST CASES
    # =========================================================================
    print(f"\n--- 4. DETAILED CASE ANALYSIS (best & worst) ---")
    
    if mvar_results_file.exists():
        best_idx = int(np.argmax(r2_recons))
        worst_idx = int(np.argmin(r2_recons))
        
        for label, idx in [("BEST", best_idx), ("WORST", worst_idx)]:
            case_dir = test_dir / f"test_{idx:03d}"
            print(f"\n   [{label}] Test case {idx} (R²={r2_recons[idx]:.4f}):")
            
            # Load true density
            true_file = case_dir / "density_true.npz"
            pred_mvar_file = case_dir / "density_pred_mvar.npz"
            
            if true_file.exists() and pred_mvar_file.exists():
                true_data = np.load(true_file)
                pred_data = np.load(pred_mvar_file)
                
                print(f"     True density keys: {list(true_data.keys())}")
                print(f"     Pred density keys: {list(pred_data.keys())}")
                
                rho_true = true_data['rho'] if 'rho' in true_data else true_data[list(true_data.keys())[0]]
                rho_pred = pred_data['rho'] if 'rho' in pred_data else pred_data[list(pred_data.keys())[0]]
                
                print(f"     True shape: {rho_true.shape}, range: [{rho_true.min():.4f}, {rho_true.max():.4f}]")
                print(f"     Pred shape: {rho_pred.shape}, range: [{rho_pred.min():.4f}, {rho_pred.max():.4f}]")
                
                # Mass conservation check
                true_mass = np.array([rho_true[t].sum() for t in range(len(rho_true))])
                pred_mass = np.array([rho_pred[t].sum() for t in range(len(rho_pred))])
                print(f"     True mass: mean={true_mass.mean():.2f}, std={true_mass.std():.4f}")
                print(f"     Pred mass: mean={pred_mass.mean():.2f}, std={pred_mass.std():.4f}")
                
                # Negative fraction in prediction
                neg_frac = np.mean(rho_pred < 0) * 100
                print(f"     Negative pixels in pred: {neg_frac:.1f}%")
                
                # Frame-by-frame R² decay
                n_frames = min(len(rho_true), len(rho_pred))
                # Find forecast start (where pred diverges from true)
                # The conditioning window should match exactly
                frame_r2 = []
                for t in range(n_frames):
                    ss_res = np.sum((rho_true[t].flatten() - rho_pred[t].flatten())**2)
                    ss_tot = np.sum((rho_true[t].flatten() - rho_true[t].flatten().mean())**2)
                    r2_t = 1 - ss_res / max(ss_tot, 1e-12)
                    frame_r2.append(r2_t)
                
                frame_r2 = np.array(frame_r2)
                # Find where R² first drops below 0
                neg_start = np.where(frame_r2 < 0)[0]
                print(f"     Frame-by-frame R²:")
                print(f"       First 5: {[f'{r:.3f}' for r in frame_r2[:5]]}")
                print(f"       Last 5:  {[f'{r:.3f}' for r in frame_r2[-5:]]}")
                if len(neg_start) > 0:
                    print(f"       R² drops below 0 at frame {neg_start[0]}/{n_frames}")
                else:
                    print(f"       R² stays positive throughout")
    
    # =========================================================================
    # 5. R² vs TIME ANALYSIS
    # =========================================================================
    print(f"\n--- 5. R² vs TIME (averaged across test cases) ---")
    r2_time_all = []
    n_test_cases = len(list(test_dir.glob("test_*"))) if test_dir.exists() else 0
    
    for idx in range(n_test_cases):
        r2_file = test_dir / f"test_{idx:03d}" / "r2_vs_time.csv"
        if r2_file.exists():
            import csv
            with open(r2_file) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                r2_vals = []
                for row in rows:
                    # Try different possible column names
                    for key in ['r2_reconstructed', 'r2', 'R2']:
                        if key in row:
                            r2_vals.append(float(row[key]))
                            break
                if r2_vals:
                    r2_time_all.append(r2_vals)
    
    if r2_time_all:
        # Pad to same length
        max_len = max(len(r) for r in r2_time_all)
        r2_matrix = np.full((len(r2_time_all), max_len), np.nan)
        for i, r in enumerate(r2_time_all):
            r2_matrix[i, :len(r)] = r
        
        mean_r2 = np.nanmean(r2_matrix, axis=0)
        print(f"   Total time steps: {max_len}")
        # Print every 5th step
        step = max(1, max_len // 20)
        print(f"   {'Step':>6s}  {'Mean R²':>10s}  {'Median':>10s}  {'Min':>10s}  {'Max':>10s}")
        for t in range(0, max_len, step):
            col = r2_matrix[:, t]
            valid = col[~np.isnan(col)]
            if len(valid) > 0:
                print(f"   {t:>6d}  {np.mean(valid):>10.4f}  {np.median(valid):>10.4f}  "
                      f"{np.min(valid):>10.4f}  {np.max(valid):>10.4f}")
        # Last step
        t = max_len - 1
        col = r2_matrix[:, t]
        valid = col[~np.isnan(col)]
        if len(valid) > 0:
            print(f"   {t:>6d}  {np.mean(valid):>10.4f}  {np.median(valid):>10.4f}  "
                  f"{np.min(valid):>10.4f}  {np.max(valid):>10.4f}")
    else:
        print(f"   ⚠️  No r2_vs_time.csv files found")
    
    # =========================================================================
    # 6. ORDER PARAMETER ANALYSIS
    # =========================================================================
    print(f"\n--- 6. ORDER PARAMETERS ---")
    for idx in range(min(3, n_test_cases)):
        metrics_file = test_dir / f"test_{idx:03d}" / "metrics_summary.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            print(f"\n   Test {idx} metrics: {list(metrics.keys())}")
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    print(f"     {key}: {val:.4f}")
                elif isinstance(val, dict):
                    print(f"     {key}:")
                    for k2, v2 in val.items():
                        if isinstance(v2, (int, float)):
                            print(f"       {k2}: {v2:.4f}")
    
    # =========================================================================
    # 7. LATENT TRAJECTORY ANALYSIS (best & worst case)
    # =========================================================================
    print(f"\n--- 7. LATENT TRAJECTORY COMPARISON ---")
    if pod_file.exists():
        pod = np.load(pod_file)
        U_r = pod['U'] if 'U' in pod else pod['U_r']
        X_mean = np.load(base / "rom_common" / "X_train_mean.npy") if (base / "rom_common" / "X_train_mean.npy").exists() else (pod['X_mean'] if 'X_mean' in pod else np.zeros(U_r.shape[0]))
        
        for label, idx in [("BEST", best_idx), ("WORST", worst_idx)]:
            case_dir = test_dir / f"test_{idx:03d}"
            true_file = case_dir / "density_true.npz"
            pred_file = case_dir / "density_pred_mvar.npz"
            
            if true_file.exists() and pred_file.exists():
                rho_true = np.load(true_file)['rho']
                rho_pred = np.load(pred_file)['rho']
                
                # Project both to latent space
                true_flat = rho_true.reshape(len(rho_true), -1)
                pred_flat = rho_pred.reshape(len(rho_pred), -1)
                
                true_lat = (true_flat - X_mean) @ U_r
                pred_lat = (pred_flat - X_mean) @ U_r
                
                n = min(len(true_lat), len(pred_lat))
                print(f"\n   [{label}] Test {idx} latent trajectories ({n} frames):")
                
                for mode in range(min(5, true_lat.shape[1])):
                    err = np.abs(true_lat[:n, mode] - pred_lat[:n, mode])
                    rel_err = err / (np.abs(true_lat[:n, mode]).max() + 1e-12)
                    print(f"     Mode {mode}: true_range=[{true_lat[:n,mode].min():.3f}, {true_lat[:n,mode].max():.3f}], "
                          f"pred_range=[{pred_lat[:n,mode].min():.3f}, {pred_lat[:n,mode].max():.3f}], "
                          f"mean_abs_err={err.mean():.4f}, max_rel_err={rel_err.max():.4f}")
                
                # When does the latent error blow up?
                total_err = np.sqrt(np.sum((true_lat[:n] - pred_lat[:n])**2, axis=1))
                true_norm = np.sqrt(np.sum(true_lat[:n]**2, axis=1))
                rel_total = total_err / (true_norm + 1e-12)
                
                print(f"     Total latent error: first={total_err[0]:.4f}, "
                      f"last={total_err[-1]:.4f}, max={total_err.max():.4f}")
                print(f"     Relative error: first={rel_total[0]:.4f}, "
                      f"last={rel_total[-1]:.4f}, max={rel_total.max():.4f}")
                
                # Find blowup point (where error > signal)
                blowup = np.where(rel_total > 1.0)[0]
                if len(blowup) > 0:
                    print(f"     ⚠️  Error exceeds signal at frame {blowup[0]}/{n}")
                else:
                    print(f"     ✓ Error stays below signal norm throughout")

    print(f"\n{'='*80}")
    print(f"END ANALYSIS: {run_name}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Analyze all completed runs
    for run in ["synthesis_v2_2", "synthesis_v2"]:
        analyze_run(run)
