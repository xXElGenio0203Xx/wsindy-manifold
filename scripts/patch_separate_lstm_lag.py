#!/usr/bin/env python3
"""
Patch ROM_pipeline.py to build SEPARATE latent datasets for MVAR and LSTM
when they have different lag values.

Previously: One shared dataset built with MVAR's lag, LSTM forced to use it.
After:      If LSTM lag != MVAR lag, LSTM gets its own dataset with its own lag.

Run on Oscar:
    python scripts/patch_separate_lstm_lag.py
"""
import re

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.path.join(os.path.dirname(SCRIPT_DIR), "ROM_pipeline.py")

# Read current file
with open(PIPELINE, 'r') as f:
    content = f.read()

# ─── PATCH 1: Replace the shared dataset construction ───────────────────────
OLD_BLOCK = '''    # Determine lag (use MVAR lag if enabled, else LSTM lag)
    if mvar_enabled:
        lag = models_cfg['mvar'].get('lag', 5)
    else:
        lag = models_cfg['lstm'].get('lag', 20)
    
    print(f"\\nUsing lag={lag} for windowed dataset")
    print(f"   (This will be shared by both MVAR and LSTM)")
    
    # Build windowed dataset
    X_all, Y_all = build_latent_dataset(y_trajs, lag=lag)
    
    print(f"\\n✓ Windowed dataset built:")
    print(f"   X_all: {X_all.shape}  [N_samples, lag, d]")
    print(f"   Y_all: {Y_all.shape}  [N_samples, d]")
    print(f"   Total samples: {X_all.shape[0]:,}")

    # Build multi-step targets for LSTM supervised rollout loss
    Y_multi = None
    X_lstm = X_all  # default: same dataset for LSTM and MVAR
    Y_lstm = Y_all
    if lstm_enabled:
        lstm_ms_cfg = models_cfg.get('lstm', {})
        ms_enabled = lstm_ms_cfg.get('multistep_loss', False)
        ms_k = lstm_ms_cfg.get('multistep_k', 5)
        if ms_enabled and ms_k > 1:
            X_lstm, Y_multi = build_multistep_latent_dataset(y_trajs, lag=lag, k_steps=ms_k)
            Y_lstm = Y_multi[:, 0, :]  # 1-step target aligned with X_lstm
            print(f"\\n✓ Multi-step targets built:")
            print(f"   X_lstm:  {X_lstm.shape}  (aligned with Y_multi)")
            print(f"   Y_multi: {Y_multi.shape}  [N_samples, k={ms_k}, d]")
            print(f"   (Dropped {X_all.shape[0] - X_lstm.shape[0]} samples near trajectory ends)")'''

NEW_BLOCK = '''    # Determine lag for each model (MVAR builds its own internally;
    # this dataset is primarily for LSTM, but we keep X_all for legacy compat)
    mvar_lag = models_cfg.get('mvar', {}).get('lag', 5) if mvar_enabled else 5
    lstm_lag_cfg = models_cfg.get('lstm', {}).get('lag', 20) if lstm_enabled else mvar_lag
    
    # Use MVAR lag as the "shared" lag (backward compat for saved dataset)
    lag = mvar_lag if mvar_enabled else lstm_lag_cfg
    
    print(f"\\nMVAR lag={mvar_lag}, LSTM lag={lstm_lag_cfg}")
    
    # Build shared windowed dataset (MVAR actually builds its own in trainer)
    X_all, Y_all = build_latent_dataset(y_trajs, lag=lag)
    print(f"\\n✓ Shared windowed dataset built (lag={lag}):")
    print(f"   X_all: {X_all.shape}  [N_samples, lag, d]")
    print(f"   Y_all: {Y_all.shape}  [N_samples, d]")
    print(f"   Total samples: {X_all.shape[0]:,}")

    # Build LSTM-specific dataset if its lag differs from the shared lag
    Y_multi = None
    if lstm_enabled and lstm_lag_cfg != lag:
        print(f"\\n   Building SEPARATE LSTM dataset with lag={lstm_lag_cfg}")
        X_lstm, Y_lstm = build_latent_dataset(y_trajs, lag=lstm_lag_cfg)
        print(f"   X_lstm: {X_lstm.shape}  [N_samples, lag={lstm_lag_cfg}, d]")
        print(f"   Y_lstm: {Y_lstm.shape}  [N_samples, d]")
        
        # Multi-step targets for LSTM supervised rollout loss
        lstm_ms_cfg = models_cfg.get('lstm', {})
        ms_enabled = lstm_ms_cfg.get('multistep_loss', False)
        ms_k = lstm_ms_cfg.get('multistep_k', 5)
        if ms_enabled and ms_k > 1:
            X_lstm, Y_multi = build_multistep_latent_dataset(y_trajs, lag=lstm_lag_cfg, k_steps=ms_k)
            Y_lstm = Y_multi[:, 0, :]
            print(f"   Multi-step targets: Y_multi {Y_multi.shape}")
    else:
        X_lstm = X_all
        Y_lstm = Y_all
        if lstm_enabled:
            lstm_ms_cfg = models_cfg.get('lstm', {})
            ms_enabled = lstm_ms_cfg.get('multistep_loss', False)
            ms_k = lstm_ms_cfg.get('multistep_k', 5)
            if ms_enabled and ms_k > 1:
                X_lstm, Y_multi = build_multistep_latent_dataset(y_trajs, lag=lag, k_steps=ms_k)
                Y_lstm = Y_multi[:, 0, :]
                print(f"\\n✓ Multi-step targets built:")
                print(f"   X_lstm:  {X_lstm.shape}  (aligned with Y_multi)")
                print(f"   Y_multi: {Y_multi.shape}  [N_samples, k={ms_k}, d]")
                print(f"   (Dropped {X_all.shape[0] - X_lstm.shape[0]} samples near trajectory ends)")'''

if OLD_BLOCK not in content:
    # Try with the Oscar line-continuation formatting
    print("WARNING: Exact old block not found. Trying flexible match...")
    # Normalize whitespace for matching
    old_normalized = re.sub(r'[ \t]+', ' ', OLD_BLOCK.strip())
    content_normalized = re.sub(r'[ \t]+', ' ', content.strip())
    if old_normalized in content_normalized:
        print("  Found with normalized whitespace — applying patch")
    else:
        print("ERROR: Could not find the target block to patch.")
        print("The pipeline may have already been patched or modified.")
        import sys
        sys.exit(1)

content = content.replace(OLD_BLOCK, NEW_BLOCK)

# Verify the replacement happened
if NEW_BLOCK not in content:
    print("ERROR: Replacement failed")
    import sys
    sys.exit(1)

# Write back
with open(PIPELINE, 'w') as f:
    f.write(content)

print(f"✓ Patched {PIPELINE}: LSTM now gets its own dataset when lag differs from MVAR")
print(f"  MVAR trainer already builds its own dataset from rom_config — unaffected")
