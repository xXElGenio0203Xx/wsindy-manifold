#!/bin/bash
# download_all_missing.sh — Download ALL missing results from OSCAR
# Covers timed-out jobs that still have partial results (MVAR, LSTM, test metrics)
set -euo pipefail

OSCAR="oscar"
REMOTE="~/wsindy-manifold/oscar_output"
LOCAL="oscar_output"

EXPERIMENTS=(
  DO_ES01_escsym_C3_l09
  NDYN04_gas_eta0p50_slim
  NDYN04_gas_eta1p50_slim
  NDYN04_gas_eta2p00_slim
  NDYN04_gas_lag2_bic
  NDYN04_gas_lag42_fpe
  NDYN04_gas_lag50_aic
  NDYN04_gas_lag6_hqic
  NDYN05_blackhole_eta0p05_slim
  NDYN05_blackhole_eta0p15_slim
  NDYN05_blackhole_lag10_hqic
  NDYN05_blackhole_lag2_bic
  NDYN05_blackhole_lag39_fpe
  NDYN05_blackhole_lag50_aic
  NDYN06_supernova_lag49_fpe
  NDYN06_supernova_wsindy_v3
  NDYN07_crystal_VS_wsindy_v3
  NDYN08_pure_vicsek_lag10_aic
  NDYN08_pure_vicsek_lag10_fpe
  NDYN08_pure_vicsek_lag2_bic
  NDYN08_pure_vicsek_lag4_hqic
  NDYN08_pure_vicsek_N0050
  NDYN08_pure_vicsek_N0100
  NDYN08_pure_vicsek_N0200
  NDYN08_pure_vicsek_N0300
  NDYN08_pure_vicsek_N0500
)

TOTAL=${#EXPERIMENTS[@]}
COUNT=0

for EXP in "${EXPERIMENTS[@]}"; do
  ((COUNT++))
  echo "[$COUNT/$TOTAL] Downloading: $EXP"

  mkdir -p "$LOCAL/$EXP/rom_common" "$LOCAL/$EXP/MVAR" "$LOCAL/$EXP/LSTM" \
           "$LOCAL/$EXP/WSINDy" "$LOCAL/$EXP/test"

  # config + summary
  rsync -avz "$OSCAR:$REMOTE/$EXP/config_used.yaml" "$LOCAL/$EXP/" 2>/dev/null || true
  rsync -avz "$OSCAR:$REMOTE/$EXP/summary.json" "$LOCAL/$EXP/" 2>/dev/null || true

  # POD basis + shift align
  rsync -avz \
    --include='pod_basis.npz' --include='pod_basis_unaligned.npz' \
    --include='shift_align.npz' --include='shift_align_data.npz' \
    --include='X_train_mean.npy' --exclude='*' \
    "$OSCAR:$REMOTE/$EXP/rom_common/" "$LOCAL/$EXP/rom_common/" 2>/dev/null || true

  # MVAR test_results.csv + model
  rsync -avz --include='test_results.csv' --include='mvar_model.npz' --exclude='*' \
    "$OSCAR:$REMOTE/$EXP/MVAR/" "$LOCAL/$EXP/MVAR/" 2>/dev/null || true

  # LSTM test_results.csv
  rsync -avz --include='test_results.csv' --exclude='*' \
    "$OSCAR:$REMOTE/$EXP/LSTM/" "$LOCAL/$EXP/LSTM/" 2>/dev/null || true
  [ -f "$LOCAL/$EXP/LSTM/test_results.csv" ] && touch "$LOCAL/$EXP/LSTM/.exists"

  # WSINDy artifacts
  rsync -avz --include='*.json' --include='*.csv' --include='*.npz' --include='*.yaml' --exclude='*' \
    "$OSCAR:$REMOTE/$EXP/WSINDy/" "$LOCAL/$EXP/WSINDy/" 2>/dev/null || true

  # Test metadata + r2_vs_time CSVs + metrics
  rsync -avz "$OSCAR:$REMOTE/$EXP/test/metadata.json" "$LOCAL/$EXP/test/" 2>/dev/null || true
  rsync -avz \
    --include='*/' --include='r2_vs_time*.csv' --include='test_results.csv' \
    --include='metrics_summary*.json' --exclude='*' \
    "$OSCAR:$REMOTE/$EXP/test/" "$LOCAL/$EXP/test/" 2>/dev/null || true

  echo ""
done

echo "=== DOWNLOAD COMPLETE: $COUNT/$TOTAL experiments ==="
