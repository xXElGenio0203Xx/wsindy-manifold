#!/bin/bash
# download_new_completed.sh — Download all 22 completed experiments not yet local
set -euo pipefail

OSCAR="oscar"
REMOTE="~/wsindy-manifold/oscar_output"
LOCAL="oscar_output"

EXPERIMENTS=(
  NDYN04_gas_VS
  NDYN04_gas_VS_preproc_sqrt_none
  NDYN04_gas_VS_thesis_final
  NDYN04_gas_preproc_raw_none
  NDYN04_gas_preproc_raw_scale
  NDYN04_gas_preproc_raw_simplex
  NDYN04_gas_preproc_sqrt_none
  NDYN04_gas_preproc_sqrt_scale
  NDYN04_gas_preproc_sqrt_simplex
  NDYN04_gas_thesis_final
  NDYN04_gas_wsindy_v2
  NDYN05_blackhole
  NDYN05_blackhole_VS
  NDYN05_blackhole_VS_thesis_final
  NDYN05_blackhole_thesis_final
  NDYN06_supernova
  NDYN06_supernova_VS
  NDYN06_supernova_VS_thesis_final
  NDYN06_supernova_thesis_final
  NDYN08_pure_vicsek
  NDYN08_pure_vicsek_thesis_final
  NDYN08_pure_vicsek_wsindy_v2
)

TOTAL=${#EXPERIMENTS[@]}
COUNT=0

for EXP in "${EXPERIMENTS[@]}"; do
  ((COUNT++))
  echo "[$COUNT/$TOTAL] Downloading: $EXP"

  mkdir -p "$LOCAL/$EXP/rom_common" "$LOCAL/$EXP/MVAR" "$LOCAL/$EXP/LSTM" \
           "$LOCAL/$EXP/WSINDy" "$LOCAL/$EXP/test"

  rsync -avz "$OSCAR:$REMOTE/$EXP/config_used.yaml" "$LOCAL/$EXP/" 2>/dev/null || true
  rsync -avz "$OSCAR:$REMOTE/$EXP/summary.json" "$LOCAL/$EXP/" 2>/dev/null || true

  rsync -avz \
    --include='pod_basis.npz' --include='pod_basis_unaligned.npz' \
    --include='shift_align.npz' --include='shift_align_data.npz' \
    --include='X_train_mean.npy' --exclude='*' \
    "$OSCAR:$REMOTE/$EXP/rom_common/" "$LOCAL/$EXP/rom_common/" 2>/dev/null || true

  rsync -avz --include='test_results.csv' --include='mvar_model.npz' --exclude='*' \
    "$OSCAR:$REMOTE/$EXP/MVAR/" "$LOCAL/$EXP/MVAR/" 2>/dev/null || true

  rsync -avz --include='test_results.csv' --exclude='*' \
    "$OSCAR:$REMOTE/$EXP/LSTM/" "$LOCAL/$EXP/LSTM/" 2>/dev/null || true
  [ -f "$LOCAL/$EXP/LSTM/test_results.csv" ] && touch "$LOCAL/$EXP/LSTM/.exists"

  rsync -avz --include='*.json' --include='*.csv' --include='*.npz' --include='*.yaml' --exclude='*' \
    "$OSCAR:$REMOTE/$EXP/WSINDy/" "$LOCAL/$EXP/WSINDy/" 2>/dev/null || true

  rsync -avz "$OSCAR:$REMOTE/$EXP/test/metadata.json" "$LOCAL/$EXP/test/" 2>/dev/null || true
  rsync -avz \
    --include='*/' --include='r2_vs_time*.csv' --include='test_results.csv' \
    --include='metrics_summary*.json' --exclude='*' \
    "$OSCAR:$REMOTE/$EXP/test/" "$LOCAL/$EXP/test/" 2>/dev/null || true

  echo ""
done

echo "=== DOWNLOAD COMPLETE: $COUNT/$TOTAL experiments ==="
