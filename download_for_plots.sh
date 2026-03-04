#!/bin/bash
# download_for_plots.sh — Download lightweight plot data from Oscar
# Downloads only CSVs, POD basis, and metadata (no density .npz files)
# Run from: /Users/maria_1/Desktop/wsindy-manifold

set -e
OSCAR="oscar"
REMOTE="~/wsindy-manifold/oscar_output"
LOCAL="oscar_output"

# The 16 completed experiments (tasks 0-11, 14-17)
COMPLETED=(
  DO_CS01_swarm_C01_l05
  DO_CS01_swarm_C01_l05_VS
  DO_CS02_swarm_C05_l3
  DO_CS02_swarm_C05_l3_VS
  DO_CS03_swarm_C09_l3
  DO_CS03_swarm_C09_l3_VS
  DO_DM01_dmill_C09_l05
  DO_DM01_dmill_C09_l05_VS
  DO_DR01_dring_C01_l01
  DO_DR01_dring_C01_l01_VS
  DO_DR02_dring_C09_l09
  DO_DR02_dring_C09_l09_VS
  DO_EC02_esccol_C3_l05
  DO_EC02_esccol_C3_l05_VS
  DO_ES01_escsym_C3_l09
  DO_ES01_escsym_C3_l09_VS
)

mkdir -p "$LOCAL"

for EXP in "${COMPLETED[@]}"; do
  echo "========================================"
  echo "  Downloading: $EXP"
  echo "========================================"

  mkdir -p "$LOCAL/$EXP/rom_common" "$LOCAL/$EXP/MVAR" "$LOCAL/$EXP/LSTM" "$LOCAL/$EXP/test"

  # config_used.yaml
  rsync -avz "$OSCAR:$REMOTE/$EXP/config_used.yaml" "$LOCAL/$EXP/" 2>/dev/null || true

  # summary.json (runtime info)
  rsync -avz "$OSCAR:$REMOTE/$EXP/summary.json" "$LOCAL/$EXP/" 2>/dev/null || true

  # POD basis + shift align (rom_common/) — ~20MB each
  rsync -avz \
    --include='pod_basis.npz' \
    --include='pod_basis_unaligned.npz' \
    --include='shift_align.npz' \
    --include='shift_align_data.npz' \
    --include='X_train_mean.npy' \
    --exclude='*' \
    "$OSCAR:$REMOTE/$EXP/rom_common/" "$LOCAL/$EXP/rom_common/" 2>/dev/null || true

  # MVAR test_results.csv + model (for detect_available_models)
  rsync -avz \
    --include='test_results.csv' \
    --include='mvar_model.npz' \
    --exclude='*' \
    "$OSCAR:$REMOTE/$EXP/MVAR/" "$LOCAL/$EXP/MVAR/" 2>/dev/null || true

  # LSTM test_results.csv (lightweight)
  rsync -avz \
    --include='test_results.csv' \
    --exclude='*' \
    "$OSCAR:$REMOTE/$EXP/LSTM/" "$LOCAL/$EXP/LSTM/" 2>/dev/null || true

  # Create LSTM marker so detect_available_models finds it
  touch "$LOCAL/$EXP/LSTM/.exists"

  # Test metadata
  rsync -avz "$OSCAR:$REMOTE/$EXP/test/metadata.json" "$LOCAL/$EXP/test/" 2>/dev/null || true

  # R² vs time CSVs for ALL test runs (no density files!)
  rsync -avz \
    --include='*/' \
    --include='r2_vs_time*.csv' \
    --exclude='*' \
    "$OSCAR:$REMOTE/$EXP/test/" "$LOCAL/$EXP/test/" 2>/dev/null || true

  echo ""
done

echo "============================================"
echo "  DOWNLOAD COMPLETE"
echo "============================================"
echo ""
echo "Total size:"
du -sh "$LOCAL"
echo ""
echo "Experiments downloaded: ${#COMPLETED[@]}"
echo ""
echo "Next step — run plots:"
echo "  python 3_models_plots.py --data_dir oscar_output --skip_kde --skip_phase --skip_wsindy_detail"
