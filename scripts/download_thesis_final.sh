#!/bin/bash
# Download thesis-final experiment results from OSCAR (CSVs, JSONs, bootstrap — no large .npz density files)
set -e

EXPERIMENTS=(
  NDYN04_gas_thesis_final
  NDYN05_blackhole_thesis_final
  NDYN05_blackhole_VS_thesis_final
  NDYN04_gas_VS_thesis_final
  NDYN06_supernova_VS_thesis_final
  NDYN08_pure_vicsek_thesis_final
)

for EXP in "${EXPERIMENTS[@]}"; do
  echo "=== Downloading $EXP ==="
  REMOTE="oscar:~/scratch/oscar_output/$EXP"
  LOCAL="oscar_output/systematics/$EXP"

  mkdir -p "$LOCAL"/{MVAR,LSTM,WSINDy,rom_common,test}

  rsync -avz "$REMOTE/config_used.yaml" "$LOCAL/" 2>/dev/null || true
  rsync -avz "$REMOTE/summary.json" "$LOCAL/" 2>/dev/null || true

  rsync -avz --include='test_results.csv' --include='*.json' --exclude='*' \
    "$REMOTE/MVAR/" "$LOCAL/MVAR/" 2>/dev/null || true

  rsync -avz --include='test_results.csv' --include='*.json' --exclude='*' \
    "$REMOTE/LSTM/" "$LOCAL/LSTM/" 2>/dev/null || true

  rsync -avz --include='*.csv' --include='*.json' --include='bootstrap_*.npz' --include='*.yaml' --exclude='*' \
    "$REMOTE/WSINDy/" "$LOCAL/WSINDy/" 2>/dev/null || true

  rsync -avz -r --include='*/' --include='metadata.json' --include='r2_vs_time*.csv' --include='test_results.csv' --exclude='*' \
    "$REMOTE/test/" "$LOCAL/test/" 2>/dev/null || true

  rsync -avz --include='shift_align_data.npz' --exclude='*' \
    "$REMOTE/rom_common/" "$LOCAL/rom_common/" 2>/dev/null || true

  echo ""
done
echo "=== ALL DONE ==="
