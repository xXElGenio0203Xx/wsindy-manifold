#!/bin/bash
# Download WSINDy model artifacts from v3 experiments
set -e

WSINDY_EXPERIMENTS=(
  NDYN04_gas_wsindy_v3
  NDYN05_blackhole_wsindy_v3
  NDYN05_blackhole_VS_wsindy_v3
  NDYN06_supernova_VS_wsindy_v3
  NDYN08_pure_vicsek_wsindy_v3
)

for EXP in "${WSINDY_EXPERIMENTS[@]}"; do
  echo "=== Downloading $EXP/WSINDy ==="
  LOCAL="oscar_output/wsindy_v3/$EXP/WSINDy"
  mkdir -p "$LOCAL"
  rsync -avz --include='*.json' --include='*.csv' --include='*.npz' --include='*.yaml' --exclude='*' \
    "oscar:~/scratch/oscar_output/$EXP/WSINDy/" "$LOCAL/" 2>/dev/null || true
  echo ""
done
echo "=== DONE ==="
