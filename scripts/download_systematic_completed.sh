#!/bin/bash
# Download thesis-extracted data for the 16 completed systematic experiments.
# Run from project root: bash scripts/download_systematic_completed.sh

set -euo pipefail

REMOTE="oscar:~/wsindy-manifold/oscar_output"
LOCAL="oscar_output"

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

echo "Downloading thesis data for ${#COMPLETED[@]} completed systematic experiments..."
echo ""

for name in "${COMPLETED[@]}"; do
  echo "=== ${name} ==="
  mkdir -p "${LOCAL}/${name}/MVAR" "${LOCAL}/${name}/LSTM" "${LOCAL}/${name}/rom_common" "${LOCAL}/${name}/test"

  # Extracted thesis data (kde, mass, spatial order, lp errors)
  rsync -avz \
    --include='kde_snapshots.npz' \
    --include='mass_timeseries.npz' \
    --include='spatial_order.npz' \
    --include='lp_errors.npz' \
    --include='mass_conservation_plot.png' \
    --exclude='*' \
    "${REMOTE}/${name}/" "${LOCAL}/${name}/" 2>&1 | grep -E "\.npz|\.png|sent" || true

  # MVAR test results + r2_vs_time
  rsync -avz --include='*/' --include='test_results.csv' --include='test/*/r2_vs_time*.csv' --exclude='*' \
    "${REMOTE}/${name}/MVAR/" "${LOCAL}/${name}/MVAR/" 2>/dev/null || true

  # LSTM test results + r2_vs_time
  rsync -avz --include='*/' --include='test_results.csv' --include='test/*/r2_vs_time*.csv' --exclude='*' \
    "${REMOTE}/${name}/LSTM/" "${LOCAL}/${name}/LSTM/" 2>/dev/null || true

  # POD bases + shift alignment data
  rsync -avz --include='pod_basis*.npz' --include='shift_align*.npz' --exclude='*' \
    "${REMOTE}/${name}/rom_common/" "${LOCAL}/${name}/rom_common/" 2>/dev/null || true

  # Config + metadata
  rsync -avz "${REMOTE}/${name}/config_used.yaml" "${LOCAL}/${name}/" 2>/dev/null || true
  rsync -avz "${REMOTE}/${name}/test/metadata.json" "${LOCAL}/${name}/test/" 2>/dev/null || true

  echo ""
done

echo "════════════════════════════════════════════"
echo "  Download complete. ${#COMPLETED[@]} experiments in ${LOCAL}/"
echo "════════════════════════════════════════════"
echo ""
echo "Next: python 3_models_plots.py --data_dir oscar_output --thesis --output_dir Thesis_Figures"
