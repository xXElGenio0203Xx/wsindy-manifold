#!/bin/bash
# Download thesis-relevant lightweight data from Oscar.
# Run from the project root: bash scripts/download_thesis_data.sh
#
# Downloads:
#   - test_results.csv  (MVAR + LSTM + WSINDy per experiment)
#   - r2_vs_time_*.csv  (per test run)
#   - pod_basis.npz + pod_basis_unaligned.npz (SVD spectra)
#   - kde_snapshots.npz (pre-extracted density frames)
#   - mass_timeseries.npz (pre-extracted mass curves)
#   - spatial_order.npz  (spatial order parameter time series)
#   - lp_errors.npz      (relative L1/L2/Linf per model per test)
#   - mass_conservation_plot.png (rendered on Oscar)
#   - shift_align*.npz   (sPOD shift data for phase dynamics)
#   - config_used.yaml (experiment configuration)
#   - test/metadata.json
#
# Excludes all heavy files: density_true/pred, trajectory, train data.

set -euo pipefail

REMOTE="oscar:~/wsindy-manifold/oscar_output/"
LOCAL="oscar_output/"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       Download thesis data from Oscar                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Create local dir
mkdir -p "$LOCAL"

# --- 1. CSV files (test_results, r2_vs_time) ---
echo ""
echo "── 1/8  test_results.csv (MVAR + LSTM + WSINDy) ──"
rsync -avz --progress \
  --include='*/' \
  --include='MVAR/test_results.csv' \
  --include='LSTM/test_results.csv' \
  --include='WSINDy/test_results.csv' \
  --exclude='*' \
  "$REMOTE" "$LOCAL"

echo ""
echo "── 2/8  r2_vs_time_*.csv (per test run) ──"
rsync -avz --progress \
  --include='*/' \
  --include='r2_vs_time*.csv' \
  --exclude='*' \
  "$REMOTE" "$LOCAL"

# --- 2. POD basis files ---
echo ""
echo "── 3/8  POD basis files + shift data ──"
rsync -avz --progress \
  --include='*/' \
  --include='rom_common/pod_basis.npz' \
  --include='rom_common/pod_basis_unaligned.npz' \
  --include='rom_common/shift_align*.npz' \
  --exclude='*' \
  "$REMOTE" "$LOCAL"

# --- 3. Pre-extracted thesis data ---
echo ""
echo "── 4/8  kde_snapshots.npz ──"
rsync -avz --progress \
  --include='*/' \
  --include='kde_snapshots.npz' \
  --exclude='*' \
  "$REMOTE" "$LOCAL"

echo ""
echo "── 5/8  mass_timeseries.npz + mass_conservation_plot.png ──"
rsync -avz --progress \
  --include='*/' \
  --include='mass_timeseries.npz' \
  --include='mass_conservation_plot.png' \
  --exclude='*' \
  "$REMOTE" "$LOCAL"

echo ""
echo "── 6/8  spatial_order.npz ──"
rsync -avz --progress \
  --include='*/' \
  --include='spatial_order.npz' \
  --exclude='*' \
  "$REMOTE" "$LOCAL"

echo ""
echo "── 7/8  lp_errors.npz ──"
rsync -avz --progress \
  --include='*/' \
  --include='lp_errors.npz' \
  --exclude='*' \
  "$REMOTE" "$LOCAL"

# --- 4. Config + metadata ---
echo ""
echo "── 8/8  config_used.yaml + metadata.json ──"
rsync -avz --progress \
  --include='*/' \
  --include='config_used.yaml' \
  --include='test/metadata.json' \
  --exclude='*' \
  "$REMOTE" "$LOCAL"

echo ""
echo "════════════════════════════════════════════"
echo "  Download complete.  Local data in: $LOCAL"
echo "════════════════════════════════════════════"
echo ""
echo "Next: python 3_models_plots.py --data_dir oscar_output --thesis --output_dir Thesis_Figures"
