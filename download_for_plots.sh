#!/bin/bash
# download_for_plots.sh — Download plot data from Oscar for ALL systematic experiments
# Downloads CSVs, POD basis, metadata, and WSINDy artifacts (no full density .npz)
# Run from: /Users/maria_1/Desktop/wsindy-manifold
#
# Step 1: ssh oscar   (authenticate with password + Duo 2FA)
# Step 2: exit        (back to local)
# Step 3: bash download_for_plots.sh
#
# To download only specific experiments:
#   bash download_for_plots.sh DO_CS01_swarm_C01_l05 NDYN02_flock

set -e
OSCAR="oscar"
REMOTE="~/wsindy-manifold/oscar_output"
LOCAL="oscar_output/systematics"

# ── All 54 systematic experiments ──────────────────────────────────────────
ALL_EXPERIMENTS=(
  # D'Orsogna — Collective Swarm (CS)
  DO_CS01_swarm_C01_l05      DO_CS01_swarm_C01_l05_VS
  DO_CS02_swarm_C05_l3       DO_CS02_swarm_C05_l3_VS
  DO_CS03_swarm_C09_l3       DO_CS03_swarm_C09_l3_VS
  # D'Orsogna — Double Mill (DM)
  DO_DM01_dmill_C09_l05      DO_DM01_dmill_C09_l05_VS
  # D'Orsogna — Double Ring (DR)
  DO_DR01_dring_C01_l01      DO_DR01_dring_C01_l01_VS
  DO_DR02_dring_C09_l09      DO_DR02_dring_C09_l09_VS
  # D'Orsogna — Escape Collapse (EC)
  DO_EC01_esccol_C2_l3       DO_EC01_esccol_C2_l3_VS
  DO_EC02_esccol_C3_l05      DO_EC02_esccol_C3_l05_VS
  # D'Orsogna — Escape Symmetric (ES)
  DO_ES01_escsym_C3_l09      DO_ES01_escsym_C3_l09_VS
  # D'Orsogna — Escape Unstable (EU)
  DO_EU01_escuns_C2_l2       DO_EU01_escuns_C2_l2_VS
  DO_EU02_escuns_C3_l3       DO_EU02_escuns_C3_l3_VS
  # D'Orsogna — Stable Mill (SM)
  DO_SM01_mill_C05_l01       DO_SM01_mill_C05_l01_VS
  DO_SM02_mill_C3_l01        DO_SM02_mill_C3_l01_VS
  DO_SM03_mill_C2_l05        DO_SM03_mill_C2_l05_VS
  # NDYN — Constant Speed
  NDYN01_crawl               NDYN01_crawl_VS
  NDYN02_flock               NDYN02_flock_VS
  NDYN03_sprint              NDYN03_sprint_VS
  NDYN04_gas                 NDYN04_gas_VS
  NDYN05_blackhole           NDYN05_blackhole_VS
  NDYN06_supernova           NDYN06_supernova_VS
  NDYN07_crystal             NDYN07_crystal_VS
  NDYN08_pure_vicsek
  NDYN09_longrange           NDYN09_longrange_VS
  NDYN10_shortrange          NDYN10_shortrange_VS
  NDYN11_noisy_collapse      NDYN11_noisy_collapse_VS
  NDYN12_fast_explosion      NDYN12_fast_explosion_VS
  NDYN13_chaos               NDYN13_chaos_VS
  NDYN14_varspeed
)

# Allow user to pass specific experiment names as arguments
if [ $# -gt 0 ]; then
  EXPERIMENTS=("$@")
else
  EXPERIMENTS=("${ALL_EXPERIMENTS[@]}")
fi

echo "============================================"
echo "  Downloading ${#EXPERIMENTS[@]} experiments from Oscar"
echo "============================================"
echo ""

# ── Step 0: Check which experiments exist on Oscar ─────────────────────────
echo "Checking which experiments have results on Oscar..."
AVAILABLE=$(ssh "$OSCAR" "ls -d $REMOTE/*/summary.json 2>/dev/null | sed 's|.*/oscar_output/||;s|/summary.json||'" 2>/dev/null || true)

FOUND=0
SKIPPED=0
DOWNLOADED=()

mkdir -p "$LOCAL"

for EXP in "${EXPERIMENTS[@]}"; do
  # Check if this experiment has results
  if ! echo "$AVAILABLE" | grep -qx "$EXP"; then
    echo "  SKIP: $EXP (no summary.json on Oscar — likely still running or failed)"
    ((SKIPPED++))
    continue
  fi

  ((FOUND++))
  echo "========================================"
  echo "  [$FOUND] Downloading: $EXP"
  echo "========================================"

  mkdir -p "$LOCAL/$EXP/rom_common" "$LOCAL/$EXP/MVAR" "$LOCAL/$EXP/LSTM" \
           "$LOCAL/$EXP/WSINDy" "$LOCAL/$EXP/test"

  # config_used.yaml + summary.json
  rsync -avz "$OSCAR:$REMOTE/$EXP/config_used.yaml" "$LOCAL/$EXP/" 2>/dev/null || true
  rsync -avz "$OSCAR:$REMOTE/$EXP/summary.json" "$LOCAL/$EXP/" 2>/dev/null || true

  # POD basis + shift align (rom_common/)
  rsync -avz \
    --include='pod_basis.npz' \
    --include='pod_basis_unaligned.npz' \
    --include='shift_align.npz' \
    --include='shift_align_data.npz' \
    --include='X_train_mean.npy' \
    --exclude='*' \
    "$OSCAR:$REMOTE/$EXP/rom_common/" "$LOCAL/$EXP/rom_common/" 2>/dev/null || true

  # MVAR test_results.csv + model
  rsync -avz \
    --include='test_results.csv' \
    --include='mvar_model.npz' \
    --exclude='*' \
    "$OSCAR:$REMOTE/$EXP/MVAR/" "$LOCAL/$EXP/MVAR/" 2>/dev/null || true

  # LSTM test_results.csv
  rsync -avz \
    --include='test_results.csv' \
    --exclude='*' \
    "$OSCAR:$REMOTE/$EXP/LSTM/" "$LOCAL/$EXP/LSTM/" 2>/dev/null || true

  # Create LSTM marker so detect_available_models finds it
  [ -f "$LOCAL/$EXP/LSTM/test_results.csv" ] && touch "$LOCAL/$EXP/LSTM/.exists"

  # WSINDy artifacts (model coefficients, bootstrap, discovery log)
  rsync -avz \
    --include='*.json' \
    --include='*.csv' \
    --include='*.npz' \
    --include='*.yaml' \
    --exclude='*' \
    "$OSCAR:$REMOTE/$EXP/WSINDy/" "$LOCAL/$EXP/WSINDy/" 2>/dev/null || true

  # Test metadata
  rsync -avz "$OSCAR:$REMOTE/$EXP/test/metadata.json" "$LOCAL/$EXP/test/" 2>/dev/null || true

  # R² vs time CSVs for ALL test runs (recursive into IC subdirs)
  rsync -avz \
    --include='*/' \
    --include='r2_vs_time*.csv' \
    --include='test_results.csv' \
    --exclude='*' \
    "$OSCAR:$REMOTE/$EXP/test/" "$LOCAL/$EXP/test/" 2>/dev/null || true

  DOWNLOADED+=("$EXP")
  echo ""
done

echo "============================================"
echo "  DOWNLOAD COMPLETE"
echo "============================================"
echo ""
echo "  Downloaded: ${#DOWNLOADED[@]}"
echo "  Skipped:    $SKIPPED (not ready on Oscar)"
# 'du -sh' on a large oscar_output/ can saturate login-node memory — omitted.
# If you need the size, run: interact -n 1 -m 8g -t 0:10:00  then  du -sh "$LOCAL"
echo ""

if [ $SKIPPED -gt 0 ]; then
  echo "  Re-run this script later to pick up remaining experiments."
  echo ""
fi

echo "Next step — run plots:"
echo "  python 3_models_plots.py --data_dir oscar_output/systematics --skip_kde --skip_phase --skip_wsindy_detail"
