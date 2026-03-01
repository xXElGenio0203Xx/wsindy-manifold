#!/bin/bash
# ============================================================================
# Download visualization-required files from Oscar for DYN + LST4/7
# ============================================================================
# Pulls ONLY what run_visualizations.py needs:
#   - Root: config_used.yaml, summary.json, runtime_comparison.json
#   - rom_common/: pod_basis.npz, X_train_mean.npy
#   - MVAR/: mvar_model.npz, test_results.csv, runtime_profile.json
#   - LSTM/: test_results.csv, runtime_profile.json, training_log.csv
#   - train/metadata.json + first run's density.npz (grid info)
#   - test/metadata.json, index_mapping.csv
#   - test/test_XXX/: density_true.npz, density_pred*.npz,
#                     metrics_summary*.json, r2_vs_time*.csv, trajectory.npz
#
# ~110MB per DYN experiment (no LSTM), ~150MB for LST experiments (with LSTM)
# ============================================================================

set -e

OSCAR_HOST="oscar"
OSCAR_BASE="~/wsindy-manifold/oscar_output"
LOCAL_BASE="./oscar_output"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# --- Experiments to download ---
EXPERIMENTS=(
    "DYN1_gentle_v2"
    "DYN2_hypervelocity_v2"
    "DYN3_hypernoisy_v2"
    "DYN4_blackhole_v2"
    "DYN5_supernova"
    "DYN6_varspeed_v2"
    "DYN7_pure_vicsek"
    "LST4_sqrt_simplex_align_h64_L2"
    "LST7_raw_none_align_h128_L2"
)

download_experiment() {
    local exp="$1"
    local remote="$OSCAR_HOST:$OSCAR_BASE/$exp"
    local local_dir="$LOCAL_BASE/$exp"

    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  $exp${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    mkdir -p "$local_dir"

    # 1) Root-level files
    echo -e "  ${GREEN}[1/6]${NC} Root files..."
    for f in config_used.yaml summary.json runtime_comparison.json; do
        rsync -az "$remote/$f" "$local_dir/$f" 2>/dev/null || true
    done

    # 2) rom_common/ (POD basis + mean) — skip latent_dataset.npz (large, not needed for vis)
    echo -e "  ${GREEN}[2/6]${NC} ROM common (POD basis + mean)..."
    mkdir -p "$local_dir/rom_common"
    for f in pod_basis.npz X_train_mean.npy shift_align.npz; do
        rsync -az "$remote/rom_common/$f" "$local_dir/rom_common/$f" 2>/dev/null || true
    done

    # 3) Model directories
    echo -e "  ${GREEN}[3/6]${NC} Model files (MVAR + LSTM)..."
    mkdir -p "$local_dir/MVAR" "$local_dir/LSTM"
    for f in mvar_model.npz test_results.csv runtime_profile.json; do
        rsync -az "$remote/MVAR/$f" "$local_dir/MVAR/$f" 2>/dev/null || true
    done
    for f in test_results.csv runtime_profile.json training_log.csv lstm_state_dict.pt; do
        rsync -az "$remote/LSTM/$f" "$local_dir/LSTM/$f" 2>/dev/null || true
    done

    # 4) Train metadata + first run density (for grid info only)
    echo -e "  ${GREEN}[4/6]${NC} Train metadata + grid info..."
    mkdir -p "$local_dir/train"
    rsync -az "$remote/train/metadata.json" "$local_dir/train/metadata.json" 2>/dev/null || true

    if [ -f "$local_dir/train/metadata.json" ]; then
        FIRST_RUN=$(python3 -c "import json; m=json.load(open('$local_dir/train/metadata.json')); print(m[0]['run_name'])" 2>/dev/null || echo "train_000")
    else
        FIRST_RUN="train_000"
    fi
    mkdir -p "$local_dir/train/$FIRST_RUN"
    rsync -az "$remote/train/$FIRST_RUN/density.npz" "$local_dir/train/$FIRST_RUN/density.npz" 2>/dev/null || true

    # 5) Test metadata
    echo -e "  ${GREEN}[5/6]${NC} Test metadata..."
    mkdir -p "$local_dir/test"
    for f in metadata.json index_mapping.csv test_results.csv; do
        rsync -az "$remote/test/$f" "$local_dir/test/$f" 2>/dev/null || true
    done

    # 6) Per-test files (density preds, truth, metrics, r2_vs_time, trajectory)
    echo -e "  ${GREEN}[6/6]${NC} Per-test data (density + metrics)..."
    # Use rsync include/exclude to get just what we need from test/test_*/
    rsync -avz --progress \
        --include='test_*/' \
        --include='test_*/density_true.npz' \
        --include='test_*/density_pred.npz' \
        --include='test_*/density_pred_*.npz' \
        --include='test_*/metrics_summary.json' \
        --include='test_*/metrics_summary_*.json' \
        --include='test_*/r2_vs_time.csv' \
        --include='test_*/r2_vs_time_*.csv' \
        --include='test_*/trajectory.npz' \
        --exclude='test_*/*' \
        --exclude='metadata.json' \
        --exclude='index_mapping.csv' \
        --exclude='test_results.csv' \
        "$remote/test/" "$local_dir/test/"

    # Verify
    local n_test=$(ls -d "$local_dir/test/test_"* 2>/dev/null | wc -l | tr -d ' ')
    local n_pred=$(find "$local_dir/test" -name "density_pred*.npz" 2>/dev/null | wc -l | tr -d ' ')
    local n_true=$(find "$local_dir/test" -name "density_true.npz" 2>/dev/null | wc -l | tr -d ' ')
    echo -e "  ${GREEN}✓${NC} $n_test test dirs | $n_pred pred files | $n_true truth files"

    # Print summary
    if [ -f "$local_dir/summary.json" ]; then
        python3 -c "
import json
s = json.load(open('$local_dir/summary.json'))
models = ', '.join(s.get('models_trained', []))
r2 = s.get('mean_r2_test', s.get('mean_r2_mvar', '?'))
print(f'  📋 R²={r2} | models: {models} | {s.get(\"n_train\",\"?\")} train, {s.get(\"n_test\",\"?\")} test')
" 2>/dev/null || true
    fi
}

# ============================================================================
# Main
# ============================================================================

echo "============================================================"
echo "  Downloading visualization data from Oscar"
echo "  ${#EXPERIMENTS[@]} experiments"
echo "============================================================"

for exp in "${EXPERIMENTS[@]}"; do
    download_experiment "$exp"
done

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Download complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Run visualizations with:"
for exp in "${EXPERIMENTS[@]}"; do
    echo "  python run_visualizations.py --experiment_name $exp"
done
