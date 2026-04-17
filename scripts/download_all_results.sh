#!/bin/bash
# Download all available OSCAR results for thesis tables
# Run from: /Users/maria_1/Desktop/wsindy-manifold
set -e

REMOTE="oscar"
REMOTE_BASE="/users/emaciaso/scratch/oscar_output"
LOCAL_BASE="oscar_output/results_9apr"

mkdir -p "$LOCAL_BASE"

# Collect all experiments with test_results
EXPERIMENTS=(
    NDYN04_gas_thesis_final
    NDYN04_gas_VS_thesis_final
    NDYN05_blackhole_thesis_final
    NDYN05_blackhole_VS_thesis_final
    NDYN06_supernova_thesis_final
    NDYN06_supernova_VS_thesis_final
    NDYN08_pure_vicsek_thesis_final
    NDYN04_gas_tier1_w5
    NDYN04_gas_tier1_bic
    NDYN04_gas_VS_tier1_w5
    NDYN04_gas_VS_tier1_bic
    NDYN05_blackhole_tier1_w5
    NDYN05_blackhole_tier1_bic
    NDYN05_blackhole_VS_tier1_w5
    NDYN05_blackhole_VS_tier1_bic
    NDYN06_supernova_tier1_w5
    NDYN06_supernova_tier1_bic
    NDYN06_supernova_VS_tier1_w5
    NDYN06_supernova_VS_tier1_bic
    NDYN07_crystal_tier1_w5
    NDYN07_crystal_tier1_bic
    NDYN07_crystal_VS_tier1_w5
    NDYN07_crystal_VS_tier1_bic
    NDYN08_pure_vicsek_tier1_w5
    NDYN08_pure_vicsek_tier1_bic
)

for exp in "${EXPERIMENTS[@]}"; do
    echo "=== Downloading $exp ==="
    mkdir -p "$LOCAL_BASE/$exp/test"

    # test_results.csv
    scp "$REMOTE:$REMOTE_BASE/$exp/test/test_results.csv" \
        "$LOCAL_BASE/$exp/test/" 2>/dev/null || echo "  no test_results.csv"

    # metadata
    scp "$REMOTE:$REMOTE_BASE/$exp/test/metadata.json" \
        "$LOCAL_BASE/$exp/test/" 2>/dev/null || echo "  no metadata.json"

    # Per-test-case metrics (MVAR and LSTM)
    for tid in test_000 test_001 test_002 test_003; do
        mkdir -p "$LOCAL_BASE/$exp/test/$tid"
        scp "$REMOTE:$REMOTE_BASE/$exp/test/$tid/metrics_summary_mvar.json" \
            "$LOCAL_BASE/$exp/test/$tid/" 2>/dev/null || true
        scp "$REMOTE:$REMOTE_BASE/$exp/test/$tid/metrics_summary_lstm.json" \
            "$LOCAL_BASE/$exp/test/$tid/" 2>/dev/null || true
        scp "$REMOTE:$REMOTE_BASE/$exp/test/$tid/r2_vs_time_mvar.csv" \
            "$LOCAL_BASE/$exp/test/$tid/" 2>/dev/null || true
        scp "$REMOTE:$REMOTE_BASE/$exp/test/$tid/r2_vs_time_lstm.csv" \
            "$LOCAL_BASE/$exp/test/$tid/" 2>/dev/null || true
    done

    # WSINDy model if available
    mkdir -p "$LOCAL_BASE/$exp/WSINDy"
    scp "$REMOTE:$REMOTE_BASE/$exp/WSINDy/multifield_model.json" \
        "$LOCAL_BASE/$exp/WSINDy/" 2>/dev/null || true

    echo "  Done"
done

# Also get WSINDy v3 models (separate experiments)
for exp in NDYN04_gas_wsindy_v3 NDYN05_blackhole_wsindy_v3 NDYN05_blackhole_VS_wsindy_v3 NDYN06_supernova_VS_wsindy_v3 NDYN08_pure_vicsek_wsindy_v3; do
    echo "=== WSINDy v3: $exp ==="
    mkdir -p "$LOCAL_BASE/$exp/WSINDy"
    scp "$REMOTE:$REMOTE_BASE/$exp/WSINDy/multifield_model.json" \
        "$LOCAL_BASE/$exp/WSINDy/" 2>/dev/null || echo "  no model"
done

echo ""
echo "=== Download complete ==="
find "$LOCAL_BASE" -name "*.json" -o -name "*.csv" | wc -l
echo "files downloaded"
