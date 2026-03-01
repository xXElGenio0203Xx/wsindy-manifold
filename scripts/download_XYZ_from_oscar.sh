#!/usr/bin/env bash
# Download Suite X/Y/Z results from OSCAR
# Usage: bash scripts/download_XYZ_from_oscar.sh

set -euo pipefail

OSCAR_USER="emaciaso"
OSCAR_HOST="ssh.ccv.brown.edu"
REMOTE_BASE="/oscar/home/${OSCAR_USER}/wsindy-manifold/oscar_output"
LOCAL_BASE="oscar_output"

mkdir -p "$LOCAL_BASE"

echo "=========================================="
echo "Downloading Suite X/Y/Z from OSCAR"
echo "=========================================="

# Suite X experiments (12)
SUITE_X=(
    X1_V1_raw_H37
    X2_V1_sqrtSimplex_H37
    X3_V1_raw_H162
    X4_V1_sqrtSimplex_H162
    X5_V33_raw_H37
    X6_V33_sqrtSimplex_H37
    X7_V33_raw_H162
    X8_V33_sqrtSimplex_H162
    X9_V34_raw_H37
    X10_V34_sqrtSimplex_H37
    X11_V34_raw_H162
    X12_V34_sqrtSimplex_H162
)

# Suite Y experiments (6)
SUITE_Y=(
    Y1_V1_raw_H100
    Y2_V1_sqrtSimplex_H100
    Y3_V33_raw_H100
    Y4_V33_sqrtSimplex_H100
    Y5_V34_raw_H100
    Y6_V34_sqrtSimplex_H100
)

# Suite Z experiments (6)
SUITE_Z=(
    Z1_shiftAlign_V1_raw_H162
    Z2_shiftAlign_V1_sqrtsimplex_H162
    Z3_shiftAlign_V33_raw_H162
    Z4_shiftAlign_V33_sqrtsimplex_H162
    Z5_shiftAlign_V34_raw_H162
    Z6_shiftAlign_V34_sqrtsimplex_H162
)

download_experiment() {
    local name="$1"
    local remote_dir="${REMOTE_BASE}/${name}"
    local local_dir="${LOCAL_BASE}/${name}"
    
    echo -n "  ${name}: "
    
    # Only download key files (not full simulation data)
    mkdir -p "${local_dir}/test"
    mkdir -p "${local_dir}/MVAR"
    
    # test_results.csv (per-run metrics)
    scp -q "${OSCAR_USER}@${OSCAR_HOST}:${remote_dir}/test/test_results.csv" \
        "${local_dir}/test/" 2>/dev/null && echo -n "csv " || echo -n "NO-csv "
    
    # summary.json (pipeline-level metrics)
    scp -q "${OSCAR_USER}@${OSCAR_HOST}:${remote_dir}/summary.json" \
        "${local_dir}/" 2>/dev/null && echo -n "summary " || echo -n "NO-summary "
    
    # MVAR test_results.csv (backup copy)
    scp -q "${OSCAR_USER}@${OSCAR_HOST}:${remote_dir}/MVAR/test_results.csv" \
        "${local_dir}/MVAR/" 2>/dev/null || true
    
    echo "✓"
}

download_z_experiment() {
    local name="$1"
    local remote_dir="${REMOTE_BASE}/${name}"
    local local_dir="${LOCAL_BASE}/${name}"
    
    echo -n "  ${name}: "
    mkdir -p "${local_dir}"
    
    # shift_aligned_summary.json
    scp -q "${OSCAR_USER}@${OSCAR_HOST}:${remote_dir}/shift_aligned_summary.json" \
        "${local_dir}/" 2>/dev/null && echo -n "summary " || echo -n "NO-summary "
    
    # shift_aligned_per_run.json
    scp -q "${OSCAR_USER}@${OSCAR_HOST}:${remote_dir}/shift_aligned_per_run.json" \
        "${local_dir}/" 2>/dev/null && echo -n "per-run " || echo -n "NO-per-run "
    
    # drift_timeseries.csv
    scp -q "${OSCAR_USER}@${OSCAR_HOST}:${remote_dir}/drift_timeseries.csv" \
        "${local_dir}/" 2>/dev/null && echo -n "drift " || echo -n "NO-drift "
    
    echo "✓"
}

echo ""
echo "Suite X (12 experiments):"
for exp in "${SUITE_X[@]}"; do
    download_experiment "$exp"
done

echo ""
echo "Suite Y (6 experiments):"
for exp in "${SUITE_Y[@]}"; do
    download_experiment "$exp"
done

echo ""
echo "Suite Z (6 shift-aligned evals):"
for exp in "${SUITE_Z[@]}"; do
    download_z_experiment "$exp"
done

echo ""
echo "=========================================="
echo "Download complete!"
echo "Now run:  python scripts/collect_XYZ_results.py"
echo "=========================================="
