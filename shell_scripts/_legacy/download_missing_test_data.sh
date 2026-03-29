#!/bin/bash
# Download missing test density files from Oscar
# Usage: bash download_missing_test_data.sh
# Requires: SSH to oscar configured and authenticated (will prompt for password+Duo)

set -e
REMOTE="oscar:~/wsindy-manifold/oscar_output"
LOCAL="oscar_output"

download_missing() {
    local exp="$1"
    local start="$2"
    local end="$3"
    
    echo "=== $exp (test_${start}..test_${end}) ==="
    for i in $(seq -w "$start" "$end"); do
        local dir="${exp}/test/test_0${i}"
        local f="${LOCAL}/${dir}/density_true.npz"
        
        # Skip if already complete (check size > 100MB)
        if [[ -f "$f" ]]; then
            size=$(stat -f%z "$f" 2>/dev/null || stat --format=%s "$f" 2>/dev/null)
            if (( size > 100000000 )); then
                echo "  test_0${i}: already complete (${size} bytes), skipping"
                continue
            fi
        fi
        
        mkdir -p "${LOCAL}/${dir}"
        echo -n "  test_0${i}: downloading... "
        
        # Download all files for incomplete test runs
        for fname in density_true.npz density_pred.npz density_pred_mvar.npz metrics_summary.json r2_vs_time.csv trajectory.npz; do
            local lf="${LOCAL}/${dir}/${fname}"
            if [[ ! -f "$lf" ]]; then
                scp -q "${REMOTE}/${dir}/${fname}" "$lf" 2>/dev/null || true
            fi
        done
        echo "OK"
    done
}

echo "Downloading missing test data from Oscar..."
echo "Each density file is ~147MB. Total ~3.5GB."
echo ""

# DYN7: test_007..019 (test_012 may be truncated)
download_missing "DYN7_pure_vicsek" 7 19

# LST4: test_014..019
download_missing "LST4_sqrt_simplex_align_h64_L2" 14 19

# LST7: test_009..019
download_missing "LST7_raw_none_align_h128_L2" 9 19

echo ""
echo "Download complete! Re-run visualizations to include all test runs."
