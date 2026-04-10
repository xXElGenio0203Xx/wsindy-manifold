#!/bin/bash
# ============================================================================
# check_and_download_tier1.sh
# ============================================================================
# Check tier1 job status, download any new results, trigger WSINDy fallback.
# Uses a SINGLE SSH call to check all experiments (fast).
#
# Usage:
#   bash scripts/check_and_download_tier1.sh          # check only
#   bash scripts/check_and_download_tier1.sh --submit  # auto-submit fallback
# ============================================================================
set -euo pipefail

REMOTE="oscar"
REMOTE_BASE="~/wsindy-manifold"
LOCAL_RESULTS="oscar_output/results_tier1"
AUTO_SUBMIT=false
[[ "${1:-}" == "--submit" ]] && AUTO_SUBMIT=true

mkdir -p "$LOCAL_RESULTS"

echo "========================================"
echo "  Tier 1 Status Check — $(date)"
echo "========================================"

# ── 1. Gather ALL status in a single SSH call ──
STATUS=$(ssh $REMOTE 'bash -s' << 'REMOTE_SCRIPT'
cd ~/wsindy-manifold

# Queue
echo "QUEUE_START"
squeue -u emaciaso --format="%.12i %.6t %.12M %.20j %.10l" 2>/dev/null | grep -E "JOBID|tier1" || echo "NONE"
echo "QUEUE_END"

# Experiment status
EXPS=(
    NDYN04_gas_tier1_w5 NDYN04_gas_tier1_bic
    NDYN04_gas_VS_tier1_w5 NDYN04_gas_VS_tier1_bic
    NDYN05_blackhole_tier1_w5 NDYN05_blackhole_tier1_bic
    NDYN05_blackhole_VS_tier1_w5 NDYN05_blackhole_VS_tier1_bic
    NDYN06_supernova_tier1_w5 NDYN06_supernova_tier1_bic
    NDYN06_supernova_VS_tier1_w5 NDYN06_supernova_VS_tier1_bic
    NDYN07_crystal_tier1_w5 NDYN07_crystal_tier1_bic
    NDYN07_crystal_VS_tier1_w5 NDYN07_crystal_VS_tier1_bic
    NDYN08_pure_vicsek_tier1_w5 NDYN08_pure_vicsek_tier1_bic
)

echo "STATUS_START"
for exp in "${EXPS[@]}"; do
    d="oscar_output/${exp}"
    mvar="NO"; lstm="NO"; wsindy="NO"; train="NO"; wv3="NO"
    [ -f "$d/MVAR/test_results.csv" ] && mvar="OK"
    [ -f "$d/LSTM/test_results.csv" ] && lstm="OK"
    [ -f "$d/WSINDy/multifield_model.json" ] && wsindy="OK"
    [ -d "$d/train" ] && [ -f "$d/train/metadata.json" ] && train="OK"
    [ -f "$d/wsindy_v3/multifield_model.json" ] && wv3="OK"
    echo "$exp $mvar $lstm $wsindy $train $wv3"
done
echo "STATUS_END"
REMOTE_SCRIPT
)

# ── 2. Parse queue ──
echo ""
echo "--- SLURM Queue ---"
echo "$STATUS" | sed -n '/QUEUE_START/,/QUEUE_END/p' | grep -v '_START\|_END'

# ── 3. Parse experiment status ──
echo ""
echo "--- Experiment Status ---"
printf "  %-40s  %-5s %-5s %-7s %-6s %-6s\n" "Experiment" "MVAR" "LSTM" "WSINDy" "Train" "Wv3"
printf "  %-40s  %-5s %-5s %-7s %-6s %-6s\n" "──────────" "────" "────" "──────" "─────" "───"

MVAR_DONE=0; LSTM_DONE=0; WSINDY_DONE=0
NEED_WSINDY=()
WV3_DONE=0

while IFS=' ' read -r exp mvar lstm wsindy train wv3; do
    printf "  %-40s  %-5s %-5s %-7s %-6s %-6s\n" "$exp" "$mvar" "$lstm" "$wsindy" "$train" "$wv3"
    [[ "$mvar" == "OK" ]] && ((MVAR_DONE++)) || true
    [[ "$lstm" == "OK" ]] && ((LSTM_DONE++)) || true
    [[ "$wsindy" == "OK" ]] && ((WSINDY_DONE++)) || true
    [[ "$wv3" == "OK" ]] && ((WV3_DONE++)) || true
    # Need WSINDy fallback: has training data but no WSINDy result
    if [[ "$wsindy" == "NO" && "$train" == "OK" ]]; then
        NEED_WSINDY+=("$exp")
    fi
done < <(echo "$STATUS" | sed -n '/STATUS_START/,/STATUS_END/p' | grep -v '_START\|_END')

echo ""
echo "--- Summary ---"
echo "  MVAR complete:     ${MVAR_DONE}/18"
echo "  LSTM complete:     ${LSTM_DONE}/18"
echo "  WSINDy complete:   ${WSINDY_DONE}/18"
echo "  WSINDy v3 (old):   ${WV3_DONE}/18"
echo "  Need WSINDy fallback: ${#NEED_WSINDY[@]}"
if [ ${#NEED_WSINDY[@]} -gt 0 ]; then
    echo "  Experiments: ${NEED_WSINDY[*]}"
fi

# ── 4. Download new results ──
echo ""
echo "--- Downloading results ---"
DOWNLOADED=0
for exp_line in $(echo "$STATUS" | sed -n '/STATUS_START/,/STATUS_END/p' | grep -v '_START\|_END'); do
    exp=$(echo "$exp_line" | awk '{print $1}')
    local_dir="${LOCAL_RESULTS}/${exp}"
    mkdir -p "$local_dir"

    for f in MVAR/test_results.csv LSTM/test_results.csv WSINDy/multifield_model.json wsindy_v3/multifield_model.json; do
        local_file="${local_dir}/$(echo $f | tr '/' '_')"
        if [ ! -f "$local_file" ]; then
            scp "${REMOTE}:${REMOTE_BASE}/oscar_output/${exp}/${f}" "$local_file" 2>/dev/null && {
                echo "  Downloaded: ${exp}/${f}"
                ((DOWNLOADED++)) || true
            } || true
        fi
    done
done
echo "  New files: ${DOWNLOADED}"

# ── 5. Action items / auto-submit ──
echo ""
echo "--- Action Items ---"
TIER1_RUNNING=$(echo "$STATUS" | sed -n '/QUEUE_START/,/QUEUE_END/p' | grep -c ' R ' || true)

if [[ "${#NEED_WSINDY[@]}" -gt 0 && "$TIER1_RUNNING" == "0" ]]; then
    echo "  *** No tier1 tasks running and ${#NEED_WSINDY[@]} experiments need WSINDy!"
    if $AUTO_SUBMIT; then
        echo "  Auto-submitting WSINDy fallback..."
        ssh $REMOTE "cd ~/wsindy-manifold && sbatch slurm_scripts/run_tier1_wsindy_noboot.slurm" 2>/dev/null
        echo "  Fallback submitted!"
    else
        echo "  Run with --submit to auto-submit, or manually:"
        echo "    ssh oscar 'cd ~/wsindy-manifold && sbatch slurm_scripts/run_tier1_wsindy_noboot.slurm'"
    fi
elif [[ "${#NEED_WSINDY[@]}" -gt 0 ]]; then
    echo "  ${TIER1_RUNNING} tier1 tasks still running. Wait for timeout, then submit fallback."
    echo "  Re-run: bash scripts/check_and_download_tier1.sh"
else
    echo "  All experiments with data have WSINDy results. Nothing to do."
fi

echo ""
echo "Done at $(date)"
