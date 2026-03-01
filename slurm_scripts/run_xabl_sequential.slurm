#!/bin/bash
#SBATCH --job-name=XABL_seq
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/XABL_seq_%j.out
#SBATCH -e slurm_logs/XABL_seq_%j.err

# ============================================================================
# EXTENDED ABLATION SUITE (XABL) — SEQUENTIAL
# ============================================================================
# Runs all 8 XABL experiments sequentially to avoid disk quota issues.
# After each experiment completes, its training data is removed (~1GB each).
# Expected runtime: 8 × ~45min = ~6 hours
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

export PYTHONPATH=src

declare -a CONFIGS=(
    "configs/XABL1_align_raw_none_noScale.yaml"
    "configs/XABL2_align_raw_none_scale1.yaml"
    "configs/XABL3_align_raw_simplex_noScale.yaml"
    "configs/XABL4_align_raw_simplex_scale1.yaml"
    "configs/XABL5_align_sqrt_none_noScale.yaml"
    "configs/XABL6_align_sqrt_none_scale1.yaml"
    "configs/XABL7_align_sqrt_simplex_noScale.yaml"
    "configs/XABL8_align_sqrt_simplex_scale1.yaml"
)

declare -a EXPNAMES=(
    "XABL1_align_raw_none_noScale"
    "XABL2_align_raw_none_scale1"
    "XABL3_align_raw_simplex_noScale"
    "XABL4_align_raw_simplex_scale1"
    "XABL5_align_sqrt_none_noScale"
    "XABL6_align_sqrt_none_scale1"
    "XABL7_align_sqrt_simplex_noScale"
    "XABL8_align_sqrt_simplex_scale1"
)

declare -a LABELS=(
    "align / raw / none / no scaling"
    "align / raw / none / scale→1.0"
    "align / raw / simplex / no scaling"
    "align / raw / simplex / scale→1.0"
    "align / sqrt / none / no scaling"
    "align / sqrt / none / scale→1.0"
    "align / sqrt / simplex / no scaling"
    "align / sqrt / simplex / scale→1.0"
)

N_SUCCESS=0
N_FAIL=0
declare -a RESULTS=()

echo "============================================================"
echo "EXTENDED ABLATION SUITE — SEQUENTIAL (8 experiments)"
echo "============================================================"
echo "Start time: $(date)"
echo "Node:       $(hostname)"
echo "CPUs:       $SLURM_NTASKS"
echo ""

for IDX in 0 1 2 3 4 5 6 7; do
    XABL_NUM=$((IDX + 1))
    CONFIG="${CONFIGS[$IDX]}"
    EXP_NAME="${EXPNAMES[$IDX]}"
    LABEL="${LABELS[$IDX]}"
    
    echo ""
    echo "============================================================"
    echo "[$XABL_NUM/8] XABL${XABL_NUM} — $LABEL"
    echo "============================================================"
    echo "Config:     $CONFIG"
    echo "Experiment: $EXP_NAME"
    echo "Start:      $(date)"
    echo ""
    
    # Run unified ROM pipeline
    python run_unified_rom_pipeline.py \
        --config "$CONFIG" \
        --experiment_name "$EXP_NAME"
    
    PIPE_EXIT=$?
    
    if [ $PIPE_EXIT -eq 0 ] && [ -f "oscar_output/$EXP_NAME/summary.json" ]; then
        # Extract R² from summary
        R2=$(python3 -c "
import json
d = json.load(open('oscar_output/$EXP_NAME/summary.json'))
m = d.get('mvar', {})
r2 = m.get('mean_r2_test', -999)
print(f'{r2:.4f}')
" 2>/dev/null || echo "?")
        
        echo "✓ XABL${XABL_NUM} PASSED — R²=$R2"
        RESULTS+=("XABL${XABL_NUM}: R²=$R2 ($LABEL)")
        N_SUCCESS=$((N_SUCCESS + 1))
    else
        echo "✗ XABL${XABL_NUM} FAILED (exit=$PIPE_EXIT)"
        RESULTS+=("XABL${XABL_NUM}: FAILED ($LABEL)")
        N_FAIL=$((N_FAIL + 1))
    fi
    
    # Clean training data to free disk space (~1GB each)
    # Keep train/metadata.json for visualization but remove bulk density files
    if [ -d "oscar_output/$EXP_NAME/train" ]; then
        echo "   Cleaning training data for $EXP_NAME..."
        # Keep metadata but remove density dirs
        find "oscar_output/$EXP_NAME/train" -name "train_*" -type d -exec rm -rf {} + 2>/dev/null
        echo "   ✓ Training data cleaned"
    fi
    
    echo "End:   $(date)"
    echo ""
done

# ============================================================
# FINAL SUMMARY
# ============================================================
echo ""
echo "============================================================"
echo "XABL SUITE COMPLETE"
echo "============================================================"
echo "Passed: $N_SUCCESS / 8"
echo "Failed: $N_FAIL / 8"
echo ""
echo "Results:"
for RES in "${RESULTS[@]}"; do
    echo "  $RES"
done
echo ""
echo "End time: $(date)"
echo "============================================================"
