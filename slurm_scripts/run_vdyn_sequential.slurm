#!/bin/bash
#SBATCH --job-name=VDYN_seq
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -o slurm_logs/VDYN_seq_%j.out
#SBATCH -e slurm_logs/VDYN_seq_%j.err

# ============================================================================
# VDYN SUITE — Variable Speed Dynamics (7 experiments, sequential)
# ============================================================================
# All DYN variants re-run with speed_mode="variable":
#   raw density | alignment ON | simplex OFF | MVAR + LSTM
#
# Expected runtime: 7 × ~80min = ~9 hours
# GPU requested for LSTM training.
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

export PYTHONPATH=src

declare -a CONFIGS=(
    "configs/VDYN1_gentle_varspeed.yaml"
    "configs/VDYN2_hypervelocity_varspeed.yaml"
    "configs/VDYN3_hypernoisy_varspeed.yaml"
    "configs/VDYN4_blackhole_varspeed.yaml"
    "configs/VDYN5_supernova_varspeed.yaml"
    "configs/VDYN6_baseline_varspeed.yaml"
    "configs/VDYN7_pure_vicsek_varspeed.yaml"
)

declare -a EXPNAMES=(
    "VDYN1_gentle_varspeed"
    "VDYN2_hypervelocity_varspeed"
    "VDYN3_hypernoisy_varspeed"
    "VDYN4_blackhole_varspeed"
    "VDYN5_supernova_varspeed"
    "VDYN6_baseline_varspeed"
    "VDYN7_pure_vicsek_varspeed"
)

declare -a LABELS=(
    "gentle (speed=1.5, eta=0.1, Ca=0.8/Cr=0.3)"
    "hypervelocity (speed=15)"
    "hypernoisy (eta=1.5)"
    "blackhole (Ca=12/Cr=0.1)"
    "supernova (Ca=0.1/Cr=10)"
    "baseline (standard forces)"
    "pure vicsek (no forces)"
)

N_TOTAL=${#CONFIGS[@]}
N_SUCCESS=0
N_FAIL=0
declare -a RESULTS=()

echo "============================================================"
echo "VDYN SUITE — Variable Speed Dynamics ($N_TOTAL experiments)"
echo "============================================================"
echo "Start time: $(date)"
echo "Node:       $(hostname)"
echo "CPUs:       $SLURM_NTASKS"
echo "GPU:        $CUDA_VISIBLE_DEVICES"
echo ""
echo "Pipeline: raw | align | no simplex | MVAR + LSTM"
echo ""

for IDX in $(seq 0 $((N_TOTAL - 1))); do
    NUM=$((IDX + 1))
    CONFIG="${CONFIGS[$IDX]}"
    EXP_NAME="${EXPNAMES[$IDX]}"
    LABEL="${LABELS[$IDX]}"

    echo ""
    echo "============================================================"
    echo "[$NUM/$N_TOTAL] VDYN${NUM} — $LABEL"
    echo "============================================================"
    echo "Config:     $CONFIG"
    echo "Experiment: $EXP_NAME"
    echo "Start:      $(date)"
    echo ""

    # Run ROM pipeline (MVAR + LSTM)
    python ROM_pipeline.py \
        --config "$CONFIG" \
        --experiment_name "$EXP_NAME"

    PIPE_EXIT=$?

    if [ $PIPE_EXIT -eq 0 ] && [ -f "oscar_output/$EXP_NAME/summary.json" ]; then
        # Extract R² from summary
        R2_INFO=$(python3 -c "
import json
d = json.load(open('oscar_output/$EXP_NAME/summary.json'))
m = d.get('mvar', {})
l = d.get('lstm', {})
mr2 = m.get('mean_r2_test', -999)
lr2 = l.get('mean_r2_test', -999)
print(f'MVAR={mr2:.4f}  LSTM={lr2:.4f}')
" 2>/dev/null || echo "?")

        echo "✓ VDYN${NUM} PASSED — $R2_INFO"
        RESULTS+=("VDYN${NUM}: $R2_INFO ($LABEL)")
        N_SUCCESS=$((N_SUCCESS + 1))
    else
        echo "✗ VDYN${NUM} FAILED (exit=$PIPE_EXIT)"
        RESULTS+=("VDYN${NUM}: FAILED ($LABEL)")
        N_FAIL=$((N_FAIL + 1))
    fi

    # Clean training data to free disk space (~1GB each)
    if [ -d "oscar_output/$EXP_NAME/train" ]; then
        echo "   Cleaning training data for $EXP_NAME..."
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
echo "VDYN SUITE COMPLETE"
echo "============================================================"
echo "Passed: $N_SUCCESS / $N_TOTAL"
echo "Failed: $N_FAIL / $N_TOTAL"
echo ""
echo "Results:"
for RES in "${RESULTS[@]}"; do
    echo "  $RES"
done
echo ""
echo "End time: $(date)"
echo "============================================================"
