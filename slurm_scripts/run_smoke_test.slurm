#!/bin/bash
#SBATCH --job-name=smoke_test
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/smoke_test_%j.out
#SBATCH -e slurm_logs/smoke_test_%j.err

# ============================================================================
# SMOKE TEST — verify updated code produces valid R² on full pipeline
# ============================================================================
# Single regime (gas), minimal training data, MVAR + LSTM + WSINDy
# Expected: ~15-20 min.  1hr wall as safety margin.
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs oscar_output

CONFIG="configs/systematic/thesis_final/smoke_test.yaml"
EXP_NAME="smoke_test_code_verify"

echo "============================================"
echo "SMOKE TEST — full pipeline code verification"
echo "============================================"
echo "Config:     $CONFIG"
echo "Experiment: $EXP_NAME"
echo "Start time: $(date)"
echo "Node:       $(hostname)"
echo "Python:     $(which python)"
echo "Git HEAD:   $(git rev-parse --short HEAD)"
echo ""

python ROM_WSINDY_pipeline.py \
    --config "$CONFIG" \
    --experiment_name "$EXP_NAME"

EXIT_CODE=$?

echo ""
echo "============================================"
echo "End time:  $(date)"
echo "Exit code: $EXIT_CODE"

# Quick R² sanity check
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== R² SANITY CHECK ==="
    find oscar_output/${EXP_NAME}/ -name "*.json" | head -5 | while read f; do
        echo "--- $f ---"
        python -c "import json; d=json.load(open('$f')); [print(f'  {k}: {v}') for k,v in d.items() if 'r2' in k.lower()]" 2>/dev/null
    done
fi

exit $EXIT_CODE
