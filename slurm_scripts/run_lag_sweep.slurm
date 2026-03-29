#!/bin/bash
#SBATCH --job-name=lag_sweep
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/lag_sweep_%j.out
#SBATCH -e slurm_logs/lag_sweep_%j.err

# ============================================================================
# LSTM LAG SWEEP: CS01 Regime
# ============================================================================
# Sweep LSTM lag = {5, 10, 15, 20, 25, 30, 40} with MVAR fixed at lag=5.
# BIC optimal for CS01 = 23, HQIC = 37, AIC = 50.
# Purpose: Find empirical LSTM optimum and validate against BIC/AIC theory.
#
# The patched pipeline builds SEPARATE datasets for MVAR and LSTM when
# their lags differ, fixing the shared-dataset bug.
#
# Expected runtime: ~15-25 min per config × 7 = ~2-3 hours total
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs oscar_output

echo "============================================"
echo "LSTM LAG SWEEP — CS01 Regime"
echo "============================================"
echo "Start time:  $(date)"
echo "Node:        $(hostname)"
echo "Python:      $(which python)"
echo ""

# Pipeline ROM_WSINDY_pipeline.py has built-in separate-lag support
# (builds separate LSTM dataset when LSTM lag != MVAR lag)
echo ""

# Copy lag sweep configs to Oscar if not already there
if [ ! -d configs/lag_sweep ]; then
    echo "ERROR: configs/lag_sweep/ not found. Sync configs first."
    exit 1
fi

LSTM_LAGS=(5 10 15 20 25 30 40)
TOTAL=${#LSTM_LAGS[@]}
SUCCEEDED=0
FAILED=0

for i in "${!LSTM_LAGS[@]}"; do
    LSTM_LAG=${LSTM_LAGS[$i]}
    EXP_NAME="LSWEEP_CS01_mvar5_lstm${LSTM_LAG}"
    CONFIG="configs/lag_sweep/${EXP_NAME}.yaml"
    IDX=$((i + 1))
    
    echo "============================================"
    echo "[$IDX/$TOTAL] LSTM lag=${LSTM_LAG}"
    echo "============================================"
    echo "Config:      $CONFIG"
    echo "Experiment:  $EXP_NAME"
    echo "Start:       $(date)"
    echo ""
    
    if [ ! -f "$CONFIG" ]; then
        echo "ERROR: Config file not found: $CONFIG"
        FAILED=$((FAILED + 1))
        continue
    fi
    
    # Skip if already completed (summary.json exists)
    if [ -f "oscar_output/${EXP_NAME}/summary.json" ]; then
        echo "SKIPPING: Already completed (summary.json exists)"
        SUCCEEDED=$((SUCCEEDED + 1))
        echo ""
        continue
    fi
    
    python ROM_WSINDY_pipeline.py \
        --config "$CONFIG" \
        --experiment_name "$EXP_NAME"
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "  ✓ Completed successfully"
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        echo "  ✗ FAILED (exit code $EXIT_CODE)"
        FAILED=$((FAILED + 1))
    fi
    echo "Finished:    $(date)"
    echo ""
done

echo "============================================"
echo "LAG SWEEP SUMMARY"
echo "============================================"
echo "Total:     $TOTAL"
echo "Succeeded: $SUCCEEDED"
echo "Failed:    $FAILED"
echo "End time:  $(date)"
echo ""

# Print quick results summary
echo "============================================"
echo "QUICK RESULTS"
echo "============================================"
python3 -c "
import json, os, glob
print(f'{'LSTM Lag':>10} {'MVAR R²':>12} {'LSTM R²':>12} {'LSTM ring R²':>14}')
print('-' * 55)
for lag in [5, 10, 15, 20, 25, 30, 40]:
    d = f'oscar_output/LSWEEP_CS01_mvar5_lstm{lag}'
    sf = os.path.join(d, 'summary.json')
    if os.path.exists(sf):
        s = json.load(open(sf))
        mvar = s.get('mvar', {})
        lstm = s.get('lstm', {})
        mr2 = mvar.get('mean_r2_test', '?')
        lr2 = lstm.get('mean_r2_test', '?')
        # Try to get per-test ring R²
        ring_r2 = '?'
        import csv
        csv_path = os.path.join(d, 'LSTM', 'test_results.csv')
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    tid = row.get('test_id', '')
                    if tid == '2':  # ring is test_002
                        ring_r2 = float(row.get('r2_reconstructed', 'nan'))
        mr2_s = f'{mr2:.4f}' if isinstance(mr2, float) else str(mr2)
        lr2_s = f'{lr2:.4f}' if isinstance(lr2, float) else str(lr2)
        rr2_s = f'{ring_r2:.4f}' if isinstance(ring_r2, float) else str(ring_r2)
        print(f'{lag:>10} {mr2_s:>12} {lr2_s:>12} {rr2_s:>14}')
    else:
        print(f'{lag:>10} {'(not run)':>12}')
" 2>/dev/null || echo "(Python summary failed)"

exit 0
