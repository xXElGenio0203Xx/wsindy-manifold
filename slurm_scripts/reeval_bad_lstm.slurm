#!/bin/bash
#SBATCH --job-name=reeval_bad
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/reeval_bad_%j.out
#SBATCH --error=slurm_logs/reeval_bad_%j.err

source ~/wsindy_env_new/bin/activate
cd ~/wsindy-manifold
export PYTHONPATH="$PWD/src:$PYTHONPATH"

ODIR="/users/emaciaso/scratch/oscar_output"
BAD_EXPS=(
    "NDYN04_gas_thesis_final"
    "NDYN04_gas_VS_thesis_final"
    "NDYN05_blackhole_VS_thesis_final"
    "NDYN06_supernova_thesis_final"
)

echo "============================================"
echo "Re-evaluating BAD LSTM experiments"
echo "Testing mass_postprocess = {scale, none, simplex}"
echo "Start time: $(date)"
echo "============================================"

# Step 1: Backup original metrics_summary_lstm.json for each experiment
echo ""
echo "=== BACKING UP ORIGINALS ==="
for exp in "${BAD_EXPS[@]}"; do
    for tid in test_000 test_001 test_002 test_003; do
        src="$ODIR/$exp/test/$tid/metrics_summary_lstm.json"
        bak="$ODIR/$exp/test/$tid/metrics_summary_lstm_ORIG.json"
        if [ -f "$src" ] && [ ! -f "$bak" ]; then
            cp "$src" "$bak"
            echo "  Backed up: $exp/$tid"
        fi
    done
done

# Step 2: Evaluate each mass_pp and record results
echo ""
echo "=== COMPARISON TABLE ==="
echo "experiment,mass_pp,test_000,test_001,test_002,test_003,mean_r2"

for MPP in scale none simplex; do
    for exp in "${BAD_EXPS[@]}"; do
        echo ""
        echo "--- $exp with mass_pp=$MPP ---"
        python rerun_evaluation.py "$ODIR/$exp" --lstm-only --mass-postprocess "$MPP" 2>&1 || echo "  FAILED: $exp"

        # Extract per-test R² from just-written metrics files
        r2vals=""
        for tid in test_000 test_001 test_002 test_003; do
            r2=$(python3 -c "
import json
with open('$ODIR/$exp/test/$tid/metrics_summary_lstm.json') as f:
    d = json.load(f)
print(f'{d.get(\"r2_recon\", -999):.4f}')
" 2>/dev/null || echo "-999")
            r2vals="$r2vals,$r2"
        done
        mean=$(python3 -c "
vals = [float(x) for x in '$r2vals'.strip(',').split(',') if x != '-999']
print(f'{sum(vals)/len(vals):.4f}' if vals else 'N/A')
" 2>/dev/null)
        echo "CSV: $exp,$MPP${r2vals},$mean"
    done
done

# Step 3: Restore originals (leave scale as final state)
echo ""
echo "=== RESTORING ORIGINALS (scale) ==="
for exp in "${BAD_EXPS[@]}"; do
    python rerun_evaluation.py "$ODIR/$exp" --lstm-only --mass-postprocess scale 2>&1 || true
done

echo ""
echo "End time: $(date)"
echo "Exit code: $?"
