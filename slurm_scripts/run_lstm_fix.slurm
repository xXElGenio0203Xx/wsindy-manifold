#!/bin/bash
#SBATCH --job-name=LSTM_FIX
#SBATCH --output=slurm_logs/lstm_fix_%j.out
#SBATCH --error=slurm_logs/lstm_fix_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

# LFIX experiments: LST4 LSTM architecture + sqrt/simplex on 6 failing regimes
# Each config takes ~5-6h, total ~30-36h

set -e
module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate
cd ~/wsindy-manifold

CONFIGS=(
    "configs/lstm_fix/LFIX_CS01.yaml"
    "configs/lstm_fix/LFIX_DR01.yaml"
    "configs/lstm_fix/LFIX_EC02.yaml"
    "configs/lstm_fix/LFIX_ES01.yaml"
    "configs/lstm_fix/LFIX_DM01.yaml"
    "configs/lstm_fix/LFIX_SM01.yaml"
)

echo "=========================================="
echo "LSTM FIX EXPERIMENT — LST4 arch + sqrt/simplex"
echo "Start: $(date)"
echo "=========================================="

for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    outdir="oscar_output/${name}"

    # Skip if already completed
    if [ -f "${outdir}/LSTM/test_results.csv" ] && [ -f "${outdir}/summary.json" ]; then
        echo ""
        echo ">>> SKIPPING ${name} — already completed"
        continue
    fi

    echo ""
    echo "=========================================="
    echo ">>> Running ${name}"
    echo ">>> Config: ${cfg}"
    echo ">>> Start: $(date)"
    echo "=========================================="

    python ROM_WSINDY_pipeline.py --config "$cfg" --experiment_name "$name"
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo ">>> FAILED ${name} with exit code ${exit_code}"
        continue
    fi

    echo ""
    echo ">>> COMPLETED ${name}"
    echo ">>> MVAR results:"
    cat "${outdir}/MVAR/test_results.csv" 2>/dev/null | head -5
    echo ""
    echo ">>> LSTM results:"
    cat "${outdir}/LSTM/test_results.csv" 2>/dev/null | head -5
    echo ""
done

echo ""
echo "=========================================="
echo "ALL LFIX EXPERIMENTS DONE"
echo "End: $(date)"
echo "=========================================="

echo ""
echo "=== SUMMARY ==="
for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    outdir="oscar_output/${name}"
    echo ""
    echo "--- ${name} ---"
    if [ -f "${outdir}/LSTM/test_results.csv" ]; then
        echo "LSTM:"
        head -2 "${outdir}/LSTM/test_results.csv" | tail -1 | cut -d, -f1,2,5
    else
        echo "LSTM: NO RESULTS"
    fi
    if [ -f "${outdir}/MVAR/test_results.csv" ]; then
        echo "MVAR:"
        head -2 "${outdir}/MVAR/test_results.csv" | tail -1 | cut -d, -f1,2,5
    else
        echo "MVAR: NO RESULTS"
    fi
done
