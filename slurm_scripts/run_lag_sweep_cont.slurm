#!/bin/bash
#SBATCH --job-name=lag_sweep2
#SBATCH --output=slurm_logs/lag_sweep2_%j.out
#SBATCH --error=slurm_logs/lag_sweep2_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

# Continue lag sweep for lags 25, 30, 40 (job 660978 timed out after lag=20)

set -e
module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate
cd ~/wsindy-manifold

LAGS=(25 30 40)

echo "=========================================="
echo "LAG SWEEP CONTINUATION — lags 25, 30, 40"
echo "Start: $(date)"
echo "=========================================="

for lag in "${LAGS[@]}"; do
    cfg="configs/lag_sweep/LSWEEP_CS01_mvar5_lstm${lag}.yaml"
    name="LSWEEP_CS01_mvar5_lstm${lag}"
    outdir="oscar_output/${name}"

    # Skip if already completed
    if [ -f "${outdir}/LSTM/test_results.csv" ] && [ -f "${outdir}/summary.json" ]; then
        echo ""
        echo ">>> SKIPPING lag=${lag} — already completed"
        continue
    fi

    # Clean partial output
    if [ -d "${outdir}" ] && [ ! -f "${outdir}/summary.json" ]; then
        echo ">>> Cleaning partial output for lag=${lag}"
        rm -rf "${outdir}"
    fi

    echo ""
    echo "=========================================="
    echo ">>> Running lag=${lag}"
    echo ">>> Config: ${cfg}"
    echo ">>> Start: $(date)"
    echo "=========================================="

    python ROM_WSINDY_pipeline.py --config "$cfg"
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo ">>> FAILED lag=${lag} with exit code ${exit_code}"
        continue
    fi

    echo ""
    echo ">>> COMPLETED lag=${lag}"
    echo ">>> LSTM results:"
    cat "${outdir}/LSTM/test_results.csv" 2>/dev/null | head -5
done

echo ""
echo "=========================================="
echo "LAG SWEEP CONTINUATION DONE"
echo "End: $(date)"
echo "=========================================="
