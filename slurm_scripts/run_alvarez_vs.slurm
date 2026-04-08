#!/bin/bash
#SBATCH --job-name=alvarez_vs
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/alvarez_vs_%j.out
#SBATCH -e slurm_logs/alvarez_vs_%j.err

# ============================================================================
# Alvarez Lag Selection — VS Regimes Only
# ============================================================================
# Runs ADF + BIC/AIC/HQIC/FPE lag selection on variable-speed thesis_final
# experiments. Results needed to confirm BIC lags for Tier 1 _bic configs.
#
# Pre-requisite: Phase B thesis_final VS experiments must have completed
# (need rom_common/ with latent data).
#
# Usage:
#   sbatch slurm_scripts/run_alvarez_vs.slurm
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
export PYTHONPATH=src
mkdir -p slurm_logs rom_hyperparameters/results_vs

# Install statsmodels if not present
pip install --quiet statsmodels 2>&1 | tail -2

echo "============================================"
echo "ALVAREZ LAG SELECTION — VS Regimes"
echo "Date: $(date)"
echo "============================================"

# VS experiments from Phase B thesis_final
VS_EXPS=()

for exp_dir in \
    oscar_output/NDYN04_gas_VS_thesis_final \
    oscar_output/NDYN05_blackhole_VS_thesis_final \
    oscar_output/NDYN06_supernova_VS_thesis_final; do
    if [ -d "$exp_dir/rom_common" ]; then
        VS_EXPS+=("$exp_dir")
        echo "  Found: $exp_dir"
    else
        echo "  SKIP (no rom_common/): $exp_dir"
    fi
done

if [ ${#VS_EXPS[@]} -eq 0 ]; then
    echo "ERROR: No VS experiment directories found with rom_common/"
    echo "Phase B thesis_final VS jobs may not have completed yet."
    exit 1
fi

echo ""
echo "Running lag selection on ${#VS_EXPS[@]} VS experiments..."
echo ""

python -m rom_hyperparameters.alvarez_lag_selection \
    --experiment_dir "${VS_EXPS[@]}" \
    --w_max 50 \
    --max_realizations 50 \
    --max_realizations_adf 20 \
    --max_realizations_var 10 \
    --output_dir rom_hyperparameters/results_vs

echo ""
echo "============================================"
echo "COMPLETED: $(date)"
echo "============================================"
echo ""
echo "Results in: rom_hyperparameters/results_vs/"
echo "Check summary.json files for lag_bic values."
