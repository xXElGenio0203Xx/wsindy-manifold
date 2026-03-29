#!/bin/bash
#SBATCH --job-name=alvarez_hp
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/alvarez_hp_%j.out
#SBATCH -e slurm_logs/alvarez_hp_%j.err

# ============================================================================
# Alvarez-Style Lag Selection: BIC/AIC on VAR model
# ============================================================================
# Runs ADF stationarity test + BIC/AIC lag selection on all systematic
# experiments (DO_* and NDYN_*) that have training density files.
# Uses raw density reconstruction (exact, not latent_dataset.npz fallback).
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
export PYTHONPATH=src
mkdir -p slurm_logs rom_hyperparameters/results

# Install statsmodels if not present
pip install --quiet statsmodels 2>&1 | tail -2

echo "============================================"
echo "ALVAREZ-STYLE LAG SELECTION"
echo "Date: $(date)"
echo "============================================"

# --- Representative DO_* experiments (one per regime type) ---
# CS = collective swarm, DM = double mill, DR = double ring
# EC = escape collision, ES = escape symmetric, EU = escape unsymmetric
# SM = single mill
DO_EXPS=(
    oscar_output/DO_CS01_swarm_C01_l05
    oscar_output/DO_CS02_swarm_C05_l3
    oscar_output/DO_DM01_dmill_C09_l05
    oscar_output/DO_DR01_dring_C01_l01
    oscar_output/DO_DR02_dring_C09_l09
    oscar_output/DO_EC01_esccol_C2_l3
    oscar_output/DO_EC02_esccol_C3_l05
    oscar_output/DO_ES01_escsym_C3_l09
    oscar_output/DO_EU01_escuns_C2_l2
    oscar_output/DO_EU02_escuns_C3_l3
    oscar_output/DO_SM01_mill_C05_l01
    oscar_output/DO_SM02_mill_C3_l01
    oscar_output/DO_SM03_mill_C2_l05
)

# --- DYN experiments (different dynamics regimes) ---
DYN_EXPS=(
    oscar_output/DYN1_gentle_v2
    oscar_output/DYN2_hypervelocity_v2
    oscar_output/DYN3_hypernoisy_v2
    oscar_output/DYN4_blackhole_v2
    oscar_output/DYN5_supernova
    oscar_output/DYN7_pure_vicsek
)

# --- NDYN experiments (new dynamics, one per type) ---
NDYN_EXPS=(
    oscar_output/NDYN01_crawl
    oscar_output/NDYN02_flock
    oscar_output/NDYN03_sprint
    oscar_output/NDYN04_gas
    oscar_output/NDYN05_blackhole
    oscar_output/NDYN06_supernova
    oscar_output/NDYN07_crystal
    oscar_output/NDYN08_pure_vicsek
    oscar_output/NDYN10_shortrange
    oscar_output/NDYN11_noisy_collapse
    oscar_output/NDYN12_fast_explosion
    oscar_output/NDYN13_chaos
    oscar_output/NDYN14_varspeed
)

ALL_EXPS=("${DO_EXPS[@]}" "${DYN_EXPS[@]}" "${NDYN_EXPS[@]}")

echo "Total experiments: ${#ALL_EXPS[@]}"
echo ""

# Run Alvarez selection on all experiments
python -m rom_hyperparameters.alvarez_lag_selection \
    --experiment_dir "${ALL_EXPS[@]}" \
    --w_max 50 \
    --max_realizations 50 \
    --max_realizations_adf 20 \
    --max_realizations_var 10 \
    --output_dir rom_hyperparameters/results

echo ""
echo "============================================"
echo "COMPLETED: $(date)"
echo "============================================"
