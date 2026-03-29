#!/bin/bash
#SBATCH --job-name=alvarez_unal
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/alvarez_unal_%j.out
#SBATCH -e slurm_logs/alvarez_unal_%j.err

# ============================================================================
# Alvarez-Style Lag Selection: UNALIGNED comparison
# ============================================================================
# Same 32 experiments as the aligned run, but with --unaligned flag.
# Skips shift alignment, computes fresh POD from raw (unaligned) densities,
# then runs ADF + BIC/AIC/HQIC on the unaligned latent trajectories.
# Compare results with aligned run to assess alignment → stationarity effect.
# ============================================================================
# Extra memory (48G vs 32G) because fresh SVD is computed per experiment.
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
export PYTHONPATH=src
mkdir -p slurm_logs rom_hyperparameters/results_unaligned

# Install statsmodels if not present
pip install --quiet statsmodels 2>&1 | tail -2

echo "============================================"
echo "ALVAREZ-STYLE LAG SELECTION — UNALIGNED"
echo "Date: $(date)"
echo "============================================"

# --- Representative DO_* experiments ---
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

# --- DYN experiments ---
DYN_EXPS=(
    oscar_output/DYN1_gentle_v2
    oscar_output/DYN2_hypervelocity_v2
    oscar_output/DYN3_hypernoisy_v2
    oscar_output/DYN4_blackhole_v2
    oscar_output/DYN5_supernova
    oscar_output/DYN7_pure_vicsek
)

# --- NDYN experiments ---
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
echo "Mode: UNALIGNED (fresh POD from raw densities)"
echo ""

# Run Alvarez selection with --unaligned flag
python -m rom_hyperparameters.alvarez_lag_selection \
    --experiment_dir "${ALL_EXPS[@]}" \
    --w_max 50 \
    --max_realizations 50 \
    --max_realizations_adf 20 \
    --max_realizations_var 10 \
    --output_dir rom_hyperparameters/results_unaligned \
    --unaligned

echo ""
echo "============================================"
echo "COMPLETED (UNALIGNED): $(date)"
echo "============================================"
