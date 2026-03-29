#!/bin/bash
#SBATCH --job-name=wsindy_rom
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH -p batch
#SBATCH -o slurm_logs/wsindy_rom_%j.out
#SBATCH -e slurm_logs/wsindy_rom_%j.err

# ============================================================================
# ROM + WSINDy Pipeline — DYN1 gentle with rich PDE library
# ============================================================================
# Runs the integrated pipeline:
#   Steps 1-6: Standard ROM pipeline (sims, POD, MVAR, test eval)
#   Step 7:    WSINDy PDE discovery on training densities (27+ terms)
#   Step 8:    WSINDy forecast on test runs
#
# WSINDy library: u, u², u³ × {I, dx, dy, dxx, dyy, Δ} + nonlinear diffusion
#                 + gradient nonlinearity + conservative forms = 27+ candidates
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

CONFIG=${1:-configs/DYN1_gentle_wsindy.yaml}
EXPERIMENT=${2:-DYN1_gentle_wsindy}

echo "============================================"
echo "ROM + WSINDy Pipeline"
echo "============================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "Config: $CONFIG"
echo "Experiment: $EXPERIMENT"
echo ""

python ROM_WSINDY_pipeline.py \
    --config "$CONFIG" \
    --experiment_name "$EXPERIMENT"

echo ""
echo "End time: $(date)"
echo "Exit code: $?"
