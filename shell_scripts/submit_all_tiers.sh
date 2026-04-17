#!/bin/bash
# =============================================================================
# MASTER SUBMISSION SCRIPT — Submit all tiers + reeval to OSCAR
# =============================================================================
# Run this from ~/wsindy-manifold on OSCAR after disk migration is complete.
#
# Priority order:
#   1. reeval (LSTM scale) — fastest, tells us LSTM quality
#   2. tier1 (main regimes) — most important for thesis
#   3. tier2 (noise sweep)
#   4. tier3 (extended catalogue)
#   5. tier4 (N-convergence)
# =============================================================================

set -e
cd ~/wsindy-manifold

echo "============================================"
echo "  MASTER SUBMISSION — $(date)"
echo "============================================"
echo

# Pre-flight checks
echo "1. Checking oscar_output is a symlink to scratch..."
if [ -L oscar_output ]; then
    echo "   ✓ oscar_output -> $(readlink oscar_output)"
else
    echo "   ✗ oscar_output is NOT a symlink. Run disk migration first!"
    exit 1
fi

echo "2. Checking manifests exist..."
for m in configs/systematic/tier1/manifest.txt \
         configs/noise_sweep/manifest.txt \
         configs/extended_catalogue/manifest.txt \
         configs/n_convergence/manifest.txt; do
    if [ -f "$m" ]; then
        echo "   ✓ $m ($(wc -l < "$m" | tr -d ' ') entries)"
    else
        echo "   ✗ MISSING: $m"
        exit 1
    fi
done

echo
echo "3. Submitting jobs..."
echo

# --- REEVAL (priority 1) ---
JOB_REEVAL=$(sbatch --parsable slurm_scripts/reeval_all_scale.slurm)
echo "   REEVAL LSTM scale: job $JOB_REEVAL"

# --- TIER 1 (priority 2) ---
JOB_T1=$(sbatch --parsable slurm_scripts/run_tier1.slurm)
echo "   TIER 1 (18 tasks): job $JOB_T1"

# --- TIER 2 (priority 3) ---
JOB_T2=$(sbatch --parsable slurm_scripts/run_tier2_noise.slurm)
echo "   TIER 2 (20 tasks): job $JOB_T2"

# --- TIER 3 (priority 4) ---
JOB_T3=$(sbatch --parsable slurm_scripts/run_tier3_extended.slurm)
echo "   TIER 3 (19 tasks): job $JOB_T3"

# --- TIER 4 (priority 5) ---
JOB_T4=$(sbatch --parsable slurm_scripts/run_tier4_convergence.slurm)
echo "   TIER 4 (6 tasks):  job $JOB_T4"

echo
echo "============================================"
echo "  ALL SUBMITTED"
echo "============================================"
echo "  Reeval:  $JOB_REEVAL"
echo "  Tier 1:  $JOB_T1"
echo "  Tier 2:  $JOB_T2"
echo "  Tier 3:  $JOB_T3"
echo "  Tier 4:  $JOB_T4"
echo
echo "  Total: 1 + 18 + 20 + 19 + 6 = 64 tasks"
echo "  Monitor: squeue -u \$USER"
echo "============================================"
