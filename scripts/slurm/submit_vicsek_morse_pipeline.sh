#!/bin/bash
# ==============================================================================
# Submit complete Vicsek-Morse ROM/MVAR pipeline with job dependencies
# ==============================================================================
# This script submits the full pipeline:
#   1. Ensemble generation (50 parallel simulations)
#   2. ROM/MVAR training (waits for ensemble)
#   3. ROM/MVAR evaluation (waits for training)
#
# Usage:
#   bash scripts/slurm/submit_vicsek_morse_pipeline.sh
#
# Total runtime: ~20-25 minutes end-to-end
# ==============================================================================

set -e

cd ~/src/wsindy-manifold

# Create logs directory
mkdir -p logs

echo "========================================="
echo "Submitting Vicsek-Morse ROM/MVAR Pipeline"
echo "========================================="
echo ""

# ------------------------------------------------------------------------------
# Step 1: Submit ensemble generation (50 parallel simulations)
# ------------------------------------------------------------------------------
echo "[1/3] Submitting ensemble generation (array job: 50 parallel tasks)..."

ensemble_jid=$(sbatch --parsable scripts/slurm/run_vicsek_morse_ensemble.slurm)

echo "  ✓ Ensemble job submitted: ${ensemble_jid}"
echo "    Runtime: ~1-2 minutes (parallel)"
echo ""

# ------------------------------------------------------------------------------
# Step 2: Submit ROM/MVAR training (waits for ensemble)
# ------------------------------------------------------------------------------
echo "[2/3] Submitting ROM/MVAR training (waits for ensemble completion)..."

rom_jid=$(sbatch --parsable --dependency=afterok:${ensemble_jid} scripts/slurm/run_vicsek_morse_rom.slurm)

echo "  ✓ ROM training job submitted: ${rom_jid}"
echo "    Dependency: afterok:${ensemble_jid}"
echo "    Runtime: ~5-10 minutes"
echo ""

# ------------------------------------------------------------------------------
# Step 3: Submit ROM/MVAR evaluation (waits for training)
# ------------------------------------------------------------------------------
echo "[3/3] Submitting ROM/MVAR evaluation (waits for training completion)..."

eval_jid=$(sbatch --parsable --dependency=afterok:${rom_jid} scripts/slurm/run_vicsek_morse_eval.slurm)

echo "  ✓ ROM evaluation job submitted: ${eval_jid}"
echo "    Dependency: afterok:${rom_jid}"
echo "    Runtime: ~5 minutes"
echo ""

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
echo "========================================="
echo "Pipeline submitted successfully!"
echo "========================================="
echo ""
echo "Job IDs:"
echo "  Ensemble:   ${ensemble_jid} (array: 50 tasks)"
echo "  ROM train:  ${rom_jid}"
echo "  ROM eval:   ${eval_jid}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/vicsek_*.out"
echo ""
echo "After completion:"
echo "  # On Oscar: check results"
echo "  ls -lh rom_mvar/vicsek_morse_base/"
echo ""
echo "  # On local machine: rsync and visualize"
echo "  rsync -avz emaciaso@ssh.ccv.brown.edu:~/src/wsindy-manifold/rom_mvar/ ./rom_mvar/"
echo "  python scripts/rom_mvar_visualize.py --experiment vicsek_morse_base --test-ics 0 1"
echo ""
echo "========================================="
