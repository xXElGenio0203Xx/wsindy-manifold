#!/bin/bash
# ==============================================================================
# Quick Test Pipeline (2 runs only)
# ==============================================================================

set -e

cd ~/src/wsindy-manifold
mkdir -p logs

echo "========================================="
echo "Quick Test Pipeline (2 runs)"
echo "========================================="
echo ""

# ------------------------------------------------------------------------------
# Step 1: Submit test ensemble (2 parallel simulations)
# ------------------------------------------------------------------------------
echo "[1/3] Submitting test ensemble (2 tasks)..."

ensemble_jid=$(sbatch --parsable --array=0-1 \
  --job-name=test_ensemble \
  --output=logs/test_ensemble_%A_%a.out \
  --error=logs/test_ensemble_%A_%a.err \
  --export=ALL,CONFIG=configs/vicsek_morse_test.yaml \
  scripts/slurm/run_vicsek_morse_ensemble.slurm)

echo "  ✓ Test ensemble submitted: ${ensemble_jid}"
echo ""

# ------------------------------------------------------------------------------
# Step 2: Submit ROM training
# ------------------------------------------------------------------------------
echo "[2/3] Submitting ROM training..."

rom_jid=$(sbatch --parsable --dependency=afterok:${ensemble_jid} \
  --job-name=test_rom \
  --output=logs/test_rom_%j.out \
  --error=logs/test_rom_%j.err \
  scripts/slurm/run_vicsek_morse_rom.slurm \
  configs/vicsek_morse_test.yaml)

echo "  ✓ Test ROM training submitted: ${rom_jid}"
echo ""

# ------------------------------------------------------------------------------
# Step 3: Submit evaluation
# ------------------------------------------------------------------------------
echo "[3/3] Submitting evaluation..."

eval_jid=$(sbatch --parsable --dependency=afterok:${rom_jid} \
  --job-name=test_eval \
  --output=logs/test_eval_%j.out \
  --error=logs/test_eval_%j.err \
  scripts/slurm/run_vicsek_morse_eval.slurm)

echo "  ✓ Test evaluation submitted: ${eval_jid}"
echo ""

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
echo "========================================="
echo "Test pipeline submitted!"
echo "========================================="
echo ""
echo "Job IDs:"
echo "  Ensemble:   ${ensemble_jid} (2 tasks)"
echo "  ROM train:  ${rom_jid}"
echo "  ROM eval:   ${eval_jid}"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/test_*.out"
echo ""
