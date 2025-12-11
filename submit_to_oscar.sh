#!/bin/bash
# ============================================================================
# Quick Oscar Submission Script
# ============================================================================
# Transfers optimal configuration and submits to Oscar cluster
# ============================================================================

set -e  # Exit on error

OSCAR_HOST="maria_1@oscar.ccv.brown.edu"
OSCAR_DIR="~/wsindy-manifold"

echo "======================================================================"
echo "OSCAR SUBMISSION: vicsek_rom_joint_optimal"
echo "======================================================================"
echo ""

# Check if files exist locally
echo "Checking local files..."
if [ ! -f "configs/vicsek_rom_joint_optimal.yaml" ]; then
    echo "✗ Configuration file not found!"
    exit 1
fi

if [ ! -f "slurm_scripts/run_vicsek_joint_optimal.slurm" ]; then
    echo "✗ SLURM script not found!"
    exit 1
fi

echo "✓ All files present"
echo ""

# Transfer files to Oscar
echo "======================================================================"
echo "STEP 1: Transferring files to Oscar"
echo "======================================================================"
echo ""

FILES_TO_TRANSFER=(
    "configs/vicsek_rom_joint_optimal.yaml"
    "slurm_scripts/run_vicsek_joint_optimal.slurm"
    "run_unified_rom_pipeline.py"
    "src/rom/"
    "src/rectsim/forecast_utils.py"
    "src/rectsim/rom_data_utils.py"
    "src/rectsim/test_evaluator.py"
)

echo "Syncing files to Oscar..."
for file in "${FILES_TO_TRANSFER[@]}"; do
    echo "  → $file"
    rsync -avz --relative "$file" "$OSCAR_HOST:$OSCAR_DIR/" 2>/dev/null || {
        echo "  ✗ Failed to transfer $file"
        exit 1
    }
done

echo ""
echo "✓ File transfer complete"
echo ""

# Submit job on Oscar
echo "======================================================================"
echo "STEP 2: Submitting job to Oscar"
echo "======================================================================"
echo ""

ssh "$OSCAR_HOST" << 'ENDSSH'
cd ~/wsindy-manifold || exit 1

# Create necessary directories
mkdir -p oscar_output
mkdir -p slurm_logs

# Load modules and activate environment
module load python/3.11.0 2>/dev/null || true
module load pytorch/2.0.1 2>/dev/null || true

# Check if virtual environment exists
if [ ! -d "~/venv_wsindy" ]; then
    echo "⚠ Warning: Virtual environment not found at ~/venv_wsindy"
    echo "  You may need to create it first with:"
    echo "  python -m venv ~/venv_wsindy"
    echo "  source ~/venv_wsindy/bin/activate"
    echo "  pip install -r requirements.txt"
fi

# Submit job
echo "Submitting SLURM job..."
JOB_ID=$(sbatch slurm_scripts/run_vicsek_joint_optimal.slurm | grep -oP '\d+')

if [ -n "$JOB_ID" ]; then
    echo ""
    echo "✓ Job submitted successfully!"
    echo "  Job ID: $JOB_ID"
    echo ""
    echo "Monitor with:"
    echo "  squeue -u $USER"
    echo "  tail -f slurm_logs/vicsek_joint_optimal_${JOB_ID}.out"
    echo ""
else
    echo "✗ Job submission failed"
    exit 1
fi
ENDSSH

SUBMIT_STATUS=$?

echo ""
if [ $SUBMIT_STATUS -eq 0 ]; then
    echo "======================================================================"
    echo "✓ SUBMISSION COMPLETE!"
    echo "======================================================================"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Check job status:"
    echo "   ssh $OSCAR_HOST"
    echo "   squeue -u \$USER"
    echo ""
    echo "2. Monitor progress:"
    echo "   ssh $OSCAR_HOST"
    echo "   tail -f ~/wsindy-manifold/slurm_logs/vicsek_joint_optimal_*.out"
    echo ""
    echo "3. Download results when complete:"
    echo "   ./download_from_oscar.sh vicsek_joint_optimal"
    echo ""
    echo "4. Visualize results:"
    echo "   python run_visualizations.py --experiment_name vicsek_joint_optimal"
    echo ""
else
    echo "======================================================================"
    echo "✗ SUBMISSION FAILED"
    echo "======================================================================"
    echo ""
    echo "Try submitting manually:"
    echo "  ssh $OSCAR_HOST"
    echo "  cd ~/wsindy-manifold"
    echo "  sbatch slurm_scripts/run_vicsek_joint_optimal.slurm"
    echo ""
    exit 1
fi
