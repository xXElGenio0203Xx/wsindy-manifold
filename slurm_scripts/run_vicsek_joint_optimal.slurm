#!/bin/bash
#SBATCH --job-name=vicsek_joint_optimal
#SBATCH --output=slurm_logs/vicsek_joint_optimal_%j.out
#SBATCH --error=slurm_logs/vicsek_joint_optimal_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=batch

# ============================================================================
# VICSEK ROM – JOINT MVAR + LSTM OPTIMAL CONFIG
# ============================================================================
# This script runs the unified ROM pipeline with optimal parameters:
# - 400 training runs (10s each)
# - 40 test runs (10s each)
# - d = 25 latent modes
# - lag = 9 (both MVAR and LSTM)
# - ~16,400 training windows
# - MVAR: 5,625 parameters (well-determined with ridge=1e-4)
# - LSTM: 16 hidden units, 1 layer (fair comparison)
# ============================================================================

echo "======================================================================"
echo "JOB: VICSEK ROM JOINT OPTIMAL (MVAR + LSTM)"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "======================================================================"

# Load environment
source ~/wsindy-manifold/shell_scripts/setup_oscar_env.sh

# Navigate to project directory
cd ~/wsindy-manifold || exit 1

# Create output directories
mkdir -p oscar_output
mkdir -p slurm_logs

# Configuration
CONFIG_FILE="configs/vicsek_rom_joint_optimal.yaml"
EXPERIMENT_NAME="vicsek_joint_optimal"

echo ""
echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Output dir: oscar_output/$EXPERIMENT_NAME"
echo ""

# Run unified ROM pipeline
echo "======================================================================"
echo "RUNNING UNIFIED ROM PIPELINE"
echo "======================================================================"
echo "Expected execution:"
echo "  - Training: ~400 runs × 10s = 4000s simulation"
echo "  - POD: Build shared basis with 25 modes"
echo "  - MVAR training: ~5,625 parameters, ridge=1e-4"
echo "  - LSTM training: 16 hidden units, 1 layer, max 500 epochs"
echo "  - Test evaluation: 40 runs × 10s with closed-loop forecast"
echo "======================================================================"
echo ""

python run_unified_rom_pipeline.py \
    --config "$CONFIG_FILE" \
    --experiment_name "$EXPERIMENT_NAME" \
    2>&1 | tee "oscar_output/${EXPERIMENT_NAME}_pipeline.log"

# Check if pipeline succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ PIPELINE COMPLETED SUCCESSFULLY"
    echo "======================================================================"
    
    # Print summary
    if [ -f "oscar_output/$EXPERIMENT_NAME/summary.json" ]; then
        echo ""
        echo "Pipeline Summary:"
        python -c "import json; s=json.load(open('oscar_output/$EXPERIMENT_NAME/summary.json')); \
                   print(f\"  Training runs: {s.get('n_train', 'N/A')}\"); \
                   print(f\"  Test runs: {s.get('n_test', 'N/A')}\"); \
                   print(f\"  POD modes: {s.get('R_POD', 'N/A')}\"); \
                   print(f\"  Models: {', '.join(s.get('models_trained', []))}\"); \
                   print(f\"  Total time: {s.get('total_time_minutes', 'N/A'):.1f} minutes\")" 2>/dev/null || echo "  (summary file exists but not parsed)"
    fi
    
    # List output structure
    echo ""
    echo "Output structure:"
    ls -lh "oscar_output/$EXPERIMENT_NAME/" | head -20
    
else
    echo ""
    echo "======================================================================"
    echo "✗ PIPELINE FAILED"
    echo "======================================================================"
    echo "Check log file: oscar_output/${EXPERIMENT_NAME}_pipeline.log"
    echo "Check error file: slurm_logs/vicsek_joint_optimal_${SLURM_JOB_ID}.err"
    exit 1
fi

echo ""
echo "======================================================================"
echo "JOB COMPLETE"
echo "======================================================================"
echo "End time: $(date)"
echo "Output directory: oscar_output/$EXPERIMENT_NAME"
echo ""
echo "Next steps:"
echo "  1. Download results: rsync -avz oscar:~/wsindy-manifold/oscar_output/$EXPERIMENT_NAME/ ./oscar_output/"
echo "  2. Run visualizations: python run_visualizations.py --experiment_name $EXPERIMENT_NAME"
echo "======================================================================"
