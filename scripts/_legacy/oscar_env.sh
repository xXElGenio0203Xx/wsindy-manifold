#!/usr/bin/env bash
# ==============================================================================
# Oscar Environment Setup for wsindy-manifold
# ==============================================================================
# Quick setup script to load conda environment and navigate to repo.
#
# Usage:
#   1. Copy to Oscar home: scp scripts/oscar_env.sh emaciaso@ssh.ccv.brown.edu:~/
#   2. Add alias to ~/.bashrc: alias wsindy-oscar="source ~/oscar_env.sh"
#   3. After login: wsindy-oscar
# ==============================================================================

# Load miniconda module
module --ignore_cache load miniconda3/23.11.0s-odstpk5

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate environment
conda activate wsindy

# Navigate to repo
cd ~/src/wsindy-manifold

# Show status
echo "✓ Environment: wsindy"
echo "✓ Directory: $(pwd)"
echo "✓ Python: $(which python)"
echo ""
echo "Ready to run simulations!"
echo ""
echo "Quick commands:"
echo "  sbatch scripts/slurm/run_vicsek_morse_ensemble.slurm  # Run 50 parallel sims"
echo "  bash scripts/slurm/submit_vicsek_morse_pipeline.sh     # Run full pipeline"
echo "  squeue -u \$USER                                        # Check job status"
