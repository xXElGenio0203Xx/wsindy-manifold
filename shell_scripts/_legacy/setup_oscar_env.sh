#!/bin/bash
# Setup script for Oscar environment
# Run this in your terminal: source setup_oscar_env.sh

# Load miniconda module
module load miniconda3/23.11.0s

# Initialize conda for bash
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

# Activate wsindy environment
conda activate wsindy

# Confirm setup
echo "✓ Conda environment 'wsindy' activated"
echo "✓ Python: $(which python)"
echo "✓ Current directory: $(pwd)"

# If not in repo directory, provide helpful message
if [[ $(pwd) != *"wsindy-manifold"* ]]; then
    echo ""
    echo "💡 Tip: cd to your repo with:"
    echo "   cd /users/emaciaso/src/wsindy-manifold"
fi
