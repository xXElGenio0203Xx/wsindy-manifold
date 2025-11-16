#!/bin/bash
# ==============================================================================
# Sync and Test Script for Oscar
# ==============================================================================
# This script:
# 1. Pulls latest code from GitHub
# 2. Activates conda environment
# 3. Optionally reinstalls package (if pyproject.toml or dependencies changed)
# 4. Runs a test simulation
#
# Usage:
#   ./sync_and_test.sh                    # Quick test (no reinstall)
#   ./sync_and_test.sh --reinstall        # Force reinstall package
#   ./sync_and_test.sh --config <path>    # Use specific config
# ==============================================================================

set -e  # Exit on error

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
REPO_DIR="/users/emaciaso/src/wsindy-manifold"
CONDA_ENV="wsindy"
DEFAULT_CONFIG="configs/gentle_clustering.yaml"
DEFAULT_TEST_ARGS="--sim.N 50 --sim.T 2.0 --outputs.animate false"

# Parse arguments
REINSTALL=false
CONFIG="$DEFAULT_CONFIG"
TEST_ARGS="$DEFAULT_TEST_ARGS"

while [[ $# -gt 0 ]]; do
    case $1 in
        --reinstall)
            REINSTALL=true
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--reinstall] [--config <path>]"
            exit 1
            ;;
    esac
done

# ------------------------------------------------------------------------------
# Change to repo directory
# ------------------------------------------------------------------------------
cd "$REPO_DIR" || {
    echo "❌ Error: Cannot find repo directory $REPO_DIR"
    exit 1
}

echo "========================================"
echo "🔄 Syncing wsindy-manifold from GitHub"
echo "========================================"
echo "Repository: $REPO_DIR"
echo "Current branch: $(git branch --show-current)"
echo ""

# ------------------------------------------------------------------------------
# Git pull
# ------------------------------------------------------------------------------
echo "📥 Pulling latest changes from GitHub..."
git fetch origin
git pull origin "$(git branch --show-current)"

if [ $? -eq 0 ]; then
    echo "✓ Code successfully updated"
    echo "  Latest commit: $(git log -1 --oneline)"
else
    echo "❌ Git pull failed"
    exit 1
fi
echo ""

# ------------------------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------------------------
echo "🐍 Setting up Python environment..."

# Load miniconda module
module load miniconda3/23.11.0s

# Initialize conda
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

# Activate environment
conda activate "$CONDA_ENV"

echo "✓ Conda environment activated: $CONDA_ENV"
echo "  Python: $(which python)"
echo ""

# ------------------------------------------------------------------------------
# Optional reinstall
# ------------------------------------------------------------------------------
if [ "$REINSTALL" = true ]; then
    echo "🔧 Reinstalling package..."
    pip install -e .
    echo "✓ Package reinstalled"
    echo ""
else
    echo "ℹ️  Skipping reinstall (editable install automatically uses new code)"
    echo "   Use --reinstall if you changed pyproject.toml or dependencies"
    echo ""
fi

# ------------------------------------------------------------------------------
# Verify installation
# ------------------------------------------------------------------------------
echo "✅ Verifying installation..."
python -c "import rectsim; print(f'  rectsim package: {rectsim.__file__}')" || {
    echo "❌ Error: rectsim package not found"
    echo "   Try running with --reinstall flag"
    exit 1
}

rectsim --help > /dev/null || {
    echo "❌ Error: rectsim command not working"
    exit 1
}

echo "✓ rectsim is ready"
echo ""

# ------------------------------------------------------------------------------
# Run test simulation
# ------------------------------------------------------------------------------
echo "========================================"
echo "🧪 Running test simulation"
echo "========================================"
echo "Config: $CONFIG"
echo "Args: $TEST_ARGS"
echo ""

rectsim single --config "$CONFIG" $TEST_ARGS

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ SUCCESS - Test completed"
    echo "========================================"
    echo ""
    echo "📊 Check outputs:"
    echo "   ls -lh outputs/single/"
    echo ""
    echo "🚀 Ready to submit production jobs:"
    echo "   sbatch run_rectsim_single.slurm"
else
    echo ""
    echo "========================================"
    echo "❌ FAILED - Test simulation failed"
    echo "========================================"
    exit 1
fi
