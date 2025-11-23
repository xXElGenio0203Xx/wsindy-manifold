#!/bin/bash
# Oscar Deployment Checklist for Nice Parameters Production Run

echo "=========================================="
echo "OSCAR DEPLOYMENT VERIFICATION"
echo "=========================================="
echo ""

# Check if we're on Oscar
if [[ $(hostname) == *"oscar"* ]]; then
    echo "✓ Running on Oscar cluster"
    ON_OSCAR=true
else
    echo "⚠ Not on Oscar - showing what would be checked"
    ON_OSCAR=false
fi
echo ""

echo "1. Checking repository status..."
if [ -d ".git" ]; then
    echo "   ✓ Git repository found"
    BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
    echo "   Branch: $BRANCH"
    
    if [ "$ON_OSCAR" = true ]; then
        git fetch origin >/dev/null 2>&1
        BEHIND=$(git rev-list --count HEAD..origin/$BRANCH 2>/dev/null || echo "0")
        if [ "$BEHIND" -gt 0 ]; then
            echo "   ⚠ Local is $BEHIND commits behind origin - run 'git pull'"
        else
            echo "   ✓ Up to date with origin"
        fi
    fi
else
    echo "   ✗ Not a git repository"
fi
echo ""

echo "2. Checking required files..."
files=(
    "run_data_generation.py"
    "run_visualizations.py"
    "configs/nice_params_production.yaml"
    "run_nice_params_production.sh"
)

all_files_exist=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ✗ $file (MISSING)"
        all_files_exist=false
    fi
done
echo ""

echo "3. Checking Python environment..."
if [ "$ON_OSCAR" = true ]; then
    if [ -d "$HOME/wsindy_env" ]; then
        echo "   ✓ Virtual environment exists: ~/wsindy_env"
        if [ -f "$HOME/wsindy_env/bin/python" ]; then
            echo "   ✓ Python executable found"
        fi
    else
        echo "   ✗ Virtual environment not found at ~/wsindy_env"
        echo "     Run: python -m venv ~/wsindy_env && source ~/wsindy_env/bin/activate && pip install -r requirements.txt"
    fi
else
    echo "   ⊘ Skipping (not on Oscar)"
fi
echo ""

echo "4. Checking SLURM configuration..."
if [ -f "run_nice_params_production.sh" ]; then
    echo "   Resources requested:"
    grep "#SBATCH" run_nice_params_production.sh | while read line; do
        echo "      $line"
    done
else
    echo "   ✗ SLURM script not found"
fi
echo ""

echo "5. Checking output directory..."
if [ -d "oscar_output" ]; then
    echo "   ✓ oscar_output/ exists"
    if [ -d "oscar_output/nice_params_production" ]; then
        echo "   ⚠ oscar_output/nice_params_production/ already exists"
        echo "     Previous run may be overwritten"
    else
        echo "   ✓ oscar_output/nice_params_production/ does not exist (clean slate)"
    fi
else
    echo "   ✓ oscar_output/ will be created"
fi
echo ""

echo "6. Checking slurm_logs directory..."
if [ -d "slurm_logs" ]; then
    echo "   ✓ slurm_logs/ exists"
else
    echo "   ⚠ slurm_logs/ does not exist - creating..."
    mkdir -p slurm_logs
    echo "   ✓ Created slurm_logs/"
fi
echo ""

echo "=========================================="
echo "ESTIMATED RUNTIME"
echo "=========================================="
echo "Configuration:"
echo "  • 200 training + 50 test simulations"
echo "  • N=200 particles, T=30s each"
echo "  • 16 parallel workers"
echo "  • 64×64 density grid"
echo ""
echo "Estimated time:"
echo "  • Training: ~25 minutes (200 sims)"
echo "  • POD/MVAR: ~5 minutes"
echo "  • Testing: ~6 minutes (50 sims)"
echo "  • Predictions: <1 minute"
echo "  • Total: ~40 minutes"
echo ""
echo "Requested time: 2 hours (safe buffer)"
echo ""

echo "=========================================="
echo "READY TO SUBMIT?"
echo "=========================================="
if [ "$all_files_exist" = true ]; then
    echo "✓ All required files present"
    echo ""
    echo "To submit the job on Oscar:"
    echo "  cd ~/wsindy-manifold"
    echo "  sbatch run_nice_params_production.sh"
    echo ""
    echo "To monitor:"
    echo "  squeue -u \$USER"
    echo "  tail -f slurm_logs/nice_params_<jobid>.out"
else
    echo "✗ Some files are missing - please resolve before submitting"
fi
echo "=========================================="
