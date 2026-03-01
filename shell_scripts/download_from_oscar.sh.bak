#!/bin/bash
# ============================================================================
# Download Results from Oscar
# ============================================================================
# Downloads experimental results from Oscar cluster
# Usage: ./download_from_oscar.sh <experiment_name>
# ============================================================================

set -e

OSCAR_HOST="maria_1@oscar.ccv.brown.edu"
OSCAR_DIR="~/wsindy-manifold"

if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_name>"
    echo ""
    echo "Example: $0 vicsek_joint_optimal"
    exit 1
fi

EXPERIMENT_NAME="$1"

echo "======================================================================"
echo "DOWNLOADING RESULTS FROM OSCAR"
echo "======================================================================"
echo ""
echo "Experiment: $EXPERIMENT_NAME"
echo "Source: $OSCAR_HOST:$OSCAR_DIR/oscar_output/$EXPERIMENT_NAME/"
echo "Destination: ./oscar_output/$EXPERIMENT_NAME/"
echo ""

# Create local directory
mkdir -p "oscar_output/$EXPERIMENT_NAME"

# Check if experiment exists on Oscar
echo "Checking if experiment exists on Oscar..."
ssh "$OSCAR_HOST" "[ -d $OSCAR_DIR/oscar_output/$EXPERIMENT_NAME ]" || {
    echo "✗ Experiment directory not found on Oscar!"
    echo "  Expected: $OSCAR_DIR/oscar_output/$EXPERIMENT_NAME"
    exit 1
}

echo "✓ Experiment found"
echo ""

# Download results
echo "Downloading results..."
rsync -avz --progress \
    "$OSCAR_HOST:$OSCAR_DIR/oscar_output/$EXPERIMENT_NAME/" \
    "./oscar_output/$EXPERIMENT_NAME/" || {
    echo "✗ Download failed!"
    exit 1
}

echo ""
echo "======================================================================"
echo "✓ DOWNLOAD COMPLETE!"
echo "======================================================================"
echo ""

# Check what was downloaded
if [ -d "oscar_output/$EXPERIMENT_NAME" ]; then
    echo "Downloaded files:"
    ls -lh "oscar_output/$EXPERIMENT_NAME/" | head -20
    echo ""
    
    # Check for key files
    echo "Checking key files..."
    
    if [ -f "oscar_output/$EXPERIMENT_NAME/summary.json" ]; then
        echo "  ✓ summary.json"
        python -c "import json; s=json.load(open('oscar_output/$EXPERIMENT_NAME/summary.json')); \
                   print(f\"    Training runs: {s.get('n_train', 'N/A')}\"); \
                   print(f\"    Test runs: {s.get('n_test', 'N/A')}\"); \
                   print(f\"    POD modes: {s.get('R_POD', 'N/A')}\"); \
                   print(f\"    Models: {', '.join(s.get('models_trained', []))}\"); \
                   print(f\"    Total time: {s.get('total_time_minutes', 'N/A'):.1f} min\")" 2>/dev/null || echo "    (summary not parseable)"
    else
        echo "  ⚠ summary.json not found"
    fi
    
    if [ -d "oscar_output/$EXPERIMENT_NAME/rom_common" ]; then
        echo "  ✓ rom_common/"
    fi
    
    if [ -d "oscar_output/$EXPERIMENT_NAME/MVAR" ]; then
        echo "  ✓ MVAR/"
    fi
    
    if [ -d "oscar_output/$EXPERIMENT_NAME/LSTM" ]; then
        echo "  ✓ LSTM/"
    fi
    
    if [ -d "oscar_output/$EXPERIMENT_NAME/test" ]; then
        echo "  ✓ test/"
        TEST_COUNT=$(ls -d oscar_output/$EXPERIMENT_NAME/test/test_* 2>/dev/null | wc -l)
        echo "    ($TEST_COUNT test runs)"
    fi
    
    echo ""
fi

# Next steps
echo "======================================================================"
echo "NEXT STEPS"
echo "======================================================================"
echo ""
echo "1. Run visualizations:"
echo "   python run_visualizations.py --experiment_name $EXPERIMENT_NAME"
echo ""
echo "2. Check output:"
echo "   ls -lh predictions/$EXPERIMENT_NAME/"
echo ""
echo "3. View summary:"
echo "   cat predictions/$EXPERIMENT_NAME/pipeline_summary.json"
echo ""
