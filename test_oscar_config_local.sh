#!/bin/bash
# Local test of Oscar production pipeline
# This runs a smaller version locally to verify everything works

echo "=========================================="
echo "Local Oscar Production Test"
echo "=========================================="
echo "Testing with smaller parameters..."
echo ""

# Run data generation with reduced parameters
python run_data_generation.py \
  --config configs/oscar_production.yaml \
  --experiment_name oscar_production_local_test \
  --n_train 10 \
  --n_test 2 \
  --clean

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Local test completed successfully!"
    echo "=========================================="
    echo ""
    echo "Running visualization pipeline..."
    python run_visualizations.py --experiment_name oscar_production_local_test
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Full pipeline test PASSED"
        echo "Results in: predictions/oscar_production_local_test/"
        echo ""
        echo "Ready to run on Oscar when connection is available!"
    else
        echo "❌ Visualization failed"
        exit 1
    fi
else
    echo ""
    echo "❌ Data generation failed with exit code $?"
    exit 1
fi
