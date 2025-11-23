#!/bin/bash
# Oscar Setup Script - Run this once to set everything up

echo "=== Oscar Environment Setup for wsindy-manifold ==="
echo ""

# Step 1: Load Python 3.10+
echo "Step 1: Loading Python module..."
module load python/3.10.8 || module load python/3.11 || module load anaconda/2023.09

# Step 2: Clean up old environment if it exists
echo "Step 2: Cleaning up old environment..."
if [ -d ~/wsindy_env ]; then
    echo "  Removing old wsindy_env..."
    rm -rf ~/wsindy_env
fi

# Step 3: Create new virtual environment
echo "Step 3: Creating virtual environment with Python $(python3 --version)..."
python3 -m venv ~/wsindy_env

# Step 4: Activate environment
echo "Step 4: Activating environment..."
source ~/wsindy_env/bin/activate

# Step 5: Upgrade pip
echo "Step 5: Upgrading pip..."
pip install --upgrade pip

# Step 6: Install dependencies
echo "Step 6: Installing dependencies..."
cd ~/wsindy-manifold
pip install -e .

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To run the quick test:"
echo "  source ~/wsindy_env/bin/activate"
echo "  cd ~/wsindy-manifold"
echo "  python run_data_generation.py --config configs/quick_oscar_test.yaml --experiment_name quick_oscar_test"
echo ""
