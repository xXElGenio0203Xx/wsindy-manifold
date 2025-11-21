#!/bin/bash
# Sync Oscar results back to local machine
# Usage: bash scripts/oscar_sync_results.sh [experiment_name]

OSCAR_HOST="emaciaso@ssh.ccv.brown.edu"
OSCAR_DIR="~/src/wsindy-manifold"
LOCAL_DIR="$HOME/wsindy-results"

# Default to syncing test results
EXPERIMENT="${1:-vicsek_morse_test}"

echo "========================================="
echo "Syncing Oscar Results to Local Machine"
echo "========================================="
echo "Experiment: $EXPERIMENT"
echo ""

# Create local results directory
mkdir -p "$LOCAL_DIR"

# Sync ROM/MVAR results
echo "Syncing rom_mvar/$EXPERIMENT/..."
rsync -avz --progress \
  "$OSCAR_HOST:$OSCAR_DIR/rom_mvar/$EXPERIMENT/" \
  "$LOCAL_DIR/rom_mvar/$EXPERIMENT/"

# Sync simulation outputs
echo ""
echo "Syncing simulations/$EXPERIMENT/..."
rsync -avz --progress \
  "$OSCAR_HOST:$OSCAR_DIR/simulations/$EXPERIMENT/" \
  "$LOCAL_DIR/simulations/$EXPERIMENT/"

# Sync logs
echo ""
echo "Syncing logs..."
rsync -avz --progress \
  "$OSCAR_HOST:$OSCAR_DIR/logs/test_*.out" \
  "$LOCAL_DIR/logs/" 2>/dev/null || true

echo ""
echo "========================================="
echo "✓ Sync complete!"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  $LOCAL_DIR/rom_mvar/$EXPERIMENT/"
echo "  $LOCAL_DIR/simulations/$EXPERIMENT/"
echo ""
echo "Next steps:"
echo "  1. Generate visualizations:"
echo "     python scripts/rom_mvar_visualize.py --experiment $EXPERIMENT"
echo ""
echo "  2. View results:"
echo "     open $LOCAL_DIR/rom_mvar/$EXPERIMENT/"
echo ""
