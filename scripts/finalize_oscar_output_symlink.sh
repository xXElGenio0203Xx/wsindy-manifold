#!/bin/bash
# finalize_oscar_output_symlink.sh
# Run AFTER all tier3 tasks are done (no jobs writing to home oscar_output)
# Moves remaining dirs to scratch and creates the symlink
set -euo pipefail

SRC="$HOME/wsindy-manifold/oscar_output"
DST="$HOME/scratch/oscar_output"

# Safety check: ensure no running jobs are writing to oscar_output
RUNNING=$(squeue -u "$USER" -t RUNNING -o "%j" --noheader 2>/dev/null | grep -c 'tier3_ext' || true)
if [[ "$RUNNING" -gt 0 ]]; then
    echo "ERROR: $RUNNING tier3_ext tasks still RUNNING. Wait for them to finish."
    echo "Check with: squeue -u $USER"
    exit 1
fi

echo "=== Finalizing oscar_output → scratch symlink ==="
echo ""

# Move any remaining dirs from home to scratch
REMAINING=$(find "$SRC" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
if [[ "$REMAINING" -gt 0 ]]; then
    echo "Moving $REMAINING remaining dirs to scratch..."
    for d in "$SRC"/*/; do
        [[ -d "$d" ]] || continue
        name=$(basename "$d")
        if [[ -d "$DST/$name" ]]; then
            echo "  $name already in scratch — removing from home"
            rm -rf "$SRC/$name"
        else
            echo "  mv $name → scratch"
            mv "$SRC/$name" "$DST/$name"
        fi
    done
else
    echo "No remaining dirs to move."
fi

# Verify source is empty
LEFTOVER=$(find "$SRC" -maxdepth 1 -mindepth 1 2>/dev/null | wc -l)
if [[ "$LEFTOVER" -gt 0 ]]; then
    echo "WARNING: $SRC still has $LEFTOVER items:"
    ls "$SRC"
    echo "Aborting symlink creation."
    exit 1
fi

# Remove empty dir and create symlink
echo ""
echo "Removing empty $SRC and creating symlink..."
rmdir "$SRC"
ln -s "$DST" "$SRC"
echo "Created: $SRC → $DST"

echo ""
echo "=== Verification ==="
ls -la "$SRC"
echo "Contents: $(ls "$SRC" | wc -l) dirs"
echo ""
echo "Done. All future jobs will write to scratch via symlink."
