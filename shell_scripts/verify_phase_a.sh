#!/bin/bash
# ============================================================================
# verify_phase_a.sh — Check Phase A outputs before proceeding to Phase B
# ============================================================================
# Verifies that simulation data and POD basis exist for all 7 regimes.
#
# Usage (run ON Oscar, from ~/wsindy-manifold):
#   bash shell_scripts/verify_phase_a.sh
#
# WARNING: run this on a LOGIN NODE only — it does lightweight file checks.
# Do NOT run 'du', 'find -exec', or Python training from the login node.
# For disk usage: use 'myquota' or open an interact session first.
# ============================================================================

set -euo pipefail

# Block heavy commands if on a login node
source "$(dirname "$0")/_oscar_guard.sh" 2>/dev/null || true

REGIMES=(
    NDYN04_gas_thesis_final
    NDYN04_gas_VS_thesis_final
    NDYN05_blackhole_thesis_final
    NDYN05_blackhole_VS_thesis_final
    NDYN06_supernova_thesis_final
    NDYN06_supernova_VS_thesis_final
    NDYN08_pure_vicsek_thesis_final
)

OUTDIR="oscar_output"
FAIL=0

echo "============================================"
echo "Phase A Verification"
echo "============================================"

for R in "${REGIMES[@]}"; do
    OK=true

    # Check train/metadata.json exists and has entries
    META="$OUTDIR/$R/train/metadata.json"
    if [ -f "$META" ]; then
        N_ENTRIES=$(python3 -c "import json; d=json.load(open('$META')); print(len(d))")
        if [ "$N_ENTRIES" -lt 100 ]; then
            echo "WARN  $R: metadata.json has only $N_ENTRIES entries (expected >=320)"
            OK=false
        fi
    else
        echo "FAIL  $R: $META not found"
        OK=false
    fi

    # Check POD basis
    POD="$OUTDIR/$R/rom_common/pod_basis.npz"
    if [ ! -f "$POD" ]; then
        echo "FAIL  $R: $POD not found"
        OK=false
    fi

    if [ "$OK" = true ]; then
        echo "OK    $R  ($N_ENTRIES train entries)"
    else
        FAIL=$((FAIL + 1))
    fi
done

echo ""
if [ "$FAIL" -gt 0 ]; then
    echo "RESULT: $FAIL / ${#REGIMES[@]} regimes FAILED verification."
    exit 1
else
    echo "RESULT: All ${#REGIMES[@]} regimes passed. Ready for lag selection + Phase B."
    exit 0
fi
