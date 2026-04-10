#!/bin/bash
#SBATCH --job-name=tier1_ws
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH -p batch
#SBATCH --array=0-17%7
#SBATCH -o slurm_logs/tier1ws_%A_%a.out
#SBATCH -e slurm_logs/tier1ws_%A_%a.err

# ============================================================================
# TIER 1 — WSINDy-only fallback (no bootstrap)
# ============================================================================
# Re-runs ONLY WSINDy PDE discovery on experiments where MVAR/LSTM are
# already complete but WSINDy timed out.  Uses --wsindy-only flag so
# sims/POD/models are skipped; training densities must already be on disk.
#
# Key difference from full tier1:
#   - bootstrap.enabled: false  (saves ~80% of WSINDy wall clock)
#   - --wsindy-only flag        (skips MVAR/LSTM)
#   - 6h time limit             (sufficient without bootstrap)
#
# Usage:
#   sbatch slurm_scripts/run_tier1_wsindy_noboot.slurm
# ============================================================================

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
export PYTHONPATH=src
mkdir -p slurm_logs

MANIFEST="configs/systematic/tier1/manifest.txt"
N_CONFIGS=$(wc -l < "$MANIFEST" | tr -d ' ')
LAST_INDEX=$((N_CONFIGS - 1))

if [ "$SLURM_ARRAY_TASK_ID" -gt "$LAST_INDEX" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID beyond manifest, exiting."
    exit 0
fi

CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
EXP_NAME=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['experiment_name'])")

# Skip if WSINDy multifield_model.json already exists
OUTDIR="oscar_output/${EXP_NAME}"
if [ -f "${OUTDIR}/WSINDy/multifield_model.json" ]; then
    echo "WSINDy already complete for $EXP_NAME — skipping."
    exit 0
fi

# Skip if training data doesn't exist (task never ran sims)
if [ ! -d "${OUTDIR}/train" ]; then
    echo "No training data for $EXP_NAME — skipping."
    exit 0
fi

echo "============================================"
echo "TIER 1 — WSINDy-only (no bootstrap)"
echo "============================================"
echo "Array task:  $SLURM_ARRAY_TASK_ID / $LAST_INDEX"
echo "Config:      $CONFIG"
echo "Experiment:  $EXP_NAME"
echo "Start time:  $(date)"

# Create a temporary config with bootstrap disabled
TMPCONFIG=$(mktemp /tmp/tier1ws_XXXXXX.yaml)
python3 -c "
import yaml, sys
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg.setdefault('wsindy', {}).setdefault('bootstrap', {})['enabled'] = False
with open('$TMPCONFIG', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
print('Bootstrap disabled in temp config', file=sys.stderr)
"

python ROM_WSINDY_pipeline.py \
    --config "$TMPCONFIG" \
    --experiment_name "$EXP_NAME" \
    --wsindy-only

EXIT_CODE=$?
rm -f "$TMPCONFIG"

echo "============================================"
echo "Exit code:   $EXIT_CODE"
echo "End time:    $(date)"
echo "============================================"
