#!/usr/bin/env bash
# sync_missing_results.sh — Pull completed noise-sweep and N-convergence WSINDy
# results from OSCAR scratch to local oscar_output/.
#
# Current status (2026-04-21):
#   Job 1824943[0-3] (wso_patch) running on OSCAR:
#     [0] NDYN04_gas_eta0p15_ws   — training done, WSINDy running (~1-2h)
#     [1] NDYN05_blackhole_eta0p05_ws — training done, WSINDy running
#     [2] NDYN08_pure_vicsek_N0500    — training done, WSINDy running
#     [3] NDYN08_pure_vicsek_N1000    — training done, WSINDy running
#   Job 1799017_3 (ws_miss[3]) running on OSCAR:
#     NDYN06_supernova_eta0p50_ws — generating training data (~8-9h)
#     → after it completes: sbatch slurm_scripts/wsonly_patch_sn.slurm
#
# Run once all jobs are done:
#   ssh oscar "squeue -u emaciaso"   # confirm empty queue
#   bash shell_scripts/sync_missing_results.sh
#   python3 generate_appendix_tables.py
#   python3 _gen_nconv_plot.py
#   python3 _gen_noise_heatmap.py    # optional, regenerates noise_coeff_stability

set -euo pipefail

REMOTE="oscar:/users/emaciaso/scratch/oscar_output"
LOCAL="oscar_output"

# ------------------------------------------------------------
# Noise-sweep WSINDy-only experiments (ws_miss array jobs)
# ------------------------------------------------------------
WS_MISSING=(
    NDYN04_gas_eta0p15_ws
    NDYN05_blackhole_eta0p05_ws
    NDYN05_blackhole_eta0p15_ws
    NDYN06_supernova_eta0p50_ws
)

echo "=== Syncing noise-sweep _ws results ==="
for exp in "${WS_MISSING[@]}"; do
    echo "  rsync $exp ..."
    rsync -av --exclude='train/' --exclude='test/' \
        "${REMOTE}/${exp}/" "${LOCAL}/${exp}/"
done

# ------------------------------------------------------------
# N-convergence full-pipeline (nconv_miss array jobs)
# N=500 and N=1000 — only pull WSINDy subdir (LSTM/MVAR already local)
# ------------------------------------------------------------
NCONV_MISSING=(
    NDYN08_pure_vicsek_N0500
    NDYN08_pure_vicsek_N1000
)

echo ""
echo "=== Syncing N-convergence WSINDy results ==="
for exp in "${NCONV_MISSING[@]}"; do
    echo "  rsync ${exp}/WSINDy ..."
    rsync -av \
        "${REMOTE}/${exp}/WSINDy/" "${LOCAL}/${exp}/WSINDy/"
done

echo ""
echo "=== Done. Now regenerate tables ==="
echo "  python3 generate_appendix_tables.py"
