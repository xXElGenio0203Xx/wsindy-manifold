#!/bin/bash
set -e
cd /Users/maria_1/Desktop/wsindy-manifold

# Experiments that need density.npz files re-synced
EXPS=(
  "S1a_v1_rawD19_p5_H100_eta0"
  "S1b_v1_rawD19_p5_H162_eta0"
  "S1c_v1_sqrtD19_p5_H100_eta0"
  "S1d_v1_sqrtD19_p5_H162_eta0"
  "S2a_v1_sqrtD19_p5_H100_eta02_aligned"
  "S2b_v1_sqrtD19_p5_H100_eta0_aligned"
  "CF8_LSTM_longPrefix_sqrtSimplex_H37"
)

for exp in "${EXPS[@]}"; do
  echo "=== Re-syncing $exp ==="
  rsync -avz \
    --include='*/' \
    --include='config_used.yaml' \
    --include='summary.json' \
    --include='runtime_comparison.json' \
    --include='metadata.json' \
    --include='density.npz' \
    --include='r2_vs_time.csv' \
    --include='metrics_summary.json' \
    --include='pod_basis.npz' \
    --include='X_train_mean.npy' \
    --include='latent_dataset.npz' \
    --include='mvar_model.npz' \
    --include='test_results.csv' \
    --include='training_log.csv' \
    --include='runtime_profile.json' \
    --exclude='*' \
    "oscar:~/wsindy-manifold/oscar_output/$exp/" \
    "oscar_output/$exp/"
  echo "--- Done: $exp ---"
done

echo "=== ALL RE-SYNCS COMPLETE ==="
