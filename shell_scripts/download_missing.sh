#!/bin/bash
set -e
cd /Users/maria_1/Desktop/wsindy-manifold

EXPS=(
  "S1a_v1_rawD19_p5_H100_eta0"
  "S1b_v1_rawD19_p5_H162_eta0"
  "S1c_v1_sqrtD19_p5_H100_eta0"
  "S1d_v1_sqrtD19_p5_H162_eta0"
  "S2a_v1_sqrtD19_p5_H100_eta02_aligned"
  "S2b_v1_sqrtD19_p5_H100_eta0_aligned"
  "CF8_LSTM_longPrefix_sqrtSimplex_H37"
  "CF10_longPrefix_sqrtSimplex_kstep4_H37"
)

for exp in "${EXPS[@]}"; do
  echo "=== Downloading $exp ==="
  rsync -avz --progress \
    --include='config_used.yaml' \
    --include='summary.json' \
    --include='runtime_comparison.json' \
    --include='train/' \
    --include='train/metadata.json' \
    --include='train/train_000/' \
    --include='train/train_000/density.npz' \
    --include='test/' \
    --include='test/metadata.json' \
    --include='test/test_*/' \
    --include='test/test_*/density.npz' \
    --include='test/test_*/r2_vs_time.csv' \
    --include='test/test_*/metrics_summary.json' \
    --include='rom_common/' \
    --include='rom_common/**' \
    --include='MVAR/' \
    --include='MVAR/**' \
    --include='LSTM/' \
    --include='LSTM/**' \
    --exclude='*' \
    "oscar:~/wsindy-manifold/oscar_output/$exp/" \
    "oscar_output/$exp/"
  echo "--- Done: $exp ---"
  echo ""
done

echo "=== ALL DOWNLOADS COMPLETE ==="
