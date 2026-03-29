#!/bin/bash
set -e
cd /Users/maria_1/Desktop/wsindy-manifold

# Experiments missing test density.npz files
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
  echo "=== $exp ==="
  # Get list of test dirs from OSCAR
  test_dirs=$(ssh oscar "ls -d ~/wsindy-manifold/oscar_output/$exp/test/test_*/ 2>/dev/null" | xargs -n1 basename)
  
  for td in $test_dirs; do
    local_dir="oscar_output/$exp/test/$td"
    mkdir -p "$local_dir"
    if [[ ! -f "$local_dir/density.npz" ]]; then
      scp "oscar:~/wsindy-manifold/oscar_output/$exp/test/$td/density.npz" "$local_dir/density.npz"
    fi
  done
  
  # Also get train density if missing
  mkdir -p "oscar_output/$exp/train/train_000"
  if [[ ! -f "oscar_output/$exp/train/train_000/density.npz" ]]; then
    scp "oscar:~/wsindy-manifold/oscar_output/$exp/train/train_000/density.npz" "oscar_output/$exp/train/train_000/density.npz" 2>/dev/null || echo "  (no train density)"
  fi
  
  # Get MVAR files if missing
  mkdir -p "oscar_output/$exp/MVAR"
  if [[ ! -f "oscar_output/$exp/MVAR/mvar_model.npz" ]]; then
    scp "oscar:~/wsindy-manifold/oscar_output/$exp/MVAR/mvar_model.npz" "oscar_output/$exp/MVAR/" 2>/dev/null || true
    scp "oscar:~/wsindy-manifold/oscar_output/$exp/MVAR/test_results.csv" "oscar_output/$exp/MVAR/" 2>/dev/null || true
  fi
  
  # Get LSTM files if they exist
  mkdir -p "oscar_output/$exp/LSTM"
  scp "oscar:~/wsindy-manifold/oscar_output/$exp/LSTM/test_results.csv" "oscar_output/$exp/LSTM/" 2>/dev/null || true
  scp "oscar:~/wsindy-manifold/oscar_output/$exp/LSTM/training_log.csv" "oscar_output/$exp/LSTM/" 2>/dev/null || true
  
  count=$(find "oscar_output/$exp/test" -name 'density.npz' | wc -l)
  echo "  -> $count test density.npz files"
done

echo ""
echo "=== DONE ==="
