#!/bin/bash
for d in thesis_final lstm tier1_w5; do
  echo "=== $d ==="
  f=~/wsindy-manifold/oscar_output/NDYN05_blackhole_VS_$d/config_used.yaml
  if [ -f "$f" ]; then
    grep -E "N_particles|n_train|n_test|T_total|dt_save|N_sim|mass_post" "$f"
  else
    echo "(no config_used.yaml)"
  fi
  echo ""
done

echo "=== Per-test R2 breakdown ==="
echo ""
echo "--- thesis_final MVAR ---"
tail -n+2 ~/wsindy-manifold/oscar_output/NDYN05_blackhole_VS_thesis_final/MVAR/test_results.csv 2>/dev/null | cut -d, -f1,2,3,5
echo ""
echo "--- thesis_final LSTM ---"
tail -n+2 ~/wsindy-manifold/oscar_output/NDYN05_blackhole_VS_thesis_final/LSTM/test_results.csv 2>/dev/null | cut -d, -f1,2,3,5
echo ""
echo "--- _lstm LSTM ---"
tail -n+2 ~/wsindy-manifold/oscar_output/NDYN05_blackhole_VS_lstm/LSTM/test_results.csv 2>/dev/null | cut -d, -f1,2,3,5
echo ""
echo "--- tier1_w5 MVAR ---"
tail -n+2 ~/wsindy-manifold/oscar_output/NDYN05_blackhole_VS_tier1_w5/MVAR/test_results.csv 2>/dev/null | cut -d, -f1,2,3,5
echo ""
echo "--- tier1_w5 LSTM ---"
tail -n+2 ~/wsindy-manifold/oscar_output/NDYN05_blackhole_VS_tier1_w5/LSTM/test_results.csv 2>/dev/null | cut -d, -f1,2,3,5

echo ""
echo "--- preproc variants LSTM ---"
for pp in raw_none raw_scale raw_simplex sqrt_none sqrt_scale sqrt_simplex; do
  f=~/wsindy-manifold/oscar_output/NDYN05_blackhole_VS_preproc_${pp}/LSTM/test_results.csv
  if [ -f "$f" ]; then
    mr2=$(tail -n+2 "$f" | cut -d, -f2 | awk '{s+=$1; n++} END {if(n>0) printf "%.4f",s/n; else print "N/A"}')
    echo "  preproc_${pp}  LSTM R2=$mr2"
  fi
done
for pp in raw_none raw_scale raw_simplex sqrt_none sqrt_scale sqrt_simplex; do
  f=~/wsindy-manifold/oscar_output/NDYN05_blackhole_VS_preproc_${pp}/MVAR/test_results.csv
  if [ -f "$f" ]; then
    mr2=$(tail -n+2 "$f" | cut -d, -f2 | awk '{s+=$1; n++} END {if(n>0) printf "%.4f",s/n; else print "N/A"}')
    echo "  preproc_${pp}  MVAR R2=$mr2"
  fi
done
