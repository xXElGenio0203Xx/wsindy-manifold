#!/bin/bash
for d in ~/wsindy-manifold/oscar_output/NDYN05_blackhole_VS*/; do
  name=$(basename "$d")
  mvar="$d/MVAR/test_results.csv"
  lstm="$d/LSTM/test_results.csv"

  if [ -f "$mvar" ]; then
    mr2=$(tail -n+2 "$mvar" | cut -d, -f2 | awk '{s+=$1; n++} END {if(n>0) printf "%.4f",s/n; else print "N/A"}')
    echo "MVAR  $name  R2=$mr2"
  fi

  if [ -f "$lstm" ]; then
    lr2=$(tail -n+2 "$lstm" | cut -d, -f2 | awk '{s+=$1; n++} END {if(n>0) printf "%.4f",s/n; else print "N/A"}')
    echo "LSTM  $name  R2=$lr2"
  fi
done
