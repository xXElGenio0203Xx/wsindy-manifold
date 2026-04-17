#!/bin/bash
echo "=== WSINDy STATUS ==="
for regime in NDYN04_gas NDYN04_gas_VS NDYN05_blackhole NDYN05_blackhole_VS NDYN06_supernova NDYN06_supernova_VS NDYN07_crystal NDYN07_crystal_VS NDYN08_pure_vicsek; do
  ws3="/users/emaciaso/scratch/oscar_output/${regime}_wsindy_v3/WSINDy/multifield_model.json"
  tf="/users/emaciaso/scratch/oscar_output/${regime}_thesis_final"
  has_ws3="NO"; [ -f "$ws3" ] && has_ws3="YES"
  has_tf="NO"; [ -d "$tf" ] && has_tf="YES"
  has_pod="NO"; [ -f "$tf/pod_modes.npz" ] && has_pod="YES"
  has_train="NO"; [ -d "$tf/train" ] && has_train="YES"
  echo "$regime | WSINDy_v3=$has_ws3 | thesis_final=$has_tf | pod=$has_pod | train_dir=$has_train"
done
