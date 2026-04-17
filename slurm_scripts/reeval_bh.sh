#!/bin/bash
#SBATCH --job-name=reeval_bh
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -p batch
#SBATCH -o slurm_logs/reeval_bh_%j.out
#SBATCH -e slurm_logs/reeval_bh_%j.err

cd ~/wsindy-manifold
module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate
export PYTHONPATH=src:$PYTHONPATH

echo "=== Original model: simplex ==="
python3 rerun_evaluation.py oscar_output/NDYN05_blackhole_VS_tier1_w5 --lstm-only --mass-postprocess simplex 2>&1

echo ""
echo "=== Original model: none ==="
python3 rerun_evaluation.py oscar_output/NDYN05_blackhole_VS_tier1_w5 --lstm-only --mass-postprocess none 2>&1

echo ""
echo "=== Retrain model: simplex ==="
python3 rerun_evaluation.py oscar_output/NDYN05_blackhole_VS_tier1_w5_lstm --lstm-only --mass-postprocess simplex 2>&1

echo ""
echo "=== Retrain model: none ==="
python3 rerun_evaluation.py oscar_output/NDYN05_blackhole_VS_tier1_w5_lstm --lstm-only --mass-postprocess none 2>&1
