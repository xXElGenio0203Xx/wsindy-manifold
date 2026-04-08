#!/bin/bash
#SBATCH --job-name=reeval_phB
#SBATCH --output=/oscar/home/emaciaso/wsindy-manifold/slurm_logs/reeval_phB_%j.out
#SBATCH --error=/oscar/home/emaciaso/wsindy-manifold/slurm_logs/reeval_phB_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=batch

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd /oscar/home/emaciaso/wsindy-manifold
export PYTHONPATH=src:$PYTHONPATH
mkdir -p slurm_logs

OUTBASE=/oscar/home/emaciaso/wsindy-manifold/oscar_output

echo "============================================"
echo "THESIS-FINAL RE-EVALUATION (Phase B)"
echo "============================================"
echo "Start:  $(date)"
echo "Node:   $(hostname)"
echo "Python: $(which python)"
echo ""

FAIL=0
for regime in \
  NDYN04_gas_thesis_final \
  NDYN04_gas_VS_thesis_final \
  NDYN05_blackhole_thesis_final \
  NDYN05_blackhole_VS_thesis_final \
  NDYN06_supernova_thesis_final \
  NDYN06_supernova_VS_thesis_final \
  NDYN08_pure_vicsek_thesis_final; do
    echo "====== $regime ======"
    if [ ! -d "$OUTBASE/$regime" ]; then
        echo "  SKIP: directory not found"
        continue
    fi
    python rerun_evaluation.py "$OUTBASE/$regime"
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "  FAILED (exit $RC)"
        FAIL=$((FAIL + 1))
    else
        echo "  OK"
    fi
    echo ""
done

echo "============================================"
echo "End:    $(date)"
echo "Failures: $FAIL / 7"
echo "============================================"
exit $FAIL
