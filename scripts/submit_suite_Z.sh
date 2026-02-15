#!/bin/bash
# Submit all 6 Suite Z shift-aligned eval jobs on OSCAR
# These depend on Suite X H162 outputs (X3,X4,X7,X8,X11,X12)
# Use --dependency=afterok:<jobid> if X suite hasn't finished yet
set -e
cd ~/wsindy-manifold
mkdir -p slurm_logs

WRAP_PREFIX="module load python/3.11.11-5e66 && source ~/wsindy_env_new/bin/activate && cd ~/wsindy-manifold && PYTHONPATH=src python scripts/suite_Z_shift_aligned.py"

# Eval-only jobs are lightweight: 30 min, 8GB should suffice
for base in X3_V1_raw_H162 X4_V1_sqrtSimplex_H162 X7_V33_raw_H162 X8_V33_sqrtSimplex_H162 X11_V34_raw_H162 X12_V34_sqrtSimplex_H162; do
    # Derive Z name from base
    case "$base" in
        X3*)  zname="Z1" ;;
        X4*)  zname="Z2" ;;
        X7*)  zname="Z3" ;;
        X8*)  zname="Z4" ;;
        X11*) zname="Z5" ;;
        X12*) zname="Z6" ;;
    esac
    jobname="${zname}_shiftAlign_${base}"
    echo "Submitting $jobname ..."
    sbatch --job-name="$jobname" --time=00:30:00 -N1 -n1 --mem=8G -p batch \
        -o "slurm_logs/${jobname}_%j.out" -e "slurm_logs/${jobname}_%j.err" \
        --wrap="$WRAP_PREFIX --base $base --max-shift 3"
done

echo ""
echo "=== All 6 Suite Z jobs submitted ==="
squeue -u "$USER"
