#!/bin/bash
cd ~/wsindy-manifold
mkdir -p slurm_logs

submit_job() {
    local name=$1
    local time=$2
    local mem=$3
    sbatch --job-name="$name" --time="$time" -N1 -n4 --mem="$mem" -p batch \
        -o "slurm_logs/${name}_%j.out" -e "slurm_logs/${name}_%j.err" \
        --wrap="module load python/3.11.11-5e66 && source ~/wsindy_env_new/bin/activate && cd ~/wsindy-manifold && PYTHONPATH=src python ROM_pipeline.py --config configs/${name}.yaml --experiment_name ${name}"
}

# H100 jobs — 4h, 24G
submit_job AL1a_align_sqrtSimplex_D19_p5_H100_eta02_v2_speed2    04:00:00 24G
submit_job AL2a_align_sqrtSimplex_D19_p5_H100_eta085_v2_speed2   04:00:00 24G

# H250 jobs — 6h, 32G
submit_job AL1b_align_sqrtSimplex_D19_p5_H250_eta02_v2_speed2    06:00:00 32G
submit_job AL2b_align_sqrtSimplex_D19_p5_H250_eta085_v2_speed2   06:00:00 32G
submit_job AL3a_align_raw_D19_p5_H250_eta02_v2_speed2            06:00:00 32G
submit_job AL4a_align_sqrt_NOsimplex_D19_p5_H250_eta02_v2_speed2 06:00:00 32G

# H500 jobs — 10h, 48G
submit_job AL1c_align_sqrtSimplex_D19_p5_H500_eta02_v2_speed2      10:00:00 48G
submit_job AL2c_align_sqrtSimplex_D19_p5_H500_eta085_v2_speed2     10:00:00 48G
submit_job AL3b_align_raw_D19_p5_H500_eta02_v2_speed2              10:00:00 48G
submit_job AL4b_align_sqrt_NOsimplex_D19_p5_H500_eta02_v2_speed2   10:00:00 48G
submit_job AL5_align_sqrtSimplex_energy099_p5_H500_eta02_v2_speed2 10:00:00 48G

echo "=== ALL 11 JOBS SUBMITTED ==="
squeue -u emaciaso
