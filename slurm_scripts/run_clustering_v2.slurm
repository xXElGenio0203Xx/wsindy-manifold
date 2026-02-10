#!/bin/bash
#SBATCH --job-name=clustering_v2
#SBATCH --output=slurm_logs/clustering_v2_%j.out
#SBATCH --error=slurm_logs/clustering_v2_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=batch

echo "======================================================================"
echo "CLUSTERING V2 — Forces + Fast + Strong Alignment"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start time: $(date)"
echo ""

# Environment
module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

# Run pipeline
python ROM_pipeline.py --config configs/clustering_v2.yaml --experiment_name clustering_v2 2>&1

echo ""
echo "======================================================================"
echo "RUNNING VISUALIZATIONS"
echo "======================================================================"

python run_visualizations.py --experiment_name clustering_v2 2>&1 || echo "Visualizations failed (likely ffmpeg missing on OSCAR)"

echo ""
echo "======================================================================"
echo "JOB COMPLETE"
echo "======================================================================"
echo "End time: $(date)"

# Quick summary
python3 -c "
import json
with open('oscar_output/clustering_v2/summary.json') as f:
    s = json.load(f)
print(f'''
Pipeline Summary:
  Training runs:  {s.get('n_train', 'N/A')}
  Test runs:      {s.get('n_test', 'N/A')}
  POD modes:      {s.get('r_pod', 'N/A')}
  Total time:     {s.get('total_time_minutes', 'N/A'):.1f} minutes
''')
" 2>/dev/null

# Print test R² for each model
for model in MVAR LSTM; do
    csv="oscar_output/clustering_v2/\${model}/test_results.csv"
    if [ -f "\$csv" ]; then
        echo "  \${model} test R²:"
        python3 -c "
import pandas as pd
df = pd.read_csv('\$csv')
print(f'    mean={df[\"r2_reconstructed\"].mean():.4f}, min={df[\"r2_reconstructed\"].min():.4f}, max={df[\"r2_reconstructed\"].max():.4f}')
" 2>/dev/null
    fi
done

echo ""
echo "To download results locally:"
echo "  rsync -avz oscar:~/wsindy-manifold/oscar_output/clustering_v2/ ./oscar_output/clustering_v2/"
echo "  rsync -avz oscar:~/wsindy-manifold/predictions/clustering_v2/ ./predictions/clustering_v2/"
echo ""
echo "To re-run visualizations locally:"
echo "  python run_visualizations.py --experiment_name clustering_v2"
