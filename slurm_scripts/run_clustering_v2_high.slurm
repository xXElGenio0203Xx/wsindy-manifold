#!/bin/bash
#SBATCH --job-name=clust_v2_high
#SBATCH --output=slurm_logs/clustering_v2_high_%j.out
#SBATCH --error=slurm_logs/clustering_v2_high_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=batch

echo "======================================================================"
echo "CLUSTERING V2 HIGH — POD 0.80, N=150, 200 runs, T=15s"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start time: $(date)"
echo ""

module load python/3.11.11-5e66
source ~/wsindy_env_new/bin/activate

cd ~/wsindy-manifold
mkdir -p slurm_logs

python ROM_pipeline.py --config configs/clustering_v2_high.yaml 2>&1

echo ""
echo "======================================================================"
echo "RUNNING VISUALIZATIONS"
echo "======================================================================"

python run_visualizations.py --experiment_name clustering_v2_high 2>&1 || echo "Visualizations failed (likely ffmpeg missing)"

echo ""
echo "======================================================================"
echo "JOB COMPLETE"
echo "======================================================================"
echo "End time: $(date)"

python3 -c "
import json, pandas as pd
with open('oscar_output/clustering_v2_high/summary.json') as f:
    s = json.load(f)
print(f'''
Pipeline Summary:
  Training runs:  {s.get('n_train', 'N/A')}
  Test runs:      {s.get('n_test', 'N/A')}
  POD modes:      {s.get('r_pod', 'N/A')}
  Total time:     {s.get('total_time_minutes', 'N/A'):.1f} minutes
''')
for model in ['MVAR', 'LSTM']:
    csv = f'oscar_output/clustering_v2_high/{model}/test_results.csv'
    try:
        df = pd.read_csv(csv)
        print(f'  {model} test R²: mean={df[\"r2_reconstructed\"].mean():.4f}, min={df[\"r2_reconstructed\"].min():.4f}, max={df[\"r2_reconstructed\"].max():.4f}')
    except: pass
" 2>/dev/null

echo ""
echo "To download results locally:"
echo "  rsync -avz oscar:~/wsindy-manifold/oscar_output/clustering_v2_high/ ./oscar_output/clustering_v2_high/"
echo "  rsync -avz oscar:~/wsindy-manifold/predictions/clustering_v2_high/ ./predictions/clustering_v2_high/"
