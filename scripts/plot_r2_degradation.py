"""
Plot R² degradation over 12-second forecast.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load R² vs time for first test run
data_dir = Path("oscar_output/alvarez_production_20s/test/test_000")
df = pd.read_csv(data_dir / "r2_vs_time.csv")

# Create figure
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot reconstructed R²
axes[0].plot(df['time'], df['r2_reconstructed'], 'b-', linewidth=2, label='Reconstructed')
axes[0].axvline(8.0, color='gray', linestyle='--', alpha=0.5, label='Forecast Start')
axes[0].set_ylabel('R² (Reconstructed)', fontsize=12)
axes[0].set_title('ROM-MVAR Forecast Performance (d=35, w=5, T_train=8s)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc='upper right')
axes[0].set_ylim([0, 0.3])

# Add text annotation showing degradation
r2_start = df['r2_reconstructed'].iloc[0]
r2_end = df['r2_reconstructed'].iloc[-1]
axes[0].text(14, 0.25, f'R² drop: {r2_start:.3f} → {r2_end:.3f} ({100*(r2_start-r2_end)/r2_start:.1f}%)',
             fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot latent and POD R²
axes[1].plot(df['time'], df['r2_latent'], 'r-', linewidth=2, label='Latent Space')
axes[1].plot(df['time'], df['r2_pod'], 'g-', linewidth=2, label='POD Projection')
axes[1].axvline(8.0, color='gray', linestyle='--', alpha=0.5, label='Forecast Start')
axes[1].set_xlabel('Time (s)', fontsize=12)
axes[1].set_ylabel('R²', fontsize=12)
axes[1].set_title('Latent vs POD Performance', fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc='upper right')
axes[1].set_ylim([0, 1.0])

plt.tight_layout()
plt.savefig('artifacts/r2_degradation_12s_forecast.png', dpi=300, bbox_inches='tight')
print("✅ Saved plot to artifacts/r2_degradation_12s_forecast.png")
print(f"\nSummary:")
print(f"  Reconstructed R²: {r2_start:.4f} → {r2_end:.4f} (drop: {100*(r2_start-r2_end)/r2_start:.1f}%)")
print(f"  Latent R²: {df['r2_latent'].iloc[0]:.4f} → {df['r2_latent'].iloc[-1]:.4f}")
print(f"  POD R²: {df['r2_pod'].iloc[0]:.4f} → {df['r2_pod'].iloc[-1]:.4f}")
print(f"  Forecast horizon: 12 seconds (150% beyond training)")
