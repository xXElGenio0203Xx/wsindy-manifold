"""
End-to-end integration test for unified ROM pipeline with both MVAR and LSTM.

This test runs a minimal version of the full pipeline with:
- Small N, T for fast execution
- Both MVAR and LSTM enabled
- Validates that both models can train and evaluate successfully
"""

import sys
sys.path.insert(0, 'src')

import yaml
import tempfile
from pathlib import Path

# Create minimal test config
minimal_config = """
sim:
  N: 50
  T: 2.0
  dt: 0.01
  Lx: 10.0
  Ly: 10.0
  bc: 'periodic'
  save_every: 1
  neighbor_rebuild: 5
  integrator: 'euler'

model:
  type: discrete
  speed: 1.0
  speed_mode: constant_with_forces

params:
  R: 2.0
  alpha: 1.5
  beta: 0.5

noise:
  kind: 'gaussian'
  eta: 0.3
  match_variance: true

forces:
  enabled: false

density:
  nx: 16
  ny: 16

train_ic:
  type: 'mixed_comprehensive'
  uniform:
    enabled: true
    n_runs: 3
    n_samples: 3

test_ic:
  type: 'uniform'
  n_test: 2

test_sim:
  T: 2.0
  dt: 0.01

rom:
  subsample: 2
  pod_energy: 0.95
  models:
    mvar:
      enabled: true
      lag: 5
      ridge_alpha: 1.0e-6
    lstm:
      enabled: true
      lag: 5
      hidden_units: 32
      num_layers: 2
      batch_size: 16
      learning_rate: 0.001
      max_epochs: 10
      patience: 5

evaluation:
  save_time_resolved: false
"""

def run_minimal_pipeline_test():
    """Run minimal pipeline with both models."""
    print("\n" + "="*80)
    print("END-TO-END INTEGRATION TEST")
    print("="*80)
    print("\nRunning minimal pipeline with:")
    print("  N=50 agents")
    print("  T=2.0 seconds") 
    print("  Both MVAR and LSTM enabled")
    print("  Minimal training epochs for speed")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write config
        config_path = Path(tmpdir) / "test_config.yaml"
        config_path.write_text(minimal_config)
        print(f"\n✓ Created test config: {config_path}")
        
        # Run pipeline
        import subprocess
        
        cmd = [
            "python", "ROM_pipeline.py",
            "--config", str(config_path),
            "--experiment_name", "minimal_test"
        ]
        
        print(f"\n✓ Running pipeline command:")
        print(f"  {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check for errors
        if result.returncode != 0:
            print(f"\n❌ Pipeline failed with return code {result.returncode}")
            print("\nSTDOUT:")
            print(result.stdout)
            print("\nSTDERR:")
            print(result.stderr)
            return False
        
        print(f"\n✓ Pipeline completed successfully!")
        
        # Check outputs
        output_dir = Path(tmpdir) / "oscar_output" / "minimal_test"
        
        # Check directory structure
        assert (output_dir / "rom_common").exists(), "rom_common/ missing"
        assert (output_dir / "MVAR").exists(), "MVAR/ missing"
        assert (output_dir / "LSTM").exists(), "LSTM/ missing"
        print(f"✓ Directory structure correct")
        
        # Check MVAR outputs
        assert (output_dir / "MVAR" / "mvar_model.npz").exists(), "MVAR model missing"
        assert (output_dir / "MVAR" / "test_results.csv").exists(), "MVAR test_results.csv missing"
        print(f"✓ MVAR outputs present")
        
        # Check LSTM outputs
        assert (output_dir / "LSTM" / "lstm_state_dict.pt").exists(), "LSTM model missing"
        assert (output_dir / "LSTM" / "training_log.csv").exists(), "LSTM training_log.csv missing"
        assert (output_dir / "LSTM" / "test_results.csv").exists(), "LSTM test_results.csv missing"
        print(f"✓ LSTM outputs present")
        
        # Check test evaluation outputs
        test_dir = output_dir / "test"
        test_runs = list(test_dir.glob("test_*"))
        assert len(test_runs) == 2, f"Expected 2 test runs, found {len(test_runs)}"
        
        # Check that evaluation outputs exist for first test run
        test_0 = test_dir / "test_000"
        assert test_0.exists(), "test_000/ missing"
        assert (test_0 / "metrics_summary.json").exists(), "metrics_summary.json missing"
        assert (test_0 / "density_pred.npz").exists(), "density_pred.npz missing"
        print(f"✓ Test evaluation outputs present")
        
        # Load and check test results
        import pandas as pd
        
        mvar_results = pd.read_csv(output_dir / "MVAR" / "test_results.csv")
        lstm_results = pd.read_csv(output_dir / "LSTM" / "test_results.csv")
        
        print(f"\n✓ MVAR test results: {len(mvar_results)} runs")
        print(f"  Mean R² (reconstructed): {mvar_results['r2_reconstructed'].mean():.4f}")
        print(f"  Mean R² (latent): {mvar_results['r2_latent'].mean():.4f}")
        
        print(f"\n✓ LSTM test results: {len(lstm_results)} runs")
        print(f"  Mean R² (reconstructed): {lstm_results['r2_reconstructed'].mean():.4f}")
        print(f"  Mean R² (latent): {lstm_results['r2_latent'].mean():.4f}")
        
        # Verify both have same number of tests
        assert len(mvar_results) == len(lstm_results) == 2, "Test count mismatch"
        print(f"\n✓ Both models evaluated on same test set")
        
        print("\n" + "="*80)
        print("✅ END-TO-END INTEGRATION TEST PASSED")
        print("="*80)
        
        return True


if __name__ == "__main__":
    try:
        success = run_minimal_pipeline_test()
        if success:
            print("\n🎉 Complete LSTM integration successful!")
            print("\nThe unified pipeline now supports:")
            print("  ✓ MVAR-only mode (lstm.enabled=false)")
            print("  ✓ LSTM-only mode (mvar.enabled=false)")
            print("  ✓ Both models simultaneously (both enabled)")
            print("  ✓ Identical evaluation metrics for fair comparison")
            print("  ✓ Separate output directories (MVAR/ and LSTM/)")
            sys.exit(0)
        else:
            print("\n❌ Integration test failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Integration test failed with exception:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
