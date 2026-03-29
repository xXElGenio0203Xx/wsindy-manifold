#!/usr/bin/env python3
"""Move deprecated rectsim modules to deprecated/ folder."""

import subprocess
import sys
from pathlib import Path

# Change to repo root
repo_root = Path("/Users/maria_1/Desktop/wsindy-manifold")
src_dir = repo_root / "src" / "rectsim"
deprecated_dir = src_dir / "deprecated"

# List of files to move (28 total)
deprecated_files = [
    # Category A: Alternative Implementations
    "pod.py",
    "mvar.py",
    "rom_mvar.py",
    "density.py",
    
    # Category B: Evaluation/Visualization
    "rom_eval.py",
    "rom_eval_metrics.py",
    "rom_eval_viz.py",
    "rom_eval_data.py",
    "rom_eval_pipeline.py",
    "rom_video_utils.py",
    
    # Category C: Alternative Models
    "rom_mvar_model.py",
    "forecast_utils.py",
    "rom_data_utils.py",
    
    # Category D: Alternative Dynamics
    "dynamics.py",
    "morse.py",
    "integrators.py",
    "noise.py",
    
    # Category E: I/O and Config
    "io.py",
    "io_outputs.py",
    "config.py",
    "unified_config.py",
    
    # Category F: Initial Conditions
    "ic.py",
    "initial_conditions.py",
    
    # Category G: Miscellaneous
    "domain.py",
    "metrics.py",
    "cli.py",
    "rom_eval_smoke_test.py",
]

def main():
    moved_count = 0
    missing_count = 0
    error_count = 0
    
    print("="*80)
    print("MOVING DEPRECATED FILES")
    print("="*80)
    print(f"\nSource: {src_dir}")
    print(f"Destination: {deprecated_dir}\n")
    
    for filename in deprecated_files:
        src_file = src_dir / filename
        dest_file = deprecated_dir / filename
        
        # Check if file exists
        if not src_file.exists():
            print(f"⚠️  SKIP: {filename} (not found)")
            missing_count += 1
            continue
        
        # Use git mv to preserve history
        try:
            result = subprocess.run(
                ["git", "mv", str(src_file), str(dest_file)],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ Moved: {filename}")
            moved_count += 1
        except subprocess.CalledProcessError as e:
            print(f"✗ ERROR: {filename}")
            print(f"  {e.stderr}")
            error_count += 1
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Moved: {moved_count}/{len(deprecated_files)}")
    print(f"⚠️  Missing: {missing_count}")
    print(f"✗ Errors: {error_count}")
    
    if moved_count > 0:
        print(f"\n✓ Successfully moved {moved_count} deprecated files!")
        print(f"  Location: {deprecated_dir}/")
        print(f"\n  Next steps:")
        print(f"    git status")
        print(f"    git commit -m 'refactor: move {moved_count} deprecated modules to deprecated/ folder'")
    
    return 0 if error_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
