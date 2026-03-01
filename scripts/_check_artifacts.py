#!/usr/bin/env python3
"""
Simulate C2 clamping effect on ABL2 results without re-running the full pipeline.

Strategy: Load saved test density arrays and POD artifacts, reconstruct predictions,
apply C2 vs C0, compare R².

If saved predictions aren't available, we'll use a mathematical bound instead.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import glob

# Check what artifacts are saved from ABL2
abl2_dir = Path("oscar_output/ABL2_N200_raw_none_align_H300")
print("ABL2 artifacts:")
for p in sorted(abl2_dir.rglob("*.npy")):
    print(f"  {p.relative_to(abl2_dir)}  ({p.stat().st_size / 1024:.0f} KB)")
for p in sorted(abl2_dir.rglob("*.npz")):
    print(f"  {p.relative_to(abl2_dir)}  ({p.stat().st_size / 1024:.0f} KB)")
for p in sorted(abl2_dir.rglob("*.csv")):
    print(f"  {p.relative_to(abl2_dir)}  ({p.stat().st_size / 1024:.0f} KB)")

# Check for per-test saved predictions
test_dir = abl2_dir / "test"
print(f"\nTest directory contents:")
if test_dir.exists():
    for p in sorted(test_dir.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size / 1024:.0f} KB)")
