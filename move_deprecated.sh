#!/bin/bash
# Move deprecated files to src/rectsim/deprecated/

cd /Users/maria_1/Desktop/wsindy-manifold

# Category A: Alternative Implementations
git mv src/rectsim/pod.py src/rectsim/deprecated/
git mv src/rectsim/mvar.py src/rectsim/deprecated/
git mv src/rectsim/rom_mvar.py src/rectsim/deprecated/
git mv src/rectsim/density.py src/rectsim/deprecated/

# Category B: Evaluation/Visualization (not in training pipeline)
git mv src/rectsim/rom_eval.py src/rectsim/deprecated/
git mv src/rectsim/rom_eval_metrics.py src/rectsim/deprecated/
git mv src/rectsim/rom_eval_viz.py src/rectsim/deprecated/
git mv src/rectsim/rom_eval_data.py src/rectsim/deprecated/
git mv src/rectsim/rom_eval_pipeline.py src/rectsim/deprecated/
git mv src/rectsim/rom_video_utils.py src/rectsim/deprecated/

# Category C: Alternative Models
git mv src/rectsim/rom_mvar_model.py src/rectsim/deprecated/
git mv src/rectsim/forecast_utils.py src/rectsim/deprecated/
git mv src/rectsim/rom_data_utils.py src/rectsim/deprecated/

# Category D: Alternative Dynamics
git mv src/rectsim/dynamics.py src/rectsim/deprecated/
git mv src/rectsim/morse.py src/rectsim/deprecated/
git mv src/rectsim/integrators.py src/rectsim/deprecated/
git mv src/rectsim/noise.py src/rectsim/deprecated/

# Category E: I/O and Config
git mv src/rectsim/io.py src/rectsim/deprecated/
git mv src/rectsim/io_outputs.py src/rectsim/deprecated/
git mv src/rectsim/config.py src/rectsim/deprecated/
git mv src/rectsim/unified_config.py src/rectsim/deprecated/

# Category F: Initial Conditions
git mv src/rectsim/ic.py src/rectsim/deprecated/
git mv src/rectsim/initial_conditions.py src/rectsim/deprecated/

# Category G: Miscellaneous
git mv src/rectsim/domain.py src/rectsim/deprecated/
git mv src/rectsim/metrics.py src/rectsim/deprecated/
git mv src/rectsim/cli.py src/rectsim/deprecated/
git mv src/rectsim/rom_eval_smoke_test.py src/rectsim/deprecated/

echo "✓ Moved 28 deprecated files to src/rectsim/deprecated/"
