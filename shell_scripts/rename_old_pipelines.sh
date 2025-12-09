#!/bin/bash
# Rename old pipelines to .deprecated
# The unified pipeline replaces all three

echo "Renaming old pipelines to .deprecated..."

if [ -f "run_stable_mvar_pipeline.py" ]; then
    mv run_stable_mvar_pipeline.py run_stable_mvar_pipeline.py.deprecated
    echo "✓ Renamed run_stable_mvar_pipeline.py"
fi

if [ -f "run_robust_mvar_pipeline.py" ]; then
    mv run_robust_mvar_pipeline.py run_robust_mvar_pipeline.py.deprecated
    echo "✓ Renamed run_robust_mvar_pipeline.py"
fi

if [ -f "run_gaussians_pipeline.py" ]; then
    mv run_gaussians_pipeline.py run_gaussians_pipeline.py.deprecated
    echo "✓ Renamed run_gaussians_pipeline.py"
fi

echo ""
echo "Old pipelines have been deprecated."
echo "Use run_unified_mvar_pipeline.py for all experiments."
echo ""
echo "To restore old pipelines (not recommended):"
echo "  mv run_*_pipeline.py.deprecated run_*_pipeline.py"
