#!/bin/bash
# Quick Oscar job status checker
# Usage: bash scripts/oscar_check.sh [command]

OSCAR_HOST="emaciaso@ssh.ccv.brown.edu"
WORKDIR="~/src/wsindy-manifold"

# Check if SSH master connection exists, if not create it
if ! ssh -O check "$OSCAR_HOST" 2>/dev/null; then
    echo "Opening persistent SSH connection (requires password + 2FA)..."
    ssh -fNM "$OSCAR_HOST"
    if [ $? -ne 0 ]; then
        echo "Failed to establish SSH connection"
        exit 1
    fi
    echo "✓ Connection established"
fi

# Default command: check job status
CMD="${1:-status}"

case "$CMD" in
    status)
        echo "=== Job Queue ==="
        ssh "$OSCAR_HOST" "squeue -u \$USER"
        echo ""
        echo "=== Recent Jobs ==="
        ssh "$OSCAR_HOST" "sacct -S today --format=JobID,JobName,State,ExitCode,Elapsed | grep -E 'JobID|test_'"
        ;;
    
    logs)
        echo "=== Latest ROM Training Log ==="
        ssh "$OSCAR_HOST" "cd $WORKDIR && tail -20 logs/test_rom_*.out 2>/dev/null | tail -20"
        echo ""
        echo "=== Latest Evaluation Log ==="
        ssh "$OSCAR_HOST" "cd $WORKDIR && tail -20 logs/test_eval_*.out 2>/dev/null | tail -20"
        ;;
    
    pull)
        echo "=== Pulling latest code on Oscar ==="
        ssh "$OSCAR_HOST" "cd $WORKDIR && git pull"
        ;;
    
    submit)
        echo "=== Submitting test pipeline ==="
        ssh "$OSCAR_HOST" "cd $WORKDIR && bash scripts/slurm/submit_test_pipeline.sh"
        ;;
    
    close)
        echo "Closing persistent SSH connection..."
        ssh -O exit "$OSCAR_HOST"
        ;;
    
    *)
        echo "Usage: $0 [status|logs|pull|submit|close]"
        echo ""
        echo "Commands:"
        echo "  status  - Check job queue and recent jobs (default)"
        echo "  logs    - View latest logs"
        echo "  pull    - Pull latest code on Oscar"
        echo "  submit  - Submit test pipeline"
        echo "  close   - Close persistent SSH connection"
        exit 1
        ;;
esac
