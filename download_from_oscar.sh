#!/bin/bash
# ============================================================================
# Download Results from Oscar (Insight Data Only)
# ============================================================================
# Downloads experimental results from Oscar cluster, EXCLUDING bulk training
# data (train/ directory) to save bandwidth and disk space.
#
# Usage: 
#   ./download_from_oscar.sh <experiment_name>          # Download one experiment
#   ./download_from_oscar.sh --all                      # Download all completed experiments
#   ./download_from_oscar.sh --list                     # List available experiments
#   ./download_from_oscar.sh --status                   # Show sync status
#   ./download_from_oscar.sh --include-train <name>     # Include training data (large!)
#
# What gets downloaded (per experiment, ~110MB without train):
#   Root:          config_used.yaml, runtime_comparison.json, summary.json
#   MVAR/:         mvar_model.npz, runtime_profile.json, test_results.csv
#   LSTM/:         lstm_state_dict.pt, runtime_profile.json, test_results.csv,
#                  training_log.csv
#   rom_common/:   X_train_mean.npy, latent_dataset.npz, pod_basis.npz
#   test/:         metadata.json, index_mapping.csv
#   test/test_XXX/: density_pred*.npz, density_true.npz, metrics_summary.json,
#                   r2_vs_time.csv, trajectory.npz
#
# What gets EXCLUDED by default (~600MB per experiment):
#   train/:        Raw training density/trajectory data (only needed to retrain)
# ============================================================================

set -e

OSCAR_HOST="oscar"  # Uses SSH config alias
OSCAR_DIR="~/wsindy-manifold"
LOCAL_DIR="./oscar_output"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Parse arguments
INCLUDE_TRAIN=false
MODE="single"
EXPERIMENT_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)       MODE="all"; shift ;;
        --list)      MODE="list"; shift ;;
        --status)    MODE="status"; shift ;;
        --include-train) INCLUDE_TRAIN=true; shift ;;
        --help|-h)
            head -28 "$0" | tail -25
            exit 0 ;;
        *)           EXPERIMENT_NAME="$1"; shift ;;
    esac
done

# ============================================================================
# Helper functions
# ============================================================================

download_experiment() {
    local exp="$1"
    
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Downloading: $exp${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Check if experiment exists on Oscar
    if ! ssh "$OSCAR_HOST" "[ -d ~/wsindy-manifold/oscar_output/$exp ]" 2>/dev/null; then
        echo -e "  ${RED}✗ Experiment not found on Oscar${NC}"
        return 1
    fi
    
    # Check if experiment is complete
    if ! ssh "$OSCAR_HOST" "[ -f ~/wsindy-manifold/oscar_output/$exp/summary.json ]" 2>/dev/null; then
        echo -e "  ${YELLOW}⚠ No summary.json — experiment may still be running or failed${NC}"
        if [ "$MODE" = "all" ]; then
            echo -e "  ${YELLOW}  Skipping (use single download to force)${NC}"
            return 0
        fi
        read -p "  Download anyway? [y/N] " -n 1 -r
        echo
        [[ ! $REPLY =~ ^[Yy]$ ]] && return 0
    fi
    
    # Create local directory
    mkdir -p "$LOCAL_DIR/$exp"
    
    # Build rsync command
    RSYNC_EXCLUDES=""
    if [ "$INCLUDE_TRAIN" = false ]; then
        RSYNC_EXCLUDES="--exclude=train/"
        echo -e "  ${CYAN}Excluding train/ (use --include-train to include)${NC}"
    fi
    
    echo -e "  Source: $OSCAR_HOST:~/wsindy-manifold/oscar_output/$exp/"
    echo -e "  Dest:   $LOCAL_DIR/$exp/"
    echo ""
    
    rsync -avz --progress $RSYNC_EXCLUDES \
        "$OSCAR_HOST:~/wsindy-manifold/oscar_output/$exp/" \
        "$LOCAL_DIR/$exp/" || {
        echo -e "  ${RED}✗ rsync failed!${NC}"
        return 1
    }
    
    # Always download minimal train files needed by viz pipeline
    if [ "$INCLUDE_TRAIN" = false ]; then
        echo ""
        echo -e "  ${CYAN}Downloading minimal train files for viz pipeline...${NC}"
        mkdir -p "$LOCAL_DIR/$exp/train"
        
        # train/metadata.json is required
        rsync -az "$OSCAR_HOST:~/wsindy-manifold/oscar_output/$exp/train/metadata.json" \
            "$LOCAL_DIR/$exp/train/metadata.json" 2>/dev/null || true
        
        # First training run's density.npz (for grid dimensions)
        if [ -f "$LOCAL_DIR/$exp/train/metadata.json" ]; then
            FIRST_RUN=$(python3 -c "import json; m=json.load(open('$LOCAL_DIR/$exp/train/metadata.json')); print(m[0]['run_name'])" 2>/dev/null)
            if [ -n "$FIRST_RUN" ]; then
                mkdir -p "$LOCAL_DIR/$exp/train/$FIRST_RUN"
                rsync -az "$OSCAR_HOST:~/wsindy-manifold/oscar_output/$exp/train/$FIRST_RUN/density.npz" \
                    "$LOCAL_DIR/$exp/train/$FIRST_RUN/density.npz" 2>/dev/null || true
            fi
        fi
    fi
    
    echo ""
    verify_experiment "$exp"
}

verify_experiment() {
    local exp="$1"
    local dir="$LOCAL_DIR/$exp"
    
    echo -e "  ${CYAN}Verifying insight files...${NC}"
    
    # Key insight files (path:label)
    local key_files=(
        "config_used.yaml:Config"
        "summary.json:Summary"
        "runtime_comparison.json:Runtime Comparison"
        "MVAR/test_results.csv:MVAR Results"
        "MVAR/runtime_profile.json:MVAR Runtime"
        "MVAR/mvar_model.npz:MVAR Model"
        "LSTM/test_results.csv:LSTM Results"
        "LSTM/runtime_profile.json:LSTM Runtime"
        "LSTM/training_log.csv:LSTM Training Log"
        "LSTM/lstm_state_dict.pt:LSTM Weights"
        "rom_common/pod_basis.npz:POD Basis"
        "rom_common/X_train_mean.npy:Train Mean"
        "rom_common/latent_dataset.npz:Latent Dataset"
        "test/metadata.json:Test Metadata"
        "test/index_mapping.csv:Index Mapping"
    )
    
    local present=0
    local missing=0
    
    for entry in "${key_files[@]}"; do
        IFS=':' read -r filepath label <<< "$entry"
        if [ -f "$dir/$filepath" ]; then
            present=$((present + 1))
        else
            echo -e "    ${YELLOW}missing: $filepath${NC}"
            missing=$((missing + 1))
        fi
    done
    
    # Per-test files
    local test_count=$(ls -d "$dir/test/test_"* 2>/dev/null | wc -l | tr -d ' ')
    local metrics_count=$(find "$dir/test" -name "metrics_summary.json" 2>/dev/null | wc -l | tr -d ' ')
    local r2time_count=$(find "$dir/test" -name "r2_vs_time.csv" 2>/dev/null | wc -l | tr -d ' ')
    
    echo -e "  ${GREEN}✓ $present/$((present + missing)) key files | $test_count tests | $metrics_count metrics | $r2time_count r2_vs_time${NC}"
    
    # Print summary highlights if available
    if [ -f "$dir/summary.json" ]; then
        python3 -c "
import json
s = json.load(open('$dir/summary.json'))
models = ', '.join(s.get('models_trained', []))
t = s.get('total_time_minutes', 0)
n_train = s.get('n_train', '?')
n_test = s.get('n_test', '?')
R = s.get('R_POD', '?')
print(f'  📋 {n_train} train | {n_test} test | {R} POD modes | {models} | {t:.0f}min')
" 2>/dev/null || true
    fi
}

# ============================================================================
# Main
# ============================================================================

echo "======================================================================"
echo "OSCAR EXPERIMENT DOWNLOADER"
echo "======================================================================"

case "$MODE" in
    list)
        echo ""
        echo -e "${BLUE}Querying Oscar for experiments...${NC}"
        echo ""
        printf "  %-35s %-12s %s\n" "Experiment" "Status" "Details"
        printf "  %-35s %-12s %s\n" "-----------------------------------" "------------" "----------------------------"
        
        ssh "$OSCAR_HOST" 'for d in ~/wsindy-manifold/oscar_output/synthesis_*/; do exp=$(basename "$d"); if [ -f "$d/summary.json" ]; then echo "$exp COMPLETE"; else echo "$exp INCOMPLETE"; fi; done' 2>/dev/null | while read -r exp status; do
            if [ "$status" = "COMPLETE" ]; then
                printf "  %-35s ${GREEN}%-12s${NC}\n" "$exp" "$status"
            else
                printf "  %-35s ${YELLOW}%-12s${NC}\n" "$exp" "$status"
            fi
        done
        
        echo ""
        echo "To download:     $0 <experiment_name>"
        echo "To download all: $0 --all"
        ;;
    
    status)
        echo ""
        echo -e "${BLUE}Checking sync status...${NC}"
        echo ""
        
        printf "  %-35s %-12s %-12s %s\n" "Experiment" "Oscar" "Local" "Status"
        printf "  %-35s %-12s %-12s %s\n" "-----------------------------------" "------------" "------------" "---------------"
        
        OSCAR_EXPS=$(ssh "$OSCAR_HOST" 'ls -d ~/wsindy-manifold/oscar_output/synthesis_*/ 2>/dev/null | xargs -I{} basename {}' 2>/dev/null)
        
        for exp in $OSCAR_EXPS; do
            oscar_status=$(ssh "$OSCAR_HOST" "[ -f ~/wsindy-manifold/oscar_output/$exp/summary.json ] && echo 'COMPLETE' || echo 'RUNNING'" 2>/dev/null)
            
            if [ -d "$LOCAL_DIR/$exp" ]; then
                # Count key files present locally
                local_count=0
                for f in summary.json config_used.yaml MVAR/test_results.csv MVAR/mvar_model.npz LSTM/test_results.csv rom_common/pod_basis.npz test/metadata.json; do
                    [ -f "$LOCAL_DIR/$exp/$f" ] && local_count=$((local_count + 1))
                done
                local_status="YES ($local_count/7)"
            else
                local_status="NO"
            fi
            
            # Sync status
            if [ "$oscar_status" = "COMPLETE" ] && [ -d "$LOCAL_DIR/$exp" ] && [ "$local_count" -ge 5 ]; then
                sync_icon="${GREEN}✓ synced${NC}"
            elif [ "$oscar_status" = "COMPLETE" ]; then
                sync_icon="${RED}✗ needs download${NC}"
            else
                sync_icon="${YELLOW}⏳ running${NC}"
            fi
            
            printf "  %-35s %-12s %-12s " "$exp" "$oscar_status" "$local_status"
            echo -e "$sync_icon"
        done
        
        echo ""
        echo "To sync all: $0 --all"
        ;;
    
    all)
        echo ""
        echo -e "${BLUE}Downloading all completed experiments...${NC}"
        
        COMPLETED_EXPS=$(ssh "$OSCAR_HOST" 'for d in ~/wsindy-manifold/oscar_output/synthesis_*/; do [ -f "$d/summary.json" ] && basename "$d"; done; true' 2>/dev/null)
        
        if [ -z "$COMPLETED_EXPS" ]; then
            echo -e "${RED}No completed experiments found on Oscar${NC}"
            exit 1
        fi
        
        COUNT=$(echo "$COMPLETED_EXPS" | wc -l | tr -d ' ')
        echo -e "Found ${GREEN}$COUNT${NC} completed experiments"
        
        for exp in $COMPLETED_EXPS; do
            download_experiment "$exp"
        done
        
        echo ""
        echo -e "${GREEN}======================================================================"
        echo "ALL DOWNLOADS COMPLETE!"
        echo "======================================================================${NC}"
        echo ""
        echo "Next: Run visualizations for each experiment:"
        for exp in $COMPLETED_EXPS; do
            echo "  python run_visualizations.py --experiment_name $exp"
        done
        ;;
    
    single)
        if [ -z "$EXPERIMENT_NAME" ]; then
            echo ""
            echo "Usage: $0 <experiment_name>"
            echo "       $0 --all           Download all completed"
            echo "       $0 --list          List experiments on Oscar"
            echo "       $0 --status        Show sync status"
            echo ""
            echo "Run '$0 --list' to see available experiments."
            exit 1
        fi
        
        download_experiment "$EXPERIMENT_NAME"
        
        echo ""
        echo -e "${GREEN}======================================================================${NC}"
        echo -e "${GREEN}DOWNLOAD COMPLETE!${NC}"
        echo -e "${GREEN}======================================================================${NC}"
        echo ""
        echo "Next steps:"
        echo "  python run_visualizations.py --experiment_name $EXPERIMENT_NAME"
        ;;
esac
