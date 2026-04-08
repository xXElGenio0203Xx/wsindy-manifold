#!/bin/bash
# migrate_oscar_output.sh — Move idle oscar_output dirs from home to scratch
# Safe: skips actively-written dirs (tier3 running) and dirs that overlap with scratch
set -euo pipefail

SRC="$HOME/wsindy-manifold/oscar_output"
DST="$HOME/scratch/oscar_output"

# 7 dirs actively being written by running tier3 tasks
ACTIVE=(
  NDYN01_crawl
  NDYN02_flock
  NDYN09_longrange
  NDYN10_shortrange
  NDYN13_chaos
  DO_CS01_swarm_C01_l05
  DO_CS02_swarm_C05_l3
)

# 7 dirs that already exist in scratch (stale N=100 thesis_final)
OVERLAP=(
  NDYN04_gas_thesis_final
  NDYN04_gas_VS_thesis_final
  NDYN05_blackhole_thesis_final
  NDYN05_blackhole_VS_thesis_final
  NDYN06_supernova_thesis_final
  NDYN06_supernova_VS_thesis_final
  NDYN08_pure_vicsek_thesis_final
)

is_skip() {
  local name="$1"
  for s in "${ACTIVE[@]}" "${OVERLAP[@]}"; do
    [[ "$name" == "$s" ]] && return 0
  done
  return 1
}

echo "=== oscar_output migration: HOME → scratch ==="
echo "Source: $SRC"
echo "Dest:   $DST"
echo ""

mkdir -p "$DST"

MOVE_COUNT=0
SKIP_COUNT=0
MOVED_LIST=()
SKIPPED_LIST=()

for d in "$SRC"/*/; do
  [[ -d "$d" ]] || continue
  name=$(basename "$d")
  if is_skip "$name"; then
    SKIP_COUNT=$((SKIP_COUNT + 1))
    SKIPPED_LIST+=("$name")
  else
    MOVE_COUNT=$((MOVE_COUNT + 1))
    MOVED_LIST+=("$name")
  fi
done

echo "Will move:  $MOVE_COUNT dirs"
echo "Will skip:  $SKIP_COUNT dirs (${#ACTIVE[@]} active + ${#OVERLAP[@]} overlap)"
echo ""

if [[ "$1" == "--dry-run" ]]; then
  echo "--- DRY RUN (no changes) ---"
  echo "Would move:"
  printf "  %s\n" "${MOVED_LIST[@]}"
  echo ""
  echo "Would skip:"
  printf "  %s\n" "${SKIPPED_LIST[@]}"
  exit 0
fi

echo "Moving directories..."
for name in "${MOVED_LIST[@]}"; do
  echo "  mv $name → scratch"
  mv "$SRC/$name" "$DST/$name"
done

echo ""
echo "Now removing stale overlap dirs from HOME (also in scratch)..."
for name in "${OVERLAP[@]}"; do
  if [[ -d "$SRC/$name" ]]; then
    echo "  rm -rf $SRC/$name"
    rm -rf "$SRC/$name"
  fi
done

echo ""
echo "=== Done ==="
echo "Remaining in HOME oscar_output (active tier3 only):"
ls "$SRC"/ 2>/dev/null || echo "(empty)"
echo ""
echo "Scratch oscar_output now has:"
ls "$DST"/ | wc -l
echo "dirs"
