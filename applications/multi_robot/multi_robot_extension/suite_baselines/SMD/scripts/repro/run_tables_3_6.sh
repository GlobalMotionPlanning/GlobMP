#!/bin/bash
# ============================================================================
# Regenerate SMD (arXiv:2502.03607) Appendix C, Tables 3-6.
#
# RUN IT LIKE THIS (survives you closing the terminal / losing the network):
#
#     nohup bash <path>/SMD/scripts/repro/run_tables_3_6.sh \
#         > /tmp/tables_3_6.log 2>&1 &
#     disown
#
# Then check on it any time with:
#     tail -f /tmp/tables_3_6.log
#     bash .../run_tables_3_6.sh --status      # just prints coverage, runs nothing
#
# Safe to run again at any point -- it is idempotent. Every cell that already
# has a result file is skipped, so a reboot, a crash, or a Ctrl-C costs you
# only the single cell that was in flight. A PID lock stops two copies from
# racing (two writers to one results file corrupts it).
#
# ----------------------------------------------------------------------------
# WHAT IT PRODUCES, AND WHAT IT CANNOT
#
# Methods in SMD's Tables 3-6: EECBS, DM, MPD, MMD, Ours(=SMD).
#
#   SMD   -- yes. eval_results.json per cell.
#   DM    -- yes. Same trained weights with the augmented-Lagrangian
#            projection disabled (proj_params=None), which is what "DM"
#            denotes. Writes eval_results_dm.json.
#   MMD   -- yes, via evaluate_mmd_all.py (already complete for most cells).
#   MPD   -- partially. The MPD reproduction in ../../MPD is single-robot;
#            SMD applies MPD per-robot with no coordination. Not wired up
#            here; left for a follow-up.
#   EECBS -- NO. There is no EECBS implementation anywhere in these repos
#            (only plain CBS and prioritized planning). EECBS is a discrete
#            grid MAPF solver in C++; these environments are continuous 2D
#            with disk robots, so it needs a real port, not a wrapper. It is
#            deliberately not attempted here rather than faked.
#
# Metrics: S (success), L (path length), A (mean |acceleration|), C
# (collision ratio) are all computed. Wall-clock timing for Table 5 is NOT
# instrumented per-method in this harness.
# ============================================================================

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONDA_SH="$REPO_ROOT/../miniconda3/etc/profile.d/conda.sh"
LOG_DIR="$REPO_ROOT/logs_tables_3_6"
LOCK_FILE="/tmp/tables_3_6.lock"

# Every cell appearing in Tables 3-6, as named in scenarios.py.
# (corridor/room/shelf are this repo's stand-ins for the paper's
# Conveyor/Highways/Drop-Region geometry -- see scenarios.py.)
SCENARIOS=(
  empty-3 empty-6 empty-9
  basic-3 basic-6 basic-9
  dense-3 dense-6 dense-9
  corridor-2 corridor-3 corridor-6 corridor-9
  room-3 room-6 room-9
  shelf-3 shelf-6 shelf-9
)

# How many evals to run at once. The GPU sat ~5% busy running these one at a
# time, so the bottleneck is per-process CPU, not VRAM. 4 is comfortable
# alongside anything else on the box; raise it if the machine is idle.
PARALLEL=4

source "$CONDA_SH"
conda activate smd
cd "$SCRIPT_DIR"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------- status ---
if [ "${1:-}" = "--status" ]; then
    echo "cell            SMD   DM"
    echo "------------------------------"
    for tag in "${SCENARIOS[@]}"; do
        d=$(python3 -c "
from scenarios import get_scenario
print(get_scenario('$tag')['model_id'])" 2>/dev/null)
        smd="-"; dm="-"
        [ -f "$REPO_ROOT/data_trained_models/merged_$d/eval_results.json" ] && smd="OK"
        [ -f "$REPO_ROOT/data_trained_models/merged_$d/eval_results_dm.json" ] && dm="OK"
        printf "%-15s %-5s %-5s\n" "$tag" "$smd" "$dm"
    done
    exit 0
fi

# ------------------------------------------------------------------ lock ---
if [ -f "$LOCK_FILE" ]; then
    OLD_PID=$(cat "$LOCK_FILE" 2>/dev/null)
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[$(date '+%F %T')] Already running as PID $OLD_PID -- exiting."
        exit 0
    fi
    echo "[$(date '+%F %T')] Removing stale lock (PID $OLD_PID is gone)."
fi
echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

# ------------------------------------------------------------------- run ---
# $1 = "smd" | "dm"
run_stage() {
    local mode="$1" flag="" suffix=""
    if [ "$mode" = "dm" ]; then flag="--no-projection"; suffix="_dm"; fi

    echo
    echo "=============================================================="
    echo " STAGE: $mode   ($(date '+%F %T'))"
    echo "=============================================================="

    local running=0
    for tag in "${SCENARIOS[@]}"; do
        local mid
        mid=$(python3 -c "
from scenarios import get_scenario
print(get_scenario('$tag')['model_id'])" 2>/dev/null)
        if [ -z "$mid" ]; then
            echo "  [$tag] unknown scenario, skipping"; continue
        fi

        local ckpt_dir="$REPO_ROOT/data_trained_models/merged_$mid"
        if [ ! -d "$ckpt_dir/checkpoints" ]; then
            echo "  [$tag] no trained checkpoint -- skipping ($mode)"; continue
        fi
        if [ -f "$ckpt_dir/eval_results${suffix}.json" ]; then
            echo "  [$tag] $mode already done"; continue
        fi

        echo "  [$tag] launching $mode"
        nohup python3 -u evaluate_smd_merged_all.py "$tag" $flag \
            > "$LOG_DIR/${mode}_${tag}.log" 2>&1 &

        running=$((running + 1))
        if [ "$running" -ge "$PARALLEL" ]; then
            wait -n 2>/dev/null || wait
            running=$((running - 1))
        fi
    done
    wait
    echo " STAGE $mode complete ($(date '+%F %T'))"
}

run_stage smd
run_stage dm

# -------------------------------------------------------------- compile ---
echo
echo "=============================================================="
echo " Compiling Tables 3-6"
echo "=============================================================="
python3 compile_tables_3_6.py | tee "$REPO_ROOT/TABLES_3_6.txt"

echo
echo "Done. Table text is in $REPO_ROOT/TABLES_3_6.txt"
echo "Nothing has been written to the paper -- review first, then run"
echo "  python3 $SCRIPT_DIR/compile_smd_paper_table.py"
echo "if you want the SMD appendix table refreshed."
