#!/bin/bash
# Runs the full SMD regeneration pipeline (data -> PP-search demos -> merge ->
# 150k-step training -> evaluation) for whatever's not yet done in
# scenarios.py, then patches the completed SMD numbers into
# <path>/paper/text_new.tex (Appendix C-equivalent
# per-cell table + the tab:multirobot summary row). Never touches "our
# method"'s (Algorithms 1-2) numbers.
#
# Fully idempotent / resumable -- every stage checks its own prior output
# before doing work (see run_full_pipeline.py's docstring). Safe to:
#   - run this exact command again any time (after a reboot, a lost
#     connection, whatever) -- it picks up exactly where it left off
#   - run in the background and walk away:
#       nohup bash run_and_push_to_paper.sh > /tmp/smd_paper_push.log 2>&1 &
#       disown
#
# Checking progress:
#   tail -f <the log file above>, or
#   python3 compile_smd_paper_table.py --dry-run   (shows N/12 cells ready)
#
# When done, this prints "PAPER PATCHED" and the diff of what changed in
# text_new.tex. It does NOT recompile the PDF (no pdflatex on this machine) --
# compile text_new.tex wherever you normally do.

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONDA_SH="$REPO_ROOT/../miniconda3/etc/profile.d/conda.sh"

source "$CONDA_SH"
conda activate smd
cd "$SCRIPT_DIR"

# Guard against launching a second, racing pipeline (e.g. cron firing every
# 15 min while a run from an earlier tick, or a manually-launched one, is
# still going) -- the individual stage scripts (generate_via_pp.py,
# train_smd_merged_all.py, evaluate_smd_merged_all.py) don't lock their
# output files, so two concurrent writers to the same eval_results.json/
# checkpoint can corrupt it. If one's already running, just exit -- do NOT
# block-and-wait here, or repeated cron ticks pile up waiter processes that
# could all wake and race on patching the paper the moment the run finishes.
# The next cron tick after the real run completes will find nothing running
# and proceed normally.
#
# Uses a PID-file lock, NOT `pgrep -f <script name>` -- pgrep -f matches
# against every process's *full command line*, so it false-positives on
# anything that merely *mentions* these script names as text (e.g. a `cat`/
# `grep`/`tail` of a log file naming them, or -- as actually happened during
# development -- the debugging shell's own wrapper process quoting this
# script's invocation back into its own argv). A PID file storing the actual
# owning PID, checked with `kill -0`, has no such ambiguity.
LOCK_FILE="/tmp/smd_pipeline_push.lock"
if [ -f "$LOCK_FILE" ]; then
    OLD_PID=$(cat "$LOCK_FILE" 2>/dev/null)
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Another instance (PID $OLD_PID) is already"
        echo "running -- exiting without starting a duplicate. The next scheduled/manual"
        echo "run will pick up from wherever it's at."
        exit 0
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Found a stale lock file (PID $OLD_PID not"
    echo "running) -- removing it and proceeding."
fi
echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting/resuming full SMD pipeline..."
python3 -u run_full_pipeline.py
PIPELINE_STATUS=$?

if [ $PIPELINE_STATUS -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline exited with status $PIPELINE_STATUS."
    echo "This is usually a transient failure (e.g. reboot mid-run). Just rerun this"
    echo "exact script again -- every stage resumes from its own last checkpoint."
    exit $PIPELINE_STATUS
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline stages complete. Checking cell coverage..."
python3 compile_smd_paper_table.py --dry-run

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Attempting to patch the paper..."
python3 compile_smd_paper_table.py

if grep -q "tab:app-smd" <path>/paper/text_new.tex; then
    echo
    echo "================= PAPER PATCHED ================="
    echo "text_new.tex now includes tab:app-smd and the completed SMD row."
    echo "Compile text_new.tex wherever you normally do (no pdflatex here)."
else
    echo
    echo "Not all 12 cells were ready yet -- rerun this script later to pick up"
    echo "wherever data generation / training / evaluation left off."
fi
