#!/bin/bash
# run_sweep.sh [out_dir] : GlobMP on the simulated suite, the paper's sweep of all 168 cells
# (8 maps x N in 21..3 x caps low/medium/high) with run_grid.py at its defaults (the paper's settings).
# Cells are dealt round-robin to the core groups; each group runs its cells one at a time on its pinned cores.
# A cell whose output file exists is skipped, so the sweep can be restarted.
#
#   CORE_GROUPS  taskset core lists, one per group (default: eight groups of four cores, 0-3 ... 28-31)
#   MAPS, NS, CAPS  restrict the sweep (defaults: all)
#   PY           python with numpy, scipy, torch (default: python)
#   DRY=1        print the commands without running them
set -e
HERE=$(cd "$(dirname "$0")" && pwd); EXT=$(dirname "$HERE"); ROOT=$(dirname "$(dirname "$(dirname "$EXT")")")
OUT=${1:-$PWD/out}
PY=${PY:-python}
CORE_GROUPS=${CORE_GROUPS:-"0-3 4-7 8-11 12-15 16-19 20-23 24-27 28-31"}
MAPS=${MAPS:-"crossroom empty basic corridor corridorbay room shelf dense"}
NS=${NS:-"21 18 15 12 9 6 3"}
CAPS=${CAPS:-"low medium high"}
export PYTHONPATH=$ROOT${PYTHONPATH:+:$PYTHONPATH} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
[ -n "$DRY" ] || mkdir -p "$OUT"

cells=()
for m in $MAPS; do for n in $NS; do for r in $CAPS; do cells+=("$m $n $r"); done; done; done
read -r -a groups <<< "$CORE_GROUPS"
G=${#groups[@]}

run_group() {
  local g=$1 i m n r out
  for ((i = g; i < ${#cells[@]}; i += G)); do
    read -r m n r <<< "${cells[$i]}"
    out=$OUT/${m}__N${n}__${r}.npz
    [ -e "$out" ] && continue
    if [ -n "$DRY" ]; then
      echo "taskset -c ${groups[$g]} $PY $EXT/run_grid.py $m $n $r $out"
    else
      taskset -c "${groups[$g]}" "$PY" "$EXT/run_grid.py" "$m" "$n" "$r" "$out" > "$OUT/${m}__N${n}__${r}.log" 2>&1 \
        || echo "cell $m N=$n $r failed (see $OUT/${m}__N${n}__${r}.log)"
    fi
  done
}

for ((g = 0; g < G; g++)); do
  if [ -n "$DRY" ]; then run_group $g; else run_group $g & fi
done
wait
