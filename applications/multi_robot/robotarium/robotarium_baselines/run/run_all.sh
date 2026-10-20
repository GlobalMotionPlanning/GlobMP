#!/bin/bash
# run_all.sh [gpu-list] : the 6 baselines x 5 maps x N in {3,6,9,12} (120 cells), one cell per GPU at a time.
# Final Proj. names its results after the map's tile, which Cross-Room and Drop-Region share: their cells must not run
# concurrently, so each method's maps run one after another on its GPU.
. "$(dirname "$0")/../config/env.sh"
G=(${1:-0}); G=(${G[@]//,/ }); i=0
for M in direct cd final_projection pcd_shd pcd_lb mmd_cbs; do
  g=${G[$((i % ${#G[@]}))]}; i=$((i+1))
  ( for mp in empty crossroom shelf corridorbay corridor; do for N in 3 6 9 12; do
      [ -f "$RB_CELLS/${M}__${mp}__N$N.json" ] || bash "$RB_CODE/run/run_cell.sh" $M $mp $N $g; done; done ) &
  [ $((i % ${#G[@]})) -eq 0 ] && wait
done
wait; rb_log "all cells done"
