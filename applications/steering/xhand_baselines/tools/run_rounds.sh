#!/usr/bin/env bash
# run_rounds.sh <taskA> <taskB> [method[:tactile] ...]
# Runs one method on both tasks at a time (taskA on GPUs 0-3, taskB on GPUs 4-7), the next method only
# after both jobs have finished. A failed job is logged and the queue continues.
# Default order (fastest first): groot groot:t pi05 pi05:t act act:t dp dp:t trex trex:t
. "$(dirname "$0")/../config/env.sh"
A=$1; B=$2; shift 2
ORDER=${*:-groot groot:t pi05 pi05:t act act:t dp dp:t trex trex:t}
F=$XB_CODE/finetune; GA=0,1,2,3; GB=4,5,6,7
job(){ local task=$1 gpus=$2 m=${3%%:*} tac=""; [[ $3 == *:t ]] && tac=--tactile
  if [ $m = trex ]; then bash $F/finetune_trex.sh $task $gpus $tac $((29500 + ${gpus%%,*}))
  else bash $F/run_baseline.sh $m $task $gpus $tac; fi; }
xb_log "rounds start: $A (GPUs $GA) + $B (GPUs $GB): $ORDER"
for r in $ORDER; do
  xb_log "round $r"
  job $A $GA $r & pa=$!
  job $B $GB $r & pb=$!
  wait $pa; ra=$?; wait $pb; rb=$?
  [ $ra = 0 ] || xb_log "round $r: $A failed (exit $ra)"
  [ $rb = 0 ] || xb_log "round $r: $B failed (exit $rb)"
done
xb_log "rounds done"
