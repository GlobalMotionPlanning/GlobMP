#!/usr/bin/env bash
# convert_trex.sh <task>: T-Rex 42-D datasets (end-effector pose in chunk-start frame + hand joints),
# train and held-out; held-out is normalised with the training stats. Uses the split from prepare_task.sh.
. "$(dirname "$0")/../config/env.sh"
IFS=$'\t' read -r T SESSION SPLIT INSTR _ <<< "$(xb_task "$1")"; [ -n "$T" ] || { echo "unknown task $1"; exit 1; }
D=$XB_ROOT/data/$T; cd "$TREX_DIR"; export PYTHONPATH=$TREX_DIR
for p in train heldout; do
  O=$D/trex42_$p; [ -f $O/meta/trex_norm_stats.json ] && continue
  "$XB_PY" -u utils/convert_xhand_to_lerobot.py --src "$XB_RAW/$SESSION" --output_root $O \
    --episodes_json $D/split/$p.json --instruction "$INSTR" $( [ $p = heldout ] && echo --stats_from $D/trex42_train ) \
    > $D/logs/convert_trex42_$p.log 2>&1 || { echo "T-Rex conversion failed ($p): $D/logs/convert_trex42_$p.log"; exit 1; }
done
echo "T-Rex data ready: $D/trex42_{train,heldout}"
