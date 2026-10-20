#!/usr/bin/env bash
# prepare_task.sh <task>    (task ids: config/tasks.tsv)
# Raw session (read-only, $XB_RAW) -> checks -> gate -> LeRobot v3 + GR00T + stats -> split -> T-Rex 42-D.
# Writes $XB_ROOT/data/<task>/{checks,lerobot,gr00t,split,trex42_train,trex42_heldout,logs}. Idempotent.
. "$(dirname "$0")/../config/env.sh"
IFS=$'\t' read -r T SESSION SPLIT INSTR _ <<< "$(xb_task "$1")"; [ -n "$T" ] || { echo "unknown task $1"; exit 1; }
RAW=$XB_RAW/$SESSION; D=$XB_ROOT/data/$T; H=$XB_CODE/data_prep
[ -f $D/READY ] && { echo "$T already prepared"; exit 0; }
mkdir -p $D/checks $D/split $D/logs
if [ ! -s $D/checks/h5.json ]; then
  "$XB_PY" $H/check_h5.py $RAW $D/checks/h5.json > $D/logs/check_h5.log 2>&1
  bash $H/check_videos.sh $RAW $D/checks/videos.txt 16 > $D/logs/check_videos.log 2>&1
fi
"$XB_PY" $H/gate.py $D/checks | tee $D/checks/gate.txt | head -1 | grep -q ^PASS || { touch $D/FAILED; echo "$T: gate FAILED, see $D/checks/gate.txt"; exit 1; }
if [ ! -f $D/gr00t/meta/modality.json ]; then
  SRC=$RAW OUT=$D/lerobot TASK="$INSTR" "$XB_PY" $H/convert_lerobot.py > $D/logs/convert.log 2>&1 \
  && OUT=$D/lerobot "$XB_PY" $H/make_stats.py >> $D/logs/convert.log 2>&1 \
  && SRC=$D/lerobot OUT=$D/gr00t "$XB_PY" $H/convert_gr00t.py >> $D/logs/convert.log 2>&1 \
  || { echo "$T: conversion failed, see $D/logs/convert.log"; exit 1; }
fi
"$XB_PY" $H/split.py $D/lerobot $SPLIT $D/split
bash $H/convert_trex.sh $T || exit 1
touch $D/READY; echo "$T: ready ($(du -sh $D | cut -f1))"
