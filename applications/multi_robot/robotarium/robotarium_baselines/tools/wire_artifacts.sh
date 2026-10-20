#!/bin/bash
# wire_artifacts.sh : link the demonstrations (data_prep/generate_data.sh) and the models ($RB_MODELS, default
# checkpoints/) into the forks, under the names the models' args.yaml use.
. "$(dirname "$0")/../config/env.sh"
set -eu
R=$MRMP_ROOT
link(){ mkdir -p "$(dirname "$2")"
        [ -e "$2" ] && [ ! -L "$2" ] && { echo "  $2 exists and is not a link (left by an earlier run?): move it away"; exit 1; }
        ln -sfn "$1" "$2"; }
for mp in empty crossroom shelf corridorbay corridor; do
  c=$(rb_map $mp env_class)-RobotPlanarDisk; s=${mp}rbt; [ $mp = corridor ] && s=corridorwiderbt
  d=$R/SMD/data_trajectories/singleagent_${s}_$c
  [ -d "$d" ] || { echo "  missing $d -- run data_prep/generate_data.sh"; exit 1; }
  link "$d" "$R/DiRecT/data_trajectories/singleagent_${s}_$c"       # DiRecT, CD, Final Proj., MMD-CBS
  link "$d" "$R/PCD/data_trajectories/singleagent_${s}_$c"          # PCD
  m=$RB_MODELS/$c; [ -d "$m" ] || m=$RB_CODE/checkpoints/$c          # maps not retrained: the shipped model
  [ -d "$m" ] || { echo "  missing model $m"; exit 1; }
  link "$m" "$RB_RUNS/models_flat/$c"
  link "$m" "$R/DiRecT/data_trained_models/$c"
  link "$m" "$R/PCD/multi_robot/data_trained_models/$c"
done
mkdir -p "$R/DiRecT/results/000event_logs" "$R/PCD/multi_robot/results/000event_logs"
rb_log "wired: 5 maps, models from $RB_MODELS"
