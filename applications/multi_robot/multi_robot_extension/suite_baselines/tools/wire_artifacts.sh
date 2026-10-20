#!/bin/bash
# wire_artifacts.sh : link the demonstrations and the models into each fork (after data_prep/generate_data.sh).
#
# Every fork resolves its dataset by the name in the model's args.yaml, inside ITS OWN tree (mmd/datasets/
# trajectories.py derives the data directory from its own __file__). The demonstrations are generated once, in the SMD
# tree, under names like singleagent_shelfhard_<EnvClass>-RobotPlanarDisk; each fork needs a link under the bare
# <EnvClass>-RobotPlanarDisk its args.yaml uses. Without these links a cell dies at dataset load, or silently loads a
# different map's demonstrations. Models: $SB_MODELS (default checkpoints/, the paper's).
. "$(dirname "$0")/../config/env.sh"
set -eu
R=$MRMP_ROOT; SMD="$R/SMD"
# which map's demonstrations each class is trained on (one model per class; empty, basic and dense share
# EnvEmptyNoWait2D and are told apart at plan time by their per-instance obstacles)
declare -A SRC=(
  [EnvEmptyNoWait2D]=empty [EnvConveyor2D]=corridor [EnvHighways2D]=room
  [EnvDropRegion2D]=shelf  [EnvCorridorSMD2D]=corridorbay [EnvCrossRoom2D]=crossroom
)
link(){ mkdir -p "$(dirname "$2")"
        [ -e "$2" ] && [ ! -L "$2" ] && { echo "  $2 exists and is not a link (left by an earlier run?): move it away"; exit 1; }
        ln -sfn "$1" "$2"; }

# 1. datasets, under both the SMD name and the bare class name, in every fork that loads one
for tree in "$R/DiRecT/multi_robot" "$R/PCD" "$R/MDOC"; do
  for cls in "${!SRC[@]}"; do
    mp="${SRC[$cls]}"
    d="$SMD/data_trajectories/singleagent_${mp}hard_${cls}-RobotPlanarDisk"
    [ -d "$d" ] || { echo "  missing dataset $d -- run data_prep/generate_data.sh first"; exit 1; }
    link "$d" "$tree/data_trajectories/${cls}-RobotPlanarDisk"
    link "$d" "$tree/data_trajectories/singleagent_${mp}hard_${cls}-RobotPlanarDisk"
  done
done

# 2. models. models_flat is the flattened view run/grid.py passes as --models-root: the trainer writes
#    <cls>/0/args.yaml, the inference runners expect args.yaml directly under <cls>.
for cls in "${!SRC[@]}"; do
  m="$SB_MODELS/${cls}-RobotPlanarDisk/0"
  [ -d "$m" ] || { echo "  missing model $m"; exit 1; }
  link "$m" "$SB_MODELS_FLAT/${cls}-RobotPlanarDisk"
  # DiRecT resolves <base>/data_trained_models/<cls>; results_inference stays local to the fork, so a run does not
  # write into the model directory
  t="$R/DiRecT/multi_robot/data_trained_models/${cls}-RobotPlanarDisk"
  mkdir -p "$t/results_inference"
  link "$m/args.yaml"   "$t/args.yaml"
  link "$m/checkpoints" "$t/checkpoints"
  # PCD and MDOC read <tree>/data_trained_models/<cls> directly
  link "$m" "$R/PCD/multi_robot/data_trained_models/${cls}-RobotPlanarDisk"
  link "$m" "$R/MDOC/data_trained_models/${cls}-RobotPlanarDisk"
done

# 3. the runners write per-run event logs here and expect the folder to exist
mkdir -p "$R/DiRecT/multi_robot/results/000event_logs" "$R/PCD/multi_robot/results/000event_logs"

# 4. Diffusion Policy reads <base_dir>/shared_trained/<cls>/0 itself
link "$SB_MODELS" "$DP_WORK/shared_trained"
sb_log "wired: $(find "$SB_MODELS_FLAT" -maxdepth 1 -type l | wc -l) models from $SB_MODELS, datasets in 3 forks"
