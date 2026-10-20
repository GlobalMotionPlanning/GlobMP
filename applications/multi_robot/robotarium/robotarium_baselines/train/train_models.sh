#!/bin/bash
# train_models.sh [gpu] [map ...] : one single-robot diffusion model per map (optional: checkpoints/ holds the paper's).
# Output: SMD/data_trained_models/singleagent_<scenario>_<EnvClass>-RobotPlanarDisk; run the baselines on it with
# RB_MODELS=$RB_RUNS/trained_models tools/wire_artifacts.sh (it links them there).
. "$(dirname "$0")/../config/env.sh"
set -eu
GPU=${1:-0}; shift || true
cd "$MRMP_ROOT/SMD/scripts/repro"
# this package's SMD fork and its torch_robotics (with the arena, radius and wide-channel gates) come first
D=$MRMP_ROOT/SMD; export PYTHONPATH=$D:$D/deps/torch_robotics:$D/deps/motion_planning_baselines:$D/deps/experiment_launcher${PYTHONPATH:+:$PYTHONPATH}
for mp in ${*:-empty crossroom shelf corridorbay corridor}; do
  s=${mp}rbt; W=""; [ $mp = corridor ] && { s=corridorwiderbt; W="RBT_WIDE=1"; }
  rb_log "train $s on gpu $GPU"
  env $W OBS_SCALE=$(rb_map $mp obs_scale) CUDA_VISIBLE_DEVICES=$GPU "$PY_SMD" -u train_singleagent_all.py $s > "$RB_RUNS/train_$s.log" 2>&1
  c=$(rb_map $mp env_class); mkdir -p "$RB_RUNS/trained_models"
  ln -sfn "$MRMP_ROOT/SMD/data_trained_models/singleagent_${s}_${c}-RobotPlanarDisk" "$RB_RUNS/trained_models/${c}-RobotPlanarDisk"
done
