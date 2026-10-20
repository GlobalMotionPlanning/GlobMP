#!/bin/bash
# generate_data.sh [map ...] : draw a new set of single-robot demonstrations per map in the Robotarium setting (CPU,
# about 3 min per map), with the recipe of the paper's sets. The generator is not seeded, so the set differs from the
# paper's and so does the normalizer the planners rebuild from it (velocity range within 0.5% on four maps; on Conveyor
# the paper's set has a 4% wider one, which this generator does not give): to reproduce the paper's cells use
# data_prep/get_demonstrations.sh (the paper's sets); this script is for retraining (train/).
# Output: SMD/data_trajectories/singleagent_<map>rbt_<EnvClass>-RobotPlanarDisk (Conveyor: corridorwiderbt).
. "$(dirname "$0")/../config/env.sh"
set -eu
cd "$MRMP_ROOT/SMD/scripts/repro"
# the paper's recipe: 40000 start-goal contexts per map (the generator's default is 600), 16 perturbed variants each,
# except Conveyor's widened map, generated later with 32
export SMD_N_CONTEXTS=${SMD_N_CONTEXTS:-40000}
# this package's SMD fork and its torch_robotics (with the arena, radius and wide-channel gates) come first
D=$MRMP_ROOT/SMD; export PYTHONPATH=$D:$D/deps/torch_robotics:$D/deps/motion_planning_baselines:$D/deps/experiment_launcher${PYTHONPATH:+:$PYTHONPATH}
for mp in ${*:-empty crossroom shelf corridorbay corridor}; do
  s=${mp}rbt; W="SMD_K_VARIANTS=${SMD_K_VARIANTS:-16}"; [ $mp = corridor ] && { s=corridorwiderbt; W="RBT_WIDE=1 SMD_K_VARIANTS=${SMD_K_VARIANTS:-32}"; }
  rb_log "generate $s at obstacle scale $(rb_map $mp obs_scale)"
  env $W OBS_SCALE=$(rb_map $mp obs_scale) CUDA_VISIBLE_DEVICES="" "$PY_SMD" -u generate_singleagent_data.py $s > "$RB_RUNS/gen_$s.log" 2>&1
  tail -1 "$RB_RUNS/gen_$s.log"
done
