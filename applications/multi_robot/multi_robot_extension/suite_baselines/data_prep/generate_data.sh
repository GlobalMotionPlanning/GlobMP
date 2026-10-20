#!/bin/bash
# generate_data.sh : single-robot demonstrations, one set per environment class (about 1 h, CPU).
#
# Each set is generated IN the geometry that environment is evaluated in: scenarios.HARD_OBS_SCALE (SMD/scripts/repro)
# is the single source of the per-map obstacle scale, and eval/check_artifact_scales.py compares what each directory
# was built at against it (an earlier copy of the table drifted: Drop-Region trained at 1.15, evaluated at 1.60).
# CPU on purpose: the work is geometry on 64x2 tensors and is ~60x slower behind a busy GPU (2.8 vs 176 contexts/s).
# The planners rebuild their normalizer from these demonstrations at start-up, so this stage is needed even with the
# shipped checkpoints. Output: SMD/data_trajectories/singleagent_<map>hard_<EnvClass>-RobotPlanarDisk
. "$(dirname "$0")/../config/env.sh"
set -eu
cd "$MRMP_ROOT/SMD/scripts/repro"
LOG="$GRID_LOGDIR/gen"; mkdir -p "$LOG"
for mp in empty basic dense corridor room shelf corridorbay crossroom; do
  OBS=$("$PY_SMD" -c "import sys;sys.path.insert(0,'.');from scenarios import HARD_OBS_SCALE;print(HARD_OBS_SCALE['$mp'])")
  RR=$("$PY_SMD" -c "import sys;sys.path.insert(0,'.');from scenarios import HARD_ROBOT_R;print(HARD_ROBOT_R.get('$mp',0.05))")
  sb_log "generate $mp at obstacle scale $OBS, robot radius $RR"
  OBS_SCALE=$OBS ROBOT_R=$RR CUDA_VISIBLE_DEVICES="" \
    "$PY_SMD" -u generate_singleagent_data.py "${mp}hard" > "$LOG/$mp.log" 2>&1
  d=$(ls -d "$MRMP_ROOT/SMD/data_trajectories/singleagent_${mp}hard_"*)   # expand now: an unexpanded glob in the redirect below fails on a fresh machine
  printf '%s\n' "$OBS" > $d/.obs_scale        # what this data was actually built at
  tail -1 "$LOG/$mp.log"
done
sb_log "demonstrations done; next: tools/wire_artifacts.sh (or train/train_models.sh to retrain the models)"
