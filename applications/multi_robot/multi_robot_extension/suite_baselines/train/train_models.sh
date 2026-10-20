#!/bin/bash
# train_models.sh [gpu-list] : one single-robot diffusion model per environment class on generate_data.sh's data
# (about 1 h on 4 GPUs). Optional: checkpoints/ holds the models behind the paper's runs.
#
# Six classes, not eight maps: empty, basic and dense are all EnvEmptyNoWait2D and share one model (their obstacles
# are per instance and handled at plan time). Output: $SB_TRAINED/<EnvClass>-RobotPlanarDisk/0; run the baselines
# on it with SB_MODELS=$SB_TRAINED tools/wire_artifacts.sh.
. "$(dirname "$0")/../config/env.sh"
set -eu
cd "$MRMP_ROOT/SMD/scripts/repro"
GPUS="${1:-0}"; IFS=',' read -ra G <<< "$GPUS"
LOG="$GRID_LOGDIR/train"; mkdir -p "$LOG"
MAPS=(empty corridor room shelf corridorbay crossroom)     # one map per class; the class is what the model is keyed by
i=0; pids=()
for mp in "${MAPS[@]}"; do
  g=${G[$((i % ${#G[@]}))]}; i=$((i+1))
  OBS=$("$PY_SMD" -c "import sys;sys.path.insert(0,'.');from scenarios import HARD_OBS_SCALE;print(HARD_OBS_SCALE['$mp'])")
  sb_log "train $mp on gpu $g at scale $OBS"
  OBS_SCALE=$OBS ROBOT_R=0.05 CUDA_VISIBLE_DEVICES=$g \
    "$PY_SMD" -u retrain_shared_models.py "$mp" > "$LOG/$mp.log" 2>&1 &
  pids+=($!)
  # four at a time keeps memory sane on one card
  if [ ${#pids[@]} -ge $(( ${#G[@]} > 4 ? 4 : ${#G[@]} )) ]; then wait "${pids[0]}"; pids=("${pids[@]:1}"); fi
done
wait
for mp in "${MAPS[@]}"; do printf '  %-12s %s\n' "$mp" "$(tail -1 "$LOG/$mp.log")"; done
sb_log "models in $SB_TRAINED; next: SB_MODELS=$SB_TRAINED tools/wire_artifacts.sh"
