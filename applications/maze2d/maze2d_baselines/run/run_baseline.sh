#!/usr/bin/env bash
# run_baseline.sh <policy> <broad|narrow> [gpu] [repeats]   (policies: config/policies.tsv)
# One baseline on one setting with the shared planner (MB_WEIGHTS) and the fitted dynamics; saves the planned and the
# PD-tracked trajectories (config/config_traj.yaml: eval.save_trajectories on, which has no command-line flag).
# Output: runs/benchmark-dyn/<setting>/<policy>/eval/<run>/{trajectories.npz, statistics.csv, summary_statistics.txt}
. "$(dirname "$0")/../config/env.sh"
P=$1; S=$2; GPU=${3:-0}; R=${4:-$EVAL_REPEATS}
[ -n "$(mb_policy "$P")" ] || { echo "unknown policy $P (config/policies.tsv)" >&2; exit 1; }
[ -f "$MB_RUNS/dynamics/linear_model.npz" ] || { echo "no dynamics model: run data_prep/fit_dynamics.sh" >&2; exit 1; }
[ -f "$MB_WEIGHTS" ] || { echo "planner weights not found: $MB_WEIGHTS" >&2; exit 1; }
cd "$DIRECT_DIR" && mb_log "$S/$P gpu $GPU ($R repeats)" && \
  CUDA_VISIBLE_DEVICES=$GPU "$MB_PY" eval.py --config "$MB_CODE/config/config_traj.yaml" --policy "$P" --obstacle-type "$S" \
    --weights-path "$MB_WEIGHTS" --dynamics-path "$MB_RUNS/dynamics/linear_model.npz" --exp-name "benchmark-dyn/$S/$P" \
    --num-random-repeats "$R" --seed "$SEED" --device cuda --log-folder "$MB_RUNS"
