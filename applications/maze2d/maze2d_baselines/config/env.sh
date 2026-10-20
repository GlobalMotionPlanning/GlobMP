# Sourced by every script: machine paths (paths.env, not in git) + the paper's settings (recipe.env).
MB_CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MB_PATHS=${MB_PATHS:-$MB_CODE/config/paths.env}   # override to point at another root (e.g. a test root)
[ -f "$MB_PATHS" ] || { echo "missing $MB_PATHS (copy config/paths.env.example)" >&2; exit 1; }
set -a; . "$MB_PATHS"; . "$MB_CODE/config/recipe.env"; set +a
export MB_CODE DIRECT_DIR=$MB_CODE/DiRecT PYTHONNOUSERSITE=1 WANDB_MODE=disabled MUJOCO_GL=${MUJOCO_GL:-egl}
export D4RL_DATASET_DIR=${D4RL_DATASET_DIR:-$HOME/.d4rl/datasets}
MB_RUNS=$MB_ROOT/runs; mkdir -p "$MB_RUNS"; export MB_RUNS
mb_policy(){ awk -F'\t' -v p="$1" '$1==p' "$MB_CODE/config/policies.tsv"; }   # id  name in the paper
mb_log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MB_RUNS/jobs.log" >&2; }
