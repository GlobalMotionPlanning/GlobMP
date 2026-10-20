# Sourced by every script: machine paths (paths.env, not in git) + the paper's settings (recipe.env).
AB_CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AB_PATHS=${AB_PATHS:-$AB_CODE/config/paths.env}
[ -f "$AB_PATHS" ] || { echo "missing $AB_PATHS (copy config/paths.env.example)" >&2; exit 1; }
set -a; . "$AB_PATHS"; . "$AB_CODE/config/recipe.env"; set +a
export AB_CODE DIRECT_DIR=$AB_CODE/DiRecT PYTHONNOUSERSITE=1 WANDB_MODE=disabled
export MUJOCO_GL=${MUJOCO_GL:-egl}
AB_RUNS=$AB_ROOT/runs; mkdir -p "$AB_RUNS"; export AB_RUNS
ab_policy(){ awk -F'\t' -v p="$1" '$1==p' "$AB_CODE/config/policies.tsv"; }
ab_log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$AB_RUNS/jobs.log" >&2; }
