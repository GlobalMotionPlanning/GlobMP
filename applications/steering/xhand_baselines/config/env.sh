# Sourced by every script: machine paths (paths.env, not in git) + the frozen recipe.
XB_CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XB_PATHS=${XB_PATHS:-$XB_CODE/config/paths.env}   # override to point at another root (e.g. a test root)
[ -f "$XB_PATHS" ] || { echo "missing $XB_PATHS (copy config/paths.env.example)" >&2; exit 1; }
set -a; . "$XB_PATHS"; . "$XB_CODE/config/recipe.env"; set +a
export XB_CODE
xb_task(){ awk -F'\t' -v t="$1" '$1==t' "$XB_CODE/config/tasks.tsv"; }   # id session split instruction former_id
xb_log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$XB_ROOT/runs/jobs.log" >&2; }
