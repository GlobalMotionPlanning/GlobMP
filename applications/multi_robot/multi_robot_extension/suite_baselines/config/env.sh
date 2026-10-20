# Sourced by every script: machine paths (paths.env, not in git) + the data-generation settings (recipe.env).
# Every path hangs off two roots: SB_CODE (this package; MRMP_ROOT for the forks) and SB_ROOT (outputs, runs/).
SB_CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SB_PATHS=${SB_PATHS:-$SB_CODE/config/paths.env}
[ -f "$SB_PATHS" ] || { echo "missing $SB_PATHS (copy config/paths.env.example)" >&2; return 1 2>/dev/null || exit 1; }
set -a; . "$SB_PATHS"; . "$SB_CODE/config/recipe.env"; set +a
export SB_CODE MRMP_ROOT=$SB_CODE
# Three interpreters, because the forks disagree about torch: MDOC targets torch 2.4 and dies under 2.13 with a
# PythonDispatcher INTERNAL ASSERT; direct also needs jax for DiRecT's SCP-ADMM projector.
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
export PY_DIRECT="${PY_DIRECT:-$CONDA_BASE/envs/direct/bin/python}"
export PY_MDOC="${PY_MDOC:-$CONDA_BASE/envs/mdoc/bin/python}"
export PY_SMD="${PY_SMD:-$CONDA_BASE/envs/smd/bin/python}"
export SB_ROOT=${SB_ROOT:-$SB_CODE} SB_RUNS=${SB_ROOT:-$SB_CODE}/runs
export SB_MODELS=${SB_MODELS:-$SB_CODE/checkpoints}      # the models the baselines load (shipped: the paper's)
export SB_TRAINED=$SB_RUNS/trained_models                 # where train/train_models.sh writes
export SB_MODELS_FLAT=$SB_RUNS/models_flat                # flattened view of SB_MODELS (tools/wire_artifacts.sh)
export GRID_STATE=${GRID_STATE:-$SB_RUNS/grid_state_shared.json} GRID_LOGDIR=${GRID_LOGDIR:-$SB_RUNS/grid_logs}
export GRID_OUT=$SB_RUNS/grid_out GRID_TRAJ=$SB_RUNS/grid_traj DP_WORK=$SB_RUNS/diffusion_policy
# The instance set every method plans: shipped, so all eight are given identical start/goal pairs.
export SHARED_INSTANCES_NPZ="${SHARED_INSTANCES_NPZ:-$SB_CODE/data_prep/instances_shared.npz}"
export PYTHONNOUSERSITE=1
mkdir -p "$GRID_LOGDIR" "$GRID_OUT" "$DP_WORK"
sb_log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$SB_RUNS/jobs.log" >&2; }
if [ -z "${SB_QUIET:-}" ]; then
  for v in PY_DIRECT PY_MDOC PY_SMD; do eval p=\$$v; [ -x "$p" ] || echo "  $v missing: $p (create the conda envs, README Setup)" >&2; done
fi
