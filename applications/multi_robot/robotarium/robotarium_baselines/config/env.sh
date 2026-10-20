# Sourced by every script: machine paths (paths.env, not in git) + the Robotarium setting (maps.json).
# Every path hangs off RB_CODE (this package; MRMP_ROOT for the forks) and RB_ROOT (outputs, runs/).
RB_CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RB_PATHS=${RB_PATHS:-$RB_CODE/config/paths.env}
[ -f "$RB_PATHS" ] || { echo "missing $RB_PATHS (copy config/paths.env.example)" >&2; return 1 2>/dev/null || exit 1; }
set -a; . "$RB_PATHS"; set +a
export RB_CODE MRMP_ROOT=$RB_CODE
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
export PY_DIRECT="${PY_DIRECT:-$CONDA_BASE/envs/direct/bin/python}"   # DiRecT, CD, Final Proj., PCD
export PY_SMD="${PY_SMD:-$CONDA_BASE/envs/smd/bin/python}"            # MMD-CBS, data generation, training
export RB_ROOT=${RB_ROOT:-$RB_CODE} RB_RUNS=${RB_ROOT:-$RB_CODE}/runs
export RB_CELLS=$RB_RUNS/cells                                       # one json + log per (method, map, N)
export RB_MODELS=${RB_MODELS:-$RB_CODE/checkpoints}                  # <EnvClass>-RobotPlanarDisk/{args.yaml,checkpoints}
export RB_INSTANCES=${RB_INSTANCES:-$RB_CODE/data_prep/instances/instances_v2.npz}   # 10 per (map, N), all methods
# The Robotarium setting, at training and at test: the 3.2 x 2.0 m arena, the isotropic normalizer, r = 0.13 m
# (0.055 m GRITSBot + the margin for the holonomic-to-unicycle conversion). Per map: OBS_SCALE (config/maps.json),
# and RBT_WIDE=1 on Conveyor (the widened channels; its model is the one retrained there).
export RBT_ARENA=1 RBT_ISO=1 ROBOT_R=0.13 PYTHONNOUSERSITE=1
mkdir -p "$RB_CELLS"
rb_map(){ python3 -c "import json,sys;m=json.load(open('$RB_CODE/config/maps.json'))['maps']['$1'];print(m['$2'])"; }
rb_log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$RB_RUNS/jobs.log" >&2; }
