#!/bin/bash
# run_cell.sh <map> <N> <gpu> : GlobMP as the projector of the diffusion sampler (DiRecT, Robotarium fork) on one
# (map, N) cell of the Robotarium setting: 10 instances x 64 candidates, the baselines' own sampler, models and
# instances, one fixed GlobMP configuration.  GlobMP's per-instance certificate (the candidates it certified at the
# final call) is appended to $OUT/cert${EXP_TAG}_<map>_N<N>.jsonl; select_and_score.py returns the first certified one.
#
# Environment (paths have no defaults inside the benchmark; set them):
#   DIRECT_MR   multi_robot/ of the benchmark's Robotarium DiRecT fork, with the GPA hook (direct_hook.py) applied
#               and the Robotarium-trained single-robot models in data_trained_models/
#   DIRECT_PY   python of the benchmark's `direct` conda env
#   BENCH_DIR   directory with instances.npz and maps.json   (default: ../data/robotarium)
#   OUT         output directory for certificates and timing (default: ./robotarium_out)
#   INST_NPZ    instance file override (chunks of a cell, single-instance retries); NINITS instances in it (10)
#   SEED        diffusion seed (18; retries use 19, 20, ...); EXP_TAG suffix of the run name
#   GPA_WORKERS, GPA_DEVICE, GPA_BATCH_THREADS, GPA_PRIO_TEAM_MAX: compute only (40, cuda, 4, 900)
set -e
mp=$1; n=$2; gpu=$3
HERE=$(cd "$(dirname "$0")" && pwd); MR=$(dirname "$HERE"); GLOBMP_ROOT=$(dirname "$(dirname "$(dirname "$MR")")")
: "${DIRECT_MR:?set DIRECT_MR}" "${DIRECT_PY:?set DIRECT_PY}"
B=${BENCH_DIR:-$MR/data/robotarium}; INST=${INST_NPZ:-$B/instances.npz}; OUT=${OUT:-$PWD/robotarium_out}; mkdir -p "$OUT"
declare -A ENV=([empty]=empty [corridor]=conveyor [crossroom]=crossroom [shelf]=dropregion [corridorbay]=corridorsmd)
declare -A CLS=([empty]=EnvEmptyNoWait2D [corridor]=EnvConveyor2D [crossroom]=EnvCrossRoom2D [shelf]=EnvDropRegion2D [corridorbay]=EnvCorridorSMD2D)
S=$(python3 -c "import json;print(json.load(open('$B/maps.json'))['maps']['$mp']['obs_scale'])")
CERT=$OUT/cert${EXP_TAG}_${mp}_N${n}.jsonl; rm -f "$CERT"
cd "$DIRECT_MR"
[ "$mp" = corridor ] && export RBT_WIDE=1 || unset RBT_WIDE          # Conveyor: the wide-channel geometry
# benchmark geometry: arena 3.2 x 2.0 m, isotropic normalizer, robot radius 0.13 m, per-map obstacle scale
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$gpu RBT_ARENA=1 RBT_ISO=1 RBT_PORT=1 ROBOT_R=0.13 OBS_SCALE=$S \
  PYTHONPATH=deps/torch_robotics:$GLOBMP_ROOT:$HERE GLOBMP_MR=$MR \
  SHARED_INSTANCES_NPZ=$INST SHARED_INSTANCES_KEY=${mp}rbt/N$n OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# GlobMP configuration of the paper's runs: 25 projection calls per instance; intermediate calls steer (10 s,
# 2000 restoration steps, keep-right and circulation shifts 0.08); the final call solves every candidate team
# jointly (240 s, margin 0.019), then certifies one robot at a time if none certified, stopping at the first
export GPA_BATCH=1 GPA_NCALLS=25 GPA_VERBOSE=1 GPA_MARGIN=0.019 GPA_INTER_BUDGET=10 GPA_INTER_CAP=2000 GPA_TEAMBUDGET=240 \
  GPA_RIGHTHAND=0.08 GPA_CIRC=0.08 GPA_CERT_LOG=$CERT GPA_PRIORITIZED=1 GPA_FIRST_CERT=1 \
  GPA_WORKERS=${GPA_WORKERS:-40} GPA_DEVICE=${GPA_DEVICE:-cuda} GPA_BATCH_THREADS=${GPA_BATCH_THREADS:-4}
exec "$DIRECT_PY" scripts/inference/run_direct_cell.py \
  --env ${ENV[$mp]} --env-class ${CLS[$mp]} --model-id ${CLS[$mp]}-RobotPlanarDisk --agents $n --vmax 1.2 \
  --n-inits ${NINITS:-10} --n-samples 64 --obs-scale $S --robot-r 0.13 --gpu-id 0 --projector-type GPA --seed ${SEED:-18} --proj-start 1.0 \
  --exp-name v2_GPA${EXP_TAG}_${mp}_N$n --timing-json $OUT/timing${EXP_TAG}_${mp}_N$n.json
