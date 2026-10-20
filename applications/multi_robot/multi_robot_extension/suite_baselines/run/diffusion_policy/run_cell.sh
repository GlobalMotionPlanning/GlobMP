#!/bin/bash
# diffusion_policy (Diffuser, vanilla/unguided, NO projection) — ONE benchmark cell
# on OUR inflated geometry, against the in-distribution shared single-agent models.
#
# usage: run_cell.sh <map> <n_agents> <n_inits> <n_samples_per_agent> <gpu> <cpus> <tag> [models_dir]
set -eu
MAP=$1; NA=$2; NI=$3; NS=$4; GPU=$5; CPUS=$6; TAG=${7:-t}; MODELS=${8:-shared_trained}

# --- geometry: OBS_SCALE is the difficulty axis for obstacle maps, ROBOT_R for empty ---
declare -A OBS=( [empty]=1.0 [basic]=2.2 [dense]=1.0 [corridor]=1.5 [room]=1.45 \
                 [shelf]=1.6 [corridorbay]=1.15 [crossroom]=1.1 )
declare -A RR=(  [empty]=0.10 )            # empty has no obstacles -> radius is its axis
declare -A ECLS=( [empty]=EnvEmptyNoWait2D [basic]=EnvEmptyNoWait2D [dense]=EnvEmptyNoWait2D \
                  [corridor]=EnvConveyor2D [room]=EnvHighways2D [shelf]=EnvDropRegion2D \
                  [corridorbay]=EnvCorridorSMD2D [crossroom]=EnvCrossRoom2D )

C="$MRMP_ROOT/run/diffusion_policy"; W="${DP_WORK:?source config/env.sh first}"; mkdir -p "$W/configs"
MR=${MRMP_ROOT:?source config/env.sh first}/DiRecT/multi_robot
PY=${PY_DIRECT:?source config/env.sh first}
CFG=$W/configs/cell_${MAP}_a${NA}_i${NI}_s${NS}_${MODELS}.yaml

export OBS_SCALE=${OBS[$MAP]}
export ROBOT_R=${RR[$MAP]:-0.05}
# Only when the caller has not already chosen one. grid.py exports the GPU it
# assigned and passes 0 here meaning "the first visible device"; re-exporting 0
# re-read that as PHYSICAL GPU 0, so every cell piled onto one card.
: "${CUDA_VISIBLE_DEVICES:=$GPU}"
export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
# ONLY the DiRecT copy of torch_robotics has compute_traj_data_adherence_batch;
# the default import resolves to MDOC/deps/torch_robotics, which lacks it.
export PYTHONPATH=$MR/deps/torch_robotics

DP_MODELS_DIR=$MODELS $PY $C/make_cfg.py $MAP $NA $NI $NS $CFG

# --- PROVE OBS_SCALE/ROBOT_R took effect: rebuild the same env class in a fresh
#     subprocess at 1.0 and compare occupied fraction. Not a no-op check. ---
echo "[geom] verifying ${ECLS[$MAP]} ..."
NOM=$(OBS_SCALE=1.0 ROBOT_R=0.05 $PY $C/geomcheck.py ${ECLS[$MAP]} 2>/dev/null | tail -1)
HARD=$(OBS_SCALE=$OBS_SCALE ROBOT_R=$ROBOT_R $PY $C/geomcheck.py ${ECLS[$MAP]} 2>/dev/null | tail -1)
echo "[geom] nominal : $NOM"
echo "[geom] THIS RUN: $HARD"
if [ "$(echo "$NOM" | grep -o 'occupied_frac=[0-9.]*')" = "$(echo "$HARD" | grep -o 'occupied_frac=[0-9.]*')" ] \
   && [ "$(echo "$NOM" | grep -o 'robot_radius=[0-9.]*')" = "$(echo "$HARD" | grep -o 'robot_radius=[0-9.]*')" ]; then
  echo "[geom] !! FATAL: geometry identical to nominal -- difficulty axis did NOT apply"; exit 2
fi
echo "[geom] OK: geometry differs from nominal"

echo "=== RUN map=$MAP OBS_SCALE=$OBS_SCALE ROBOT_R=$ROBOT_R agents=$NA n_inits=$NI n_samples/agent=$NS models=$MODELS gpu=$GPU cpus=$CPUS"
/usr/bin/time -v taskset -c $CPUS $PY $MR/scripts/inference/inference_pcdiff.py \
    --config $CFG -m "cell_${MAP}_a${NA}_i${NI}_s${NS}_${TAG}" 2>&1
