#!/bin/bash
# MMD-CBS preliminary grid: 8 maps x 7 team sizes. No velocity axis (this method
# applies no velocity projection), so one cell per (map, team size).
# Requires the shared single-agent models at $SHARED (retrained on OUR geometry).
set -u
SB=<tmp>/17351827-0327-40c6-bf09-172d6047c948/scratchpad/mmdcbs
SHARED=${SB_MODELS:?source config/env.sh first}
PY=${PY_DIRECT:?source config/env.sh first}
GPU=${GPU:-7}; CPUS=${CPUS:-80-85}
INST=${INST:-20}; SAMP=${SAMP:-128}; RT=${RT:-60}
OUT=$SB/grid; mkdir -p $OUT $SB/models_shared

# model root: <root>/<EnvClass>-RobotPlanarDisk  ->  shared_trained/<EnvClass>-RobotPlanarDisk/0
for k in EnvEmptyNoWait2D EnvConveyor2D EnvHighways2D EnvDropRegion2D EnvCorridorSMD2D EnvCrossRoom2D; do
  ln -sfn $SHARED/$k-RobotPlanarDisk/0 $SB/models_shared/$k-RobotPlanarDisk
done

#            map          OBS_SCALE ROBOT_R
CELLS="empty:1.0:0.10 basic:2.2:0.05 dense:1.0:0.05 corridor:1.5:0.05 room:1.45:0.05 shelf:1.6:0.05 corridorbay:1.15:0.05 crossroom:1.1:0.05"
for c in $CELLS; do
  IFS=: read -r MAP SC R <<< "$c"
  echo "=== $MAP  OBS_SCALE=$SC ROBOT_R=$R ==="
  cd ${MRMP_ROOT:?source config/env.sh first}/DiRecT/multi_robot
  CUDA_VISIBLE_DEVICES=$GPU OBS_SCALE=$SC ROBOT_R=$R \
  OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
  OBSTACLE_MARGIN=${OBSTACLE_MARGIN:-0.08} INTER_AGENT_MARGIN=${INTER_AGENT_MARGIN:-0.15} \
  MODELS_ROOT=$SB/models_shared \
  taskset -c $CPUS $PY $SB/run_mmd_cbs_cell.py \
    --map $MAP --n-agents 3,6,9,12,15,18,21 --n-instances $INST \
    --n-samples $SAMP --runtime-limit $RT \
    --out $OUT/${MAP}_I${INST}_S${SAMP}_R${RT}.json > $OUT/${MAP}.log 2>&1
  echo "  -> $OUT/${MAP}_I${INST}_S${SAMP}_R${RT}.json"
done
