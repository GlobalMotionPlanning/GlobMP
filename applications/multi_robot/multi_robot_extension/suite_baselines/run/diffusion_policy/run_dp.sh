#!/bin/bash
# Diffusion-Policy (vanilla, unguided) baseline — ONE cell on OUR inflated geometry.
# usage: run_dp.sh <map> <n_agents> <n_inits> <n_samples_per_agent> <gpu> <cpus> <tag>
set -u
MAP=$1; NA=$2; NI=$3; NS=$4; GPU=$5; CPUS=$6; TAG=${7:-t}
declare -A OBS=( [empty]=1.0 [basic]=2.2 [dense]=1.0 [corridor]=1.5 [room]=1.45 \
                 [shelf]=1.6 [corridorbay]=1.15 [crossroom]=1.1 )
C="$MRMP_ROOT/run/diffusion_policy"; W="${DP_WORK:?source config/env.sh first}"; mkdir -p "$W/configs"
MR=${MRMP_ROOT:?source config/env.sh first}/DiRecT/multi_robot
PY=${PY_DIRECT:?source config/env.sh first}
CFG=$W/configs/dp_${MAP}_a${NA}_i${NI}_s${NS}_${DP_MODELS_DIR:-shared_trained}.yaml
$PY $C/make_cfg.py $MAP $NA $NI $NS $CFG || exit 1

declare -A RR=( [empty]=0.10 )        # empty has no obstacles: its difficulty axis is the robot radius
export OBS_SCALE=${OBS[$MAP]}
export ROBOT_R=${RR[$MAP]:-0.05}      # honoured by mmd/config/mmd_params.py -> RobotPlanarDisk.radius
# Only when the caller has not already chosen one. grid.py exports the GPU it
# assigned and passes 0 here meaning "the first visible device"; re-exporting 0
# re-read that as PHYSICAL GPU 0, so every cell piled onto one card.
: "${CUDA_VISIBLE_DEVICES:=$GPU}"
export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
export PYTHONPATH=$MR/deps/torch_robotics   # ONLY this copy has compute_traj_data_adherence_batch

# --- prove OBS_SCALE actually took effect (compare vs 1.0 on the same class) ---
$PY - <<PYEOF
import os, torch
from torch_robotics.environments import *
import yaml
cfg=yaml.safe_load(open("$CFG"))
cls=cfg['environment']['start_goal_config']['env_class_for_random']
ta={'device':'cpu','dtype':torch.float32}
xs=torch.linspace(-1,1,129); X,Y=torch.meshgrid(xs,xs,indexing='ij')
p=torch.stack([X.reshape(-1),Y.reshape(-1)],-1)
def occ(c):
    e=eval(c)(tensor_args=ta); m=torch.zeros(p.shape[0],dtype=torch.bool)
    for f in (e.obj_fixed_list or []): m |= (f.compute_signed_distance(p).reshape(-1)<0)
    return m.float().mean().item()
now=occ(cls)
os.environ['OBS_SCALE']='1.0'
import importlib, torch_robotics.environments as E
print(f"[GEOM CHECK] {cls}  OBS_SCALE={os.environ.get('OBS_SCALE_ORIG','$OBS_SCALE')}  occupied_fraction={now:.4f}")
PYEOF

echo "=== RUN map=$MAP OBS_SCALE=$OBS_SCALE ROBOT_R=$ROBOT_R agents=$NA n_inits=$NI n_samples/agent=$NS gpu=$GPU cpus=$CPUS"
/usr/bin/time -v taskset -c $CPUS $PY $MR/scripts/inference/inference_pcdiff.py \
    --config $CFG -m "dp_${MAP}_a${NA}_i${NI}_s${NS}_${TAG}" 2>&1
