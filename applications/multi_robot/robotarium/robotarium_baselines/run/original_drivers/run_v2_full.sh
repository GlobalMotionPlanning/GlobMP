#!/bin/bash
SP=<tmp>/f915b73d-2815-4d04-99b9-6801f3fcf24f/scratchpad
L=<path>
DM=$L/repro/rbt/DiRecT-main/multi_robot
PCD=$L/repro/rbt/pcd/multi_robot
MMDT=$L/repro/rbtB/DiRecT-main/multi_robot
PYD=$L/miniconda3/envs/direct/bin/python
PYS=$L/miniconda3/envs/smd/bin/python
SHARED=$L/repro/baselines/models_flat_rbt
NPZ=<path>/rbt_sim/instances_v2mix.npz
OUT=$SP/v2full_out; mkdir -p $OUT
declare -A ECLS=( [crossroom]=EnvCrossRoom2D [shelf]=EnvDropRegion2D [corridorbay]=EnvCorridorSMD2D )
declare -A DENV=( [crossroom]=crossroom [shelf]=dropregion [corridorbay]=corridorsmd )
declare -A OBS=( [crossroom]=0.85 [shelf]=0.4 [corridorbay]=0.2 )
run() {
  local M=$1 mp=$2 N=$3 g=$4
  local ec=${ECLS[$mp]} de=${DENV[$mp]} ob=${OBS[$mp]}
  local base="env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$g RBT_PORT=1 RBT_ARENA=1 RBT_ISO=1 ROBOT_R=0.13 OBS_SCALE=$ob SHARED_INSTANCES_NPZ=$NPZ SHARED_INSTANCES_KEY=${mp}rbt/N${N} MODELS_ROOT=$SHARED"
  local oj=$OUT/${M}_${mp}_N${N}.json lg=$OUT/${M}_${mp}_N${N}.log
  case $M in
    direct) (cd $DM && $base PYTHONPATH=$DM/deps/torch_robotics $PYD scripts/inference/run_direct_cell.py --env $de --env-class $ec --model-id $ec-RobotPlanarDisk --agents $N --vmax 1.2 --n-inits 10 --n-samples 64 --obs-scale $ob --robot-r 0.13 --gpu-id 0 --timing-json $oj --exp-name v2_direct_${mp}_N${N} > $lg 2>&1) ;;
    cd) (cd $DM && $base PYTHONPATH=$DM/deps/torch_robotics $PYD scripts/inference/run_cd_pdm_cell.py --method cd --map $mp --agents $N --vmax 1.2 --n-inits 10 --n-samples 64 --obs-scale $ob --robot-r 0.13 --gpu-id 0 --timing-json $oj > $lg 2>&1) ;;
    fp) (cd $DM && $base PYTHONPATH=$DM/deps/torch_robotics $PYD scripts/inference/run_finalproj_cell.py --map $mp --agents $N --vmax 1.2 --n-inits 10 --n-samples $((64*N)) --models-root $SHARED --gpu-id 0 --obs-scale $ob --timing-json $oj > $lg 2>&1) ;;
    pcd_shd) (cd $PCD && $base PYTHONPATH=$PCD/deps/torch_robotics $PYD scripts/inference/run_pcd_cell.py --map $mp --n $N --instances 10 --samples 64 --cost hinge_sqr_l2 --vreg 0 --obs-scale $ob --robot-r 0.13 --out $oj > $lg 2>&1) ;;
    pcd_lb) (cd $PCD && $base PYTHONPATH=$PCD/deps/torch_robotics $PYD scripts/inference/run_pcd_cell.py --map $mp --n $N --instances 10 --samples 64 --cost sum_log_l2 --vreg 0 --obs-scale $ob --robot-r 0.13 --out $oj > $lg 2>&1) ;;
    mmd) (cd $MMDT && $base OBSTACLE_MARGIN=0.12 INTER_AGENT_MARGIN=0.16 $PYS scripts/inference/run_mmd_cbs_cell.py --map $mp --n-agents $N --n-instances 10 --n-samples $((64*N)) --runtime-limit 60 --models-root $SHARED --out $oj > $lg 2>&1) ;;
  esac
  echo "[$(date +%H:%M:%S)] done $M $mp N$N rc=$? -> $([ -f $oj ] && echo JSON || echo NOJSON)"
}
export -f run
i=0
for M in direct cd fp pcd_shd pcd_lb mmd; do
  for mp in crossroom shelf corridorbay; do
    for N in 3 6 9 12; do
      g=$((i % 8)); i=$((i+1)); run $M $mp $N $g &
      [ $((i % 8)) -eq 0 ] && wait
    done
  done
done
wait
echo "V2FULL COMPLETE $(date +%H:%M:%S)"
