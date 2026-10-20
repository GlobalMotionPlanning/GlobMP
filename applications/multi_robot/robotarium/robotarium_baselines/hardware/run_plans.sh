#!/bin/bash
# run_plans.sh [gpu-list] : the runs behind the hardware plans (plans/), with this package's forks and the settings they
# were made with: N = 12 on the antipodal ring (instances_antipodal_N12.npz, one instance per map), 64 candidates
# (Final Proj. 64 N, MMD-CBS 64 N samples and 60 s). Exceptions, as run: DiRecT on Cross-Room, Drop-Region and
# Corridor-Bay planned the three rings of data_prep/instances/rings/instances_ring_N12.npz together (instance 0 is the
# antipodal one); MMD-CBS ran before the isotropic-normalizer fix (RBT_ISO=0 here; see eval/mmd_cbs_rerun.md).
# Then: python hardware/extract_plans.py (picks each method's plan, compares with plans/).
# Output: $RB_RUNS/hardware/<map>_<method>.{json,log}, mmd_<map>/; the planners' outputs in the forks' results/.
. "$(dirname "$0")/../config/env.sh"
G=(${1:-0}); G=(${G[@]//,/ })
O=$RB_RUNS/hardware; mkdir -p $O
A=$RB_CODE/hardware/instances_antipodal_N12.npz; RING=$RB_CODE/data_prep/instances/rings/instances_ring_N12.npz
D=$MRMP_ROOT/DiRecT; P=$MRMP_ROOT/PCD/multi_robot; F=$RB_RUNS/models_flat
DP="$D:$D/deps/torch_robotics:$D/deps/motion_planning_baselines:$D/deps/experiment_launcher"
PP="$P:$P/deps/torch_robotics:$P/deps/motion_planning_baselines:$P/deps/experiment_launcher"
MP="$MRMP_ROOT/SMD/deps/motion_planning_baselines:$MRMP_ROOT/SMD/deps/experiment_launcher"
declare -A DENV=([empty]=empty [corridor]=conveyor [crossroom]=crossroom [shelf]=dropregion [corridorbay]=corridorsmd)
job(){   # job <gpu> <map> <method>
  g=$1; mp=$2; me=$3; c=$(rb_map $mp env_class); ob=$(rb_map $mp obs_scale); o=$O/${mp}_$me
  W=""; [ $mp = corridor ] && W="RBT_WIDE=1"
  NPZ=$A; K=1; EXP=antipodal_${mp}_N12
  [ $me = direct ] && case $mp in crossroom|shelf|corridorbay) NPZ=$RING; K=3; EXP=ring_${mp}_N12;; esac
  E="$W CUDA_VISIBLE_DEVICES=$g OBS_SCALE=$ob SHARED_INSTANCES_NPZ=$NPZ SHARED_INSTANCES_KEY=${mp}rbt/N12 MODELS_ROOT=$F"
  rb_log "hardware: $me $mp gpu $g"
  case $me in
    direct) cd $D && env $E RBT_PORT=1 PYTHONPATH=$DP "$PY_DIRECT" scripts/inference/run_direct_cell.py --env ${DENV[$mp]} \
              --env-class $c --model-id $c-RobotPlanarDisk --agents 12 --vmax 1.2 --n-inits $K --n-samples 64 --obs-scale $ob \
              --robot-r 0.13 --gpu-id 0 --timing-json $o.json --exp-name $EXP ;;
    cd) cd $D && env $E RBT_PORT=1 PYTHONPATH=$DP "$PY_DIRECT" scripts/inference/run_cd_pdm_cell.py --method cd --map $mp --agents 12 \
              --vmax 1.2 --n-inits 1 --n-samples 64 --obs-scale $ob --robot-r 0.13 --gpu-id 0 --timing-json $o.json ;;
    final_projection) cd $D && env $E RBT_PORT=1 PYTHONPATH=$DP "$PY_DIRECT" scripts/inference/run_finalproj_cell.py --map $mp \
              --agents 12 --vmax 1.2 --n-inits 1 --n-samples 768 --models-root $F --gpu-id 0 --obs-scale $ob --timing-json $o.json ;;
    pcd_shd|pcd_lb) cost=hinge_sqr_l2; [ $me = pcd_lb ] && cost=sum_log_l2
          cd $P && env $E PYTHONPATH=$PP "$PY_DIRECT" scripts/inference/run_pcd_cell.py --map $mp --n 12 --instances 1 --samples 64 \
              --cost $cost --vreg 0 --obs-scale $ob --robot-r 0.13 --out $o.json ;;
    mmd_cbs) cd $D && env $E RBT_ISO=0 OBSTACLE_MARGIN=0.12 INTER_AGENT_MARGIN=0.16 MMD_SAVE_TRAJ=$O/mmd_$mp PYTHONPATH=$MP \
              "$PY_SMD" -u scripts/inference/run_mmd_cbs_cell.py --map $mp --n-agents 12 --n-instances 1 --n-samples 768 \
              --runtime-limit 60 --models-root $F --out $o.json ;;
  esac > $o.log 2>&1
  rb_log "hardware: $me $mp done rc=$?"
}
# two queues (Final Proj. names its outputs after the map's tile, which Cross-Room and Drop-Region share: keep its cells
# in one queue); MMD-CBS last in its queue, alone on that GPU
MAPS="empty corridor crossroom shelf corridorbay"
( for me in direct cd final_projection; do for mp in $MAPS; do job ${G[0]} $mp $me; done; done ) &
( for me in pcd_shd pcd_lb mmd_cbs; do for mp in $MAPS; do job ${G[1]:-${G[0]}} $mp $me; done; done ) &
wait; rb_log "hardware runs done"
