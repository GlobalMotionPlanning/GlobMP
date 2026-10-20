#!/bin/bash
# run_cell.sh <method> <map> <N> [gpu] : one baseline on one Robotarium cell (10 instances of $RB_INSTANCES), with the
# settings of the paper's runs (run/original_drivers/). methods: direct cd final_projection pcd_shd pcd_lb mmd_cbs
# maps: empty crossroom shelf corridorbay corridor; N: 3 6 9 12. Output: $RB_CELLS/<method>__<map>__N<N>.{json,log};
# the planners' raw outputs stay in the forks' results/ folders, where eval/score_cells.py reads them.
. "$(dirname "$0")/../config/env.sh"
M=$1; mp=$2; N=$3; GPU=${4:-0}
c=$(rb_map $mp env_class); ob=$(rb_map $mp obs_scale)
declare -A DENV=([empty]=empty [corridor]=conveyor [crossroom]=crossroom [shelf]=dropregion [corridorbay]=corridorsmd)
D=$MRMP_ROOT/DiRecT; P=$MRMP_ROOT/PCD/multi_robot
F=$RB_RUNS/models_flat; O=$RB_CELLS/${M}__${mp}__N$N
# CD and Final Proj. read their per-map table under RBT_PORT=1 (MMD-CBS runs without it); Conveyor adds RBT_WIDE=1,
# the widened channels (run/original_drivers/wide_allN.py)
PORT="RBT_PORT=1"; W=""; [ $mp = corridor ] && W="RBT_WIDE=1"
export CUDA_VISIBLE_DEVICES=$GPU OBS_SCALE=$ob SHARED_INSTANCES_NPZ=$RB_INSTANCES SHARED_INSTANCES_KEY=${mp}rbt/N$N MODELS_ROOT=$F
[ -n "$W" ] && export RBT_WIDE=1
DP="$D:$D/deps/torch_robotics:$D/deps/motion_planning_baselines:$D/deps/experiment_launcher"
PP="$P:$P/deps/torch_robotics:$P/deps/motion_planning_baselines:$P/deps/experiment_launcher"
# MMD-CBS (smd env) imports this fork's mmd and torch_robotics by itself; experiment_launcher and mp_baselines came
# from the SMD fork in the paper's smd env
MP="$MRMP_ROOT/SMD/deps/motion_planning_baselines:$MRMP_ROOT/SMD/deps/experiment_launcher"
rb_log "$M $mp N=$N gpu $GPU"
case $M in
  direct) cd $D && env $PORT PYTHONPATH=$DP "$PY_DIRECT" scripts/inference/run_direct_cell.py --env ${DENV[$mp]} --env-class $c \
            --model-id $c-RobotPlanarDisk --agents $N --vmax 1.2 --n-inits 10 --n-samples 64 --obs-scale $ob --robot-r 0.13 \
            --gpu-id 0 --timing-json $O.json --exp-name v2_direct_${mp}_N$N > $O.log 2>&1 ;;
  cd) cd $D && env $PORT PYTHONPATH=$DP "$PY_DIRECT" scripts/inference/run_cd_pdm_cell.py --method cd --map $mp --agents $N --vmax 1.2 \
            --n-inits 10 --n-samples 64 --obs-scale $ob --robot-r 0.13 --gpu-id 0 --timing-json $O.json > $O.log 2>&1 ;;
  final_projection) cd $D && env $PORT PYTHONPATH=$DP "$PY_DIRECT" scripts/inference/run_finalproj_cell.py --map $mp --agents $N \
            --vmax 1.2 --n-inits 10 --n-samples $((64*N)) --models-root $F --gpu-id 0 --obs-scale $ob --timing-json $O.json > $O.log 2>&1 ;;
  pcd_shd|pcd_lb) cost=hinge_sqr_l2; [ $M = pcd_lb ] && cost=sum_log_l2
          cd $P && env $PORT PYTHONPATH=$PP "$PY_DIRECT" scripts/inference/run_pcd_cell.py --map $mp --n $N --instances 10 --samples 64 \
            --cost $cost --vreg 0 --obs-scale $ob --robot-r 0.13 --out $O.json > $O.log 2>&1 ;;
  mmd_cbs) cd $D && env OBSTACLE_MARGIN=0.12 INTER_AGENT_MARGIN=0.16 PYTHONPATH=$MP "$PY_SMD" -u scripts/inference/run_mmd_cbs_cell.py --map $mp \
            --n-agents $N --n-instances 10 --n-samples $((64*N)) --runtime-limit 60 --models-root $F --out $O.json > $O.log 2>&1 ;;
  *) echo "method: direct cd final_projection pcd_shd pcd_lb mmd_cbs" >&2; exit 1 ;;
esac
rc=$?; rb_log "$M $mp N=$N done rc=$rc"; exit $rc
