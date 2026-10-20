#!/bin/bash
L=<path>; R=$L/repro
DM=$R/rbt/DiRecT-main/multi_robot; PCD=$R/rbt/pcd/multi_robot; MMDT=$R/rbtB/DiRecT-main/multi_robot
PYD=$L/miniconda3/envs/direct/bin/python; PYS=$L/miniconda3/envs/smd/bin/python
NPZ=$R/baselines/instances_corridorwide_bench_N12.npz
E="PYTHONNOUSERSITE=1 RBT_WIDE=1 RBT_ARENA=1 RBT_ISO=1 ROBOT_R=0.13 OBS_SCALE=1.1 SHARED_INSTANCES_NPZ=$NPZ SHARED_INSTANCES_KEY=corridorrbt/N12"
O=$R/rbt_sim; cd $DM
( env $E CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$DM/deps/torch_robotics $PYD scripts/inference/run_direct_cell.py --env conveyor --env-class EnvConveyor2D --model-id EnvConveyor2D-RobotPlanarDisk --agents 12 --vmax 1.2 --n-inits 10 --n-samples 64 --obs-scale 1.1 --robot-r 0.13 --gpu-id 0 --timing-json $O/wide_direct_N12.json --exp-name wide_direct_N12 > $O/wide_direct_N12.log 2>&1; echo "direct rc=$?"
  cd $PCD && env $E CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PCD/deps/torch_robotics $PYD scripts/inference/run_pcd_cell.py --map corridor --n 12 --instances 10 --samples 64 --cost hinge_sqr_l2 --vreg 0 --obs-scale 1.1 --robot-r 0.13 --out $O/wide_pcd_shd_N12.json > $O/wide_pcd_shd_N12.log 2>&1; echo "pcd_shd rc=$?" ) &
( env $E CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$DM/deps/torch_robotics $PYD scripts/inference/run_cd_pdm_cell.py --method cd --map corridor --agents 12 --vmax 1.2 --n-inits 10 --n-samples 64 --obs-scale 1.1 --robot-r 0.13 --gpu-id 0 --timing-json $O/wide_cd_N12.json > $O/wide_cd_N12.log 2>&1; echo "cd rc=$?"
  cd $PCD && env $E CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PCD/deps/torch_robotics $PYD scripts/inference/run_pcd_cell.py --map corridor --n 12 --instances 10 --samples 64 --cost sum_log_l2 --vreg 0 --obs-scale 1.1 --robot-r 0.13 --out $O/wide_pcd_lb_N12.json > $O/wide_pcd_lb_N12.log 2>&1; echo "pcd_lb rc=$?" ) &
( env $E CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$DM/deps/torch_robotics $PYD scripts/inference/run_finalproj_cell.py --map corridor --agents 12 --vmax 1.2 --n-inits 10 --n-samples 768 --models-root $R/baselines/models_flat_rbtwide --gpu-id 0 --obs-scale 1.1 --timing-json $O/wide_fp_N12.json > $O/wide_fp_N12.log 2>&1; echo "final_projection rc=$?"
  cd $MMDT && env $E CUDA_VISIBLE_DEVICES=2 OBSTACLE_MARGIN=0.12 INTER_AGENT_MARGIN=0.16 $PYS -u scripts/inference/run_mmd_cbs_cell.py --map corridor --n-agents 12 --n-instances 10 --n-samples 768 --runtime-limit 60 --models-root $R/baselines/models_flat_rbtwide --out $O/wide_mmd_N12.json > $O/wide_mmd_N12.log 2>&1; echo "mmd_cbs rc=$?" ) &
wait; echo WIDE-METHODS-COMPLETE
