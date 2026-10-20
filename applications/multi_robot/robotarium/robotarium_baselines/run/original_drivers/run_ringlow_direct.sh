#!/bin/bash
L=<path>
DM=$L/repro/rbt/DiRecT-main/multi_robot
SP=<tmp>/f915b73d-2815-4d04-99b9-6801f3fcf24f/scratchpad; NPZ=<path>/rbt_sim/ring_variants_lowN.npz
OUT=$SP/ringlow_out; mkdir -p $OUT
PY=$L/miniconda3/envs/direct/bin/python
declare -A DENV=( [corridor]=conveyor [crossroom]=crossroom [shelf]=dropregion [corridorbay]=corridorsmd )
declare -A ECLS=( [corridor]=EnvConveyor2D [crossroom]=EnvCrossRoom2D [shelf]=EnvDropRegion2D [corridorbay]=EnvCorridorSMD2D )
declare -A OBS=( [corridor]=1.1 [crossroom]=0.85 [shelf]=0.4 [corridorbay]=0.2 )
declare -A VC=( [3]=1 [6]=3 [9]=4 )   # variant count per N
cd $DM || exit 1
i=0
for mp in shelf corridorbay corridor crossroom; do
  for N in 3 6 9; do
    g=$((i % 8)); i=$((i+1))
    (
      env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$g RBT_ARENA=1 RBT_ISO=1 ROBOT_R=0.13 OBS_SCALE=${OBS[$mp]} \
          MODELS_ROOT=$L/repro/baselines/models_flat_rbt \
          SHARED_INSTANCES_NPZ=$NPZ SHARED_INSTANCES_KEY=${mp}rbt/N${N} PYTHONPATH=$DM/deps/torch_robotics \
        $PY scripts/inference/run_direct_cell.py --env ${DENV[$mp]} --env-class ${ECLS[$mp]} \
          --model-id ${ECLS[$mp]}-RobotPlanarDisk --agents $N --vmax 1.2 --n-inits ${VC[$N]} --n-samples 128 \
          --obs-scale ${OBS[$mp]} --robot-r 0.13 --gpu-id 0 \
          --timing-json $OUT/ringlow_${mp}_N${N}.json --exp-name ringlow_${mp}_N${N} \
          > $OUT/ringlow_${mp}_N${N}.log 2>&1
      echo "[$(date +%H:%M:%S)] done $mp N$N rc=$?"
    ) &
    [ $((i % 8)) -eq 0 ] && wait
  done
done
wait
echo "RINGLOW DIRECT COMPLETE $(date +%H:%M:%S)"
