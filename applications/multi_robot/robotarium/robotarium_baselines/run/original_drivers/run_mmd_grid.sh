#!/bin/bash
L=<path>
DM=$L/repro/rbtB/DiRecT-main/multi_robot
SP=<tmp>/f915b73d-2815-4d04-99b9-6801f3fcf24f/scratchpad
NPZ=<path>/baselines/instances_certified_B_eval10.npz
OUT=$SP/mmd_grid_out; mkdir -p $OUT
declare -A OBS=( [empty]=1.0 [corridor]=1.10 [crossroom]=0.85 [shelf]=0.40 [corridorbay]=0.20 )
cd $DM || exit 1
i=0
# easy maps first (fast), hard maps last
for MP in empty shelf corridorbay corridor crossroom; do
  for N in 3 6 9 12; do
    g=$((i % 8)); i=$((i+1))
    (
      env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$g RBT_ARENA=1 RBT_ISO=1 ROBOT_R=0.13 \
          OBS_SCALE=${OBS[$MP]} OBSTACLE_MARGIN=0.12 INTER_AGENT_MARGIN=0.16 SHARED_INSTANCES_NPZ=$NPZ \
        $L/miniconda3/envs/smd/bin/python -u scripts/inference/run_mmd_cbs_cell.py \
          --map $MP --n-agents $N --n-instances 10 --n-samples $((64*N)) --runtime-limit 60 \
          --models-root $L/repro/baselines/models_flat_rbt \
          --out $OUT/mmd_${MP}_N${N}.json > $OUT/mmd_${MP}_N${N}.log 2>&1
      ec=$?
      if [ -f $OUT/mmd_${MP}_N${N}.json ]; then
        sr=$($L/miniconda3/envs/smd/bin/python -c "import json;d=json.load(open('$OUT/mmd_${MP}_N${N}.json'));p=d['per_n']['$N'];print('SR=%.2f st=%s'%(p['SR'],p['statuses']))" 2>&1)
        echo "[$(date +%H:%M:%S)] DONE $MP N$N ec=$ec $sr"
      else echo "[$(date +%H:%M:%S)] NOJSON $MP N$N ec=$ec"; fi
    ) &
    [ $((i % 8)) -eq 0 ] && wait
  done
done
wait
echo "MMD GRID COMPLETE $(date +%H:%M:%S)"
