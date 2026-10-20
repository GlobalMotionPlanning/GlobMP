#!/bin/bash
SP=<tmp>/f915b73d-2815-4d04-99b9-6801f3fcf24f/scratchpad
L=<path>
MMDT=$L/repro/rbtB/DiRecT-main/multi_robot
PYS=$L/miniconda3/envs/smd/bin/python
NPZ=<path>/rbt_sim/ring_only_bench_N12.npz
OUT=$SP/mmd_ring_out; mkdir -p $OUT
declare -A OBS=( [empty]=1.0 [corridor]=1.10 [crossroom]=0.85 [shelf]=0.4 [corridorbay]=0.2 )
cd $MMDT || exit 1
i=0
for mp in empty crossroom shelf corridorbay corridor; do
  g=$i; i=$((i+1))
  WIDE=""; MROOT=$L/repro/baselines/models_flat_rbt
  if [ "$mp" = corridor ]; then WIDE="RBT_WIDE=1"; MROOT=$L/repro/baselines/models_flat_rbtwide; fi
  ( env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$g RBT_ARENA=1 RBT_ISO=1 ROBOT_R=0.13 OBS_SCALE=${OBS[$mp]} $WIDE       OBSTACLE_MARGIN=0.12 INTER_AGENT_MARGIN=0.16 SHARED_INSTANCES_NPZ=$NPZ SHARED_INSTANCES_KEY=${mp}rbt/N12       $PYS scripts/inference/run_mmd_cbs_cell.py --map $mp --n-agents 12 --n-instances 1 --n-samples 768       --runtime-limit 60 --models-root $MROOT --out $OUT/mmd_${mp}.json > $OUT/mmd_${mp}.log 2>&1
    ec=$?
    if [ -f $OUT/mmd_${mp}.json ]; then
      r=$($PYS -c "import json;d=json.load(open('$OUT/mmd_${mp}.json'));p=d['per_n']['12'];print('SR=%.2f statuses=%s'%(p['SR'],p['statuses']))" 2>&1)
      echo "[$(date +%H:%M:%S)] $mp ec=$ec $r"
    else echo "[$(date +%H:%M:%S)] $mp ec=$ec NOJSON"; tail -3 $OUT/mmd_${mp}.log | tr '\n' ' '; echo; fi
  ) &
done
wait
echo "MMD RING COMPLETE $(date +%H:%M:%S)"
