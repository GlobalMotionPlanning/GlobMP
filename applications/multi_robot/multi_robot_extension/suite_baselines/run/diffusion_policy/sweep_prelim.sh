#!/bin/bash
# diffusion_policy PRELIMINARY sweep: 6 runnable maps x 7 team sizes = 42 cells.
# The 3 velocity regimes are FREE: the method is unguided + unprojected, so v_max never
# enters the computation; score.py evaluates all three post-hoc as VelOK_{lo,mid,hi}
# from the SAME trajectories. Hence 42 runs, not 126.
#
# usage: sweep_prelim.sh <gpu> <cpus> <n_inits> <n_samples> [tag]
set -u
GPU=${1:-5}; CPUS=${2:-100-105}; NI=${3:-200}; NS=${4:-16}; TAG=${5:-prelim}
W="$MRMP_ROOT/run/diffusion_policy"   # the runners; outputs go to $DP_WORK
OUT=$W/${TAG}_results.csv; : > $W/${TAG}_index.txt
for MAP in empty corridor room shelf corridorbay crossroom; do
  for NA in 3 6 9 12 15 18 21; do
    L=$W/logs_$TAG/${MAP}_a${NA}.log; mkdir -p $(dirname $L)
    S=$(date +%s)
    timeout 1800 $W/run_cell.sh $MAP $NA $NI $NS $GPU $CPUS ${TAG} shared_trained > $L 2>&1
    RC=$?; E=$(( $(date +%s) - S ))
    D=$(grep -oP 'Results will be saved to: \K.*' $L | tail -1)
    echo "$MAP $NA rc=$RC wall=${E}s $D" | tee -a $W/${TAG}_index.txt
    [ -n "$D" ] && echo "$D|$MAP|$L" >> $W/${TAG}_dirs.txt
  done
done
echo "scoring -> $OUT"
$W/collect_cells.py $W/${TAG}_dirs.txt $OUT
