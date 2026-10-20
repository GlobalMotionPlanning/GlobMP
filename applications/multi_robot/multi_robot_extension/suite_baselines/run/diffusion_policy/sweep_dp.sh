#!/bin/bash
# PRELIMINARY diffusion_policy sweep: 6 usable maps x 7 team sizes, one GPU.
# diffusion_policy is unguided AND unprojected -> vel_max never enters the sampler, so ONE
# run per (map, team) supplies all three velocity regimes; the regime split is scored
# post-hoc by score.py from the saved trajectories.
#
# usage: sweep_dp.sh <gpu> <cpus> <n_inits> <n_samples_per_agent> [tag]
set -u
GPU=${1:-5}; CPUS=${2:-70-75}; NI=${3:-20}; NS=${4:-8}; TAG=${5:-prelim}
W="$MRMP_ROOT/run/diffusion_policy"   # the runners; outputs go to $DP_WORK
MAPS="empty corridor room shelf corridorbay crossroom"    # basic/dense have no baseline dataset
TEAMS="3 6 9 12 15 18 21"
mkdir -p $W/sweep_$TAG
for m in $MAPS; do
  for n in $TEAMS; do
    L=$W/sweep_$TAG/${m}_a${n}.log
    echo "[$(date +%T)] $m a=$n -> $L"
    $W/run_dp.sh $m $n $NI $NS $GPU $CPUS $TAG > $L 2>&1
    echo "[$(date +%T)]   exit=$? wall=$(grep -o 'Elapsed.*' $L | tail -1)"
  done
done
echo SWEEP_DONE
