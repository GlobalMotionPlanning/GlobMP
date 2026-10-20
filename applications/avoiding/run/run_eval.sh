#!/bin/bash
# run_eval.sh [seed ...] : GlobMP as the projector of DiRecT's sampler on D3IL avoiding-v0, hard constraint set (10 discs +
# 2 half-planes, D3IL_HARD=1), the shared checkpoint model_ema_10, 32 sampling steps, no early stop, 100 episodes per seed.
# GlobMP setting = the best of the sweep (README): replan every 4 steps, margin 0.02 m, deadline 3 s
# per projection (12 s for a plan's first, cold, projection), guidance in the last 0 steps (0 = all).
# Needs: the direct-d3il env (avoiding_baselines/env, README), PY = its python, a GPU (DEVICE), NWORK CPU cores.
set -e
HERE=$(cd "$(dirname "$0")" && pwd); AV=$(dirname "$HERE"); D=$AV/avoiding_baselines/DiRecT
PY=${PY:-python}; DEVICE=${DEVICE:-cuda:0}; NWORK=${NWORK:-28}
[ -d "$D/logs/avoiding-v0" ] || { mkdir -p "$D/logs"; cp -r "$AV/avoiding_baselines/checkpoints/avoiding-v0" "$D/logs/"; }
grep -q "DIRECT_PROJECTOR" "$D/policy/direct.py" || (cd "$D" && patch -p1 < "$HERE/direct_hook.patch")
export PYTHONPATH=$AV:$D${PYTHONPATH:+:$PYTHONPATH} PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=${MUJOCO_GL:-egl} D3IL_HARD=1 \
  DIRECT_PROJECTOR=globmp GPA_AV_WORKERS=$NWORK OMP_NUM_THREADS=1 GPA_AV_RATE=1 GPA_AV_GROW=0 GPA_AV_WARM=1 GPA_AV_RECOVER=0.01 \
  GPA_AV_REPLAN=4 GPA_AV_DEADLINE=3 GPA_AV_COLD_DEADLINE=12
cd "$D"
for s in ${*:-0}; do
  "$PY" eval.py --diffusion-exp-name x0pred --diffusion-cp 10 --guidance-method direct --constraints --no-early-stop \
    --n-sampling-steps 32 --guidance-last-steps 0 --random-repeat 100 --batch-size 50 --obstacle-margin 0.02 \
    --replan-steps 4 --exp-name globmp_seed$s --seed $s --device $DEVICE
done
