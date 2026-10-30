#!/usr/bin/env bash
# run.sh <broad|narrow> <seed> [gpu] [episodes]  : GlobMP in DiRecT's sampler, the paper's configuration (below).
# Output: $MB_RUNS/globmp/<setting>/seed<seed>/eval/{statistics.csv, summary_statistics.txt, <run>/trajectories.npz}
# and $MB_RUNS/globmp/<setting>/seed<seed>.plans.jsonl (the margin each plan holds).
# Needs the maze2d_baselines setup (../maze2d_baselines/README.md): config/paths.env, the dynamics model
# (data_prep/fit_dynamics.sh) and the checkpoint (checkpoints/last.ckpt).
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/../maze2d_baselines/config/env.sh"
S=$1; SEED_=${2:-0}; GPU=${3:-0}; R=${4:-10}
[ -f "$MB_RUNS/dynamics/linear_model.npz" ] || { echo "no dynamics model: run ../maze2d_baselines/data_prep/fit_dynamics.sh" >&2; exit 1; }
export GLOBMP_ROOT=$(cd "$HERE/../../.." && pwd)
# The configuration (see README):
#  cost: DiRecT's tracking and path terms + GlobMP's acceleration energy (weight 1e9)
#  every t: keep-outs (margin 0.39 + inflation 0.01, ladder 0.37/0.35/0.30/0.20/0.10), maze walls near the plan,
#           keep-outs on the executed path predicted by the fitted dynamics and the PD tracker (400 steps, margin 0.10),
#           |a| <= 1, |s| <= 1 (normalized)
#  solver: Barzilai-Borwein restoration, largest margin first, 1 s per warm solve, re-solve every 8 controlled steps
export GMP_CHARTSCALE=0.5 GMP_CSFLOOR=1e-3 GMP_LAMACC=1e9 GMP_MARGIN=0.39 GMP_MLADDER=0.37,0.35,0.30,0.20,0.10 \
  GMP_EXEC=400 GMP_EMARGIN=0.10 GMP_SBOUND=1 GMP_WALLS=1 GMP_WALLPAD=0.12 GMP_WALLOFF=0.2 GMP_KOPRE=1 GMP_KOSLACK=2.0 \
  GMP_BB=1 GMP_LADDER_UP=3 GMP_RUNG=1.5 GMP_COLD=8 GMP_ALG2COLD=1 GMP_DEADLINE=1 GMP_EVERY=8 GMP_BLOCKS=300 \
  GMP_BETABOUND=1 GMP_THREADS=${GMP_THREADS:-4}
mkdir -p "$MB_RUNS/globmp/$S"; export GMP_LOG="$MB_RUNS/globmp/$S/seed$SEED_.plans.jsonl"; rm -f "$GMP_LOG"
cd "$DIRECT_DIR" && mb_log "globmp $S seed $SEED_ gpu $GPU ($R episodes)" && \
  CUDA_VISIBLE_DEVICES=$GPU "$MB_PY" "$HERE/run_gmp.py" --config "$MB_CODE/config/config_traj.yaml" --policy direct \
    --obstacle-type "$S" --weights-path "$MB_WEIGHTS" --dynamics-path "$MB_RUNS/dynamics/linear_model.npz" \
    --exp-name "globmp/$S/seed$SEED_" --num-random-repeats "$R" --seed "$SEED_" --device cuda --log-folder "$MB_RUNS"
