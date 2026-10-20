#!/usr/bin/env bash
# train_diffuser.sh [gpu] : the diffusion planner every baseline samples from (about 3.4 h on one GPU, ~9 GB).
# Skip it to use the shipped checkpoints/last.ckpt (MB_WEIGHTS). Output: runs/maze2d-diffuser/train/checkpoints/last.ckpt
. "$(dirname "$0")/../config/env.sh"
GPU=${1:-0}
cd "$DIRECT_DIR" && mb_log "train diffuser on gpu $GPU ($TRAIN_STEPS steps, batch $TRAIN_BATCH)" && \
  CUDA_VISIBLE_DEVICES=$GPU "$MB_PY" train.py --config configs/config.yaml --exp-name maze2d-diffuser \
    --seed "$SEED" --batch-size "$TRAIN_BATCH" --n-train-steps "$TRAIN_STEPS" --device cuda --log-folder "$MB_RUNS"
