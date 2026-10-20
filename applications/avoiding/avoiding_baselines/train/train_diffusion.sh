#!/usr/bin/env bash
# train_diffusion.sh [gpu] : the planner every baseline samples from.
# 500k steps at batch 32 -- what the paper reports; DiRecT/train.py defaults to 1,000,001.
# Skip this to use checkpoints/model_ema_10.pth, which is the one behind Table 3.
. "$(dirname "$0")/../config/env.sh"
GPU=${1:-0}
cd "$DIRECT_DIR" && ab_log "train $TRAIN_STEPS steps" && \
  CUDA_VISIBLE_DEVICES=$GPU "$AB_PY" train.py --exp-name x0pred --seed "$SEED" \
    --batch-size "$TRAIN_BATCH" --n-train-steps "$TRAIN_STEPS" --log-folder "$AB_RUNS"
