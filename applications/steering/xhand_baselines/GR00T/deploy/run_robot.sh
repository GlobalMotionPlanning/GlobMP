#!/usr/bin/env bash
# Deploy GR00T on the Dexmate Vega: ramp to the dataset episode's initial pose, then run LIVE
# closed-loop -- the checkpoint generates actions from the REAL cameras/joints. Run from the
# repo root on the robot laptop (gr00t env + GROOT_BACKBONE_PATH set). Keep the e-stop in hand.
#
#   CKPT=<data>/checkpoints/ckpt_pickteleop260624-260702_gr00t3b260707 \
#   TASK="pick: grasp the white box and lift it clear of the cart." \
#   DATASET=/path/lerobot_pick bash deploy/run_robot.sh
#
# TASK and CKPT are deliberately REQUIRED (no defaults): a fallback prompt is how a
# checkpoint silently gets evaluated with the wrong instruction. TASK must match a
# training instruction verbatim (multi-task checkpoints, e.g. point: "point to the
# red block" / "point to the green block" / "point to the red marker").
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO:$REPO/deploy:$PYTHONPATH"                       # gr00t package + dex_deploy harness
export GROOT_BACKBONE_PATH="${GROOT_BACKBONE_PATH:-$REPO/backbone/cosmos-reason2-2b}"
CKPT="${CKPT:?set CKPT to the checkpoint dir (has experiment_cfg/)}"
TASK="${TASK:?set TASK to the instruction, verbatim from training}"
DATASET="${DATASET:-$REPO/data/pick}"                                    # LeRobot dataset -> ramp target (frame-0 pose)

python -m dex_deploy.deploy \
  --pretrained-path "$CKPT" \
  --task "$TASK" \
  --action-chunk-threshold 1.0 \
  --device cuda:0 \
  --publish-actions \
  --dataset-path "$DATASET" \
  --episode-index "${EPISODE:-0}" \
  --ramp-steps 40 \
  --max-steps 1000
#
# Flow: e-stop -> home pose -> Enter (test hands) -> Enter -> ramp (40 steps) to the dataset
# frame-0 pose -> LIVE rollout. The dataset is used ONLY for the ramp target. Keep the e-stop in hand.
