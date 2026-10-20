#!/usr/bin/env bash
# Serve a trained GR00T baseline checkpoint over ZMQ (stock gr00t inference service).
# The robot/eval client connects with gr00t.policy.server_client.PolicyClient.
#
#   MODEL=<data>/checkpoints/groot_pick_baseline/checkpoint-5000 PORT=5555 bash inference/run_policy_server.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$HERE"
VENV="${VENV:-$HERE/.venv}"
MODEL="${MODEL:-<data>/checkpoints/groot_pick_baseline/checkpoint-5000}"
MODALITY="$HERE/configs/vega_pick_modality.py"
PORT="${PORT:-5555}"
DEVICE="${DEVICE:-cuda:0}"
BACKBONE="${GROOT_BACKBONE_PATH:-$HERE/backbone/cosmos-reason2-2b}"
FFMPEG_LIBS="${FFMPEG_LIBS:-$HERE/ffmpeg_libs}"

export PYTHONPATH="$HERE"
export GROOT_BACKBONE_PATH="$BACKBONE" GROOT_HF_LOCAL_FIRST=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH="$FFMPEG_LIBS:${LD_LIBRARY_PATH:-}"

echo "[groot serve] model=$MODEL port=$PORT device=$DEVICE"
exec "$VENV/bin/python" gr00t/eval/run_gr00t_server.py \
  --model-path "$MODEL" \
  --embodiment-tag new_embodiment \
  --modality-config-path "$MODALITY" \
  --device "$DEVICE" \
  --port "$PORT"
