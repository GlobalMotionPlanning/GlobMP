#!/usr/bin/env bash
# Start the policy server (runs in the MODEL's venv, holds the loaded checkpoint, answers
# action queries over a Unix socket). Engine-selectable: crom | gr00t.
#
#   ENGINE=crom CKPT=<dir> MODEL_VENV=<crom-3b/.venv> MODALITY=<vega_pick_modality.py> \
#   BACKBONE=<cosmos-reason2-2b> bash drivers/run_server.sh
set -u
SB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="${ENGINE:-crom}"
CKPT="${CKPT:?set CKPT to a checkpoint dir}"
MODEL_VENV="${MODEL_VENV:?set MODEL_VENV to the engine's python venv}"
MODALITY="${MODALITY:-$SB/configs/vega_pick_modality.py}"
BACKBONE="${BACKBONE:?set BACKBONE to the cosmos-reason2-2b dir}"
SOCK="${SOCK:-/tmp/simbench_policy.sock}"
DEVICE="${DEVICE:-cuda:0}"
pgrep -f policy_server_eval | xargs -r kill -9 2>/dev/null
sleep 1; rm -f "$SOCK" "$SOCK.ready"
# both engines look for their backbone under the same env name the repos already use
export CROM3B_BACKBONE_PATH="$BACKBONE" GROOT_BACKBONE_PATH="$BACKBONE"
export GROOT_HF_LOCAL_FIRST=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec "$MODEL_VENV/bin/python" "$SB/server/policy_server_eval.py" \
  --ckpt "$CKPT" --modality-config "$MODALITY" --device "$DEVICE" \
  --sock "$SOCK" --engine "$ENGINE"
