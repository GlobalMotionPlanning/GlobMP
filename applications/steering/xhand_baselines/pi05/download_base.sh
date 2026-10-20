#!/usr/bin/env bash
# pi0.5 base model (LeRobot format) for finetune_pi05.sh and finetune_tvla.sh -> pi05/base_model (XB_PI05_BASE).
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/base_model"
DL="huggingface-cli download"; command -v hf >/dev/null 2>&1 && DL="hf download"
$DL lerobot/pi05_base --local-dir "$DIR"
