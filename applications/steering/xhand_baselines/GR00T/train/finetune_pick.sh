#!/usr/bin/env bash
# Task wrapper: PICK. Defaults for the bimanual pick dataset; the real trainer is
# train/finetune.sh, which is task-agnostic.
#
#   bash train/finetune_pick.sh                 # full baseline
#   MAX_STEPS=3 bash train/finetune_pick.sh     # smoke
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TASK="${TASK:-pick}" SOURCE="${SOURCE:-teleop}"
export MODALITY="${MODALITY:-$HERE/configs/vega_pick_modality.py}"
exec bash "$HERE/train/finetune.sh" "$@"
