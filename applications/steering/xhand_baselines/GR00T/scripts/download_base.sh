#!/usr/bin/env bash
# Download the stock GR00T-N1.7-3B base model + its Cosmos-Reason2-2B VLM backbone.
# Both are gated NVIDIA repos on the Hugging Face Hub — run `huggingface-cli login` first
# (or export HF_TOKEN). Nothing here is committed to git (see .gitignore).
#
#   bash scripts/download_base.sh
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_DIR="${BASE_DIR:-$HERE/base_model/GR00T-N1.7-3B}"
BACKBONE_DIR="${BACKBONE_DIR:-$HERE/backbone/cosmos-reason2-2b}"
BASE_REPO="${BASE_REPO:-nvidia/GR00T-N1.7-3B}"
BACKBONE_REPO="${BACKBONE_REPO:-nvidia/Cosmos-Reason2-2B}"

command -v huggingface-cli >/dev/null 2>&1 || { echo "ERROR: pip install huggingface-hub, then huggingface-cli login"; exit 1; }
mkdir -p "$BASE_DIR" "$BACKBONE_DIR"

echo "[download] base model $BASE_REPO -> $BASE_DIR"
huggingface-cli download "$BASE_REPO" --local-dir "$BASE_DIR"

echo "[download] VLM backbone $BACKBONE_REPO -> $BACKBONE_DIR"
huggingface-cli download "$BACKBONE_REPO" --local-dir "$BACKBONE_DIR"

echo "[download] DONE."
echo "  export GROOT_BACKBONE_PATH=$BACKBONE_DIR   # so training/inference run offline"
echo "  BASE_MODEL=$BASE_DIR bash train/finetune_pick.sh"
