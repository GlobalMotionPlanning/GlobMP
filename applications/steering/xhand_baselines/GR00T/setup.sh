#!/usr/bin/env bash
# One-time setup for the standalone GR00T pick baseline on a fresh server.
# Uses `uv` (fetches its own Python 3.10), creates a local .venv with pinned deps.
# Code-only bundle: the base model + Cosmos backbone + dataset are downloaded/linked
# separately (see scripts/download_base.sh and scripts/download_data.sh).
#
# Prereqs: `uv` (curl -LsSf https://astral.sh/uv/install.sh | sh),
# a CUDA 12.8-capable NVIDIA driver, and internet for the pip wheels.
#
#   bash setup.sh                       # .venv in this folder
#   VENV_DIR=/data2/groot_venv bash setup.sh   # put .venv elsewhere (e.g. if root disk is full)
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
VENV_DIR="${VENV_DIR:-$HERE/.venv}"

command -v uv >/dev/null 2>&1 || { echo "ERROR: install uv:  curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

echo "[groot setup] creating venv (Python 3.10 via uv) at $VENV_DIR ..."
uv venv --python 3.10 "$VENV_DIR"
[ "$VENV_DIR" = "$HERE/.venv" ] || ln -sfn "$VENV_DIR" "$HERE/.venv"

echo "[groot setup] installing torch 2.7.1 + torchvision 0.22.1 (CUDA 12.8) ..."
uv pip install --python "$VENV_DIR/bin/python" torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

echo "[groot setup] installing torchcodec 0.4.0 (video decode) ..."
uv pip install --python "$VENV_DIR/bin/python" torchcodec==0.4.0

echo "[groot setup] installing pinned deps ..."
uv pip install --python "$VENV_DIR/bin/python" -r requirements.txt

echo "[groot setup] import check ..."
PYTHONPATH="$HERE" GROOT_HF_LOCAL_FIRST=1 "$VENV_DIR/bin/python" -c "import gr00t; from gr00t.policy.gr00t_policy import Gr00tPolicy; print('gr00t import OK')"

cat <<EOF

[groot setup] DONE.
Next:
  1. bash scripts/download_base.sh     # GR00T-N1.7-3B base + Cosmos-Reason2-2B backbone
  2. bash scripts/download_data.sh     # DexMate Vega pick LeRobot dataset
  3. bash train/finetune_pick.sh       # train the baseline
EOF
