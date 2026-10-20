#!/usr/bin/env bash
# OPEN-LOOP 3-column video: GT | model @ real frames | model @ Genesis frames.
# Isolates the sim-vs-real appearance gap (state is teacher-forced GT in all panels).
#
# Reuses the model repo's rollout_to_npz.py (stages A & C) — that script needs the model
# engine + venv, so point GROOT_REPO/MODEL_VENV at the checkpoint's repo.
#
#   MODEL=<ckpt> DATASET=<lerobot_pick> EPISODE=459 GROOT_REPO=<gr00t> \
#   MODEL_VENV=<venv> GENESIS_VENV=<.venv-genesis> BACKBONE=<cosmos> \
#   VEGA_URDF=<urdf> bash drivers/render_3col.sh
set -eu
SB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GROOT_REPO="${GROOT_REPO:?set GROOT_REPO to the repo holding inference/rollout_to_npz.py}"
MODEL_VENV="${MODEL_VENV:?set MODEL_VENV}"
GENESIS_VENV="${GENESIS_VENV:?set GENESIS_VENV}"
MODEL="${MODEL:?set MODEL to a checkpoint dir}"
DATASET="${DATASET:?set DATASET to a LeRobot pick dataset}"
EPISODE="${EPISODE:-459}"
DEVICE="${DEVICE:-cuda:0}"
BACKBONE="${BACKBONE:?set BACKBONE to cosmos-reason2-2b}"
MODCFG="${MODCFG:-$SB/configs/vega_pick_modality.py}"
WORK="${WORK:-$SB/out/3col_ep${EPISODE}}"
GENDS="$WORK/genesis_ds"
export VEGA_URDF="${VEGA_URDF:?set VEGA_URDF}"
mkdir -p "$WORK"
COMMON=(env PYTHONPATH="$GROOT_REPO" GROOT_BACKBONE_PATH="$BACKBONE" CROM3B_BACKBONE_PATH="$BACKBONE" \
        GROOT_HF_LOCAL_FIRST=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1)

TASK=$(python3 -c "import json
for l in open('$DATASET/meta/episodes.jsonl'):
    r=json.loads(l)
    if r['episode_index']==$EPISODE: print(r['tasks'][0]); break")
echo "[3col] ep$EPISODE task: $TASK"

echo "[3col] stage A: rollout on REAL frames -> col2"
"${COMMON[@]}" "$MODEL_VENV/bin/python" "$GROOT_REPO/inference/rollout_to_npz.py" \
  --ckpt "$MODEL" --dataset "$DATASET" --episode "$EPISODE" \
  --modality-config "$MODCFG" --device "$DEVICE" --out "$WORK/dataset.npz"

echo "[3col] stage B: Genesis renders the 4 policy cameras for the GT traj"
PYOPENGL_PLATFORM=egl "$GENESIS_VENV/bin/python" "$SB/sim/genesis_render_policy_cams.py" \
  --npz "$WORK/dataset.npz" --out-dataset "$GENDS" --task "$TASK" --episode "$EPISODE"

echo "[3col] stage C: rollout on GENESIS frames -> col3"
"${COMMON[@]}" "$MODEL_VENV/bin/python" "$GROOT_REPO/inference/rollout_to_npz.py" \
  --ckpt "$MODEL" --dataset "$GENDS" --episode "$EPISODE" \
  --modality-config "$MODCFG" --device "$DEVICE" --out "$WORK/genesis.npz"

echo "[3col] stage D: merge (align by kept frames) + 3-panel render"
python3 -c "
import numpy as np
a=np.load('$WORK/dataset.npz'); b=np.load('$WORK/genesis.npz')
idx=np.load('$GENDS/meta/keep_idx.npy'); idx=idx[idx<len(a['gt'])]
g=b['pred']; n=min(len(idx),len(g))
np.savez('$WORK/combined.npz', gt=a['gt'][idx[:n]], pred=a['pred'][idx[:n]], pred2=g[:n])
print('combined aligned',n,'frames')"
PYOPENGL_PLATFORM=egl "$GENESIS_VENV/bin/python" "$SB/sim/genesis_replay_nway.py" \
  --npz "$WORK/combined.npz" --keys pred pred2 \
  --labels "model @ real frames" "model @ Genesis frames" \
  --colors "255,150,60" "60,170,255" --out "$WORK/gt_vs_real_vs_genesis"
echo "[3col] DONE -> $WORK/gt_vs_real_vs_genesis.mp4"
