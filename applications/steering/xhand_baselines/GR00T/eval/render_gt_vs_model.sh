#!/usr/bin/env bash
# Render a GT-vs-model Genesis video for a trained GR00T baseline checkpoint on one episode.
#   stage 1: open-loop rollout over a LeRobot episode -> npz {gt[L,22], pred[L,22]}
#   stage 2: Genesis 2-panel replay (GT | model) -> mp4 + gif
#
# Genesis has its own env (genesis-world, mujoco, pyopengl) that is kept separate from the
# training venv — see eval/requirements-genesis.txt. The Vega URDF is fetched separately
# (scripts/download_urdf.sh); point VEGA_URDF at vega_1p_gripper.urdf.
#
#   EPISODE=0 bash eval/render_gt_vs_model.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$HERE"
VENV="${VENV:-$HERE/.venv}"                             # gr00t venv (rollout)
GENESIS_VENV="${GENESIS_VENV:-$HERE/.venv-genesis}"     # genesis venv (render)
MODEL="${MODEL:?set MODEL to a checkpoint dir}"
DATASET="${DATASET:-<data>/data/pick}"
EPISODE="${EPISODE:-0}"
TASK="${TASK:-}"   # empty -> rollout_to_npz reads the episode's own task from the dataset
LABEL="${LABEL:-GR00T baseline pick}"
COLOR="${COLOR:-255 140 60}"
OUT="${OUT:-$MODEL/eval/gt_vs_model_ep${EPISODE}}"
BACKBONE="${GROOT_BACKBONE_PATH:-$HERE/backbone/cosmos-reason2-2b}"
FFMPEG_LIBS="${FFMPEG_LIBS:-$HERE/ffmpeg_libs}"
export VEGA_URDF="${VEGA_URDF:-$HERE/assets/dexmate-urdf/robots/humanoid/vega_1p/vega_1p_gripper.urdf}"
mkdir -p "$(dirname "$OUT")"

echo "[render] stage 1/2: rollout ep$EPISODE -> $OUT.npz"
PYTHONPATH="$HERE" GROOT_BACKBONE_PATH="$BACKBONE" GROOT_HF_LOCAL_FIRST=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LD_LIBRARY_PATH="$FFMPEG_LIBS:${LD_LIBRARY_PATH:-}" \
  "$VENV/bin/python" inference/rollout_to_npz.py \
    --ckpt "$MODEL" --dataset "$DATASET" --episode "$EPISODE" \
    ${TASK:+--task "$TASK"} --device "${DEVICE:-cuda:0}" --out "$OUT.npz"

echo "[render] stage 2/2: Genesis replay -> $OUT.mp4"
[ -x "$GENESIS_VENV/bin/python" ] || { echo "genesis venv missing at $GENESIS_VENV (see eval/requirements-genesis.txt)"; exit 1; }
PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" "$GENESIS_VENV/bin/python" eval/genesis_replay_2way.py \
  --npz "$OUT.npz" --label "$LABEL" --color $COLOR --out "$OUT"
echo "[render] done -> $OUT.mp4"
