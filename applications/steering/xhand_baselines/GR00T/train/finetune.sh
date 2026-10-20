#!/usr/bin/env bash
# Finetune GR00T-N1.7-3B on a DexMate Vega dataset.
#
# Task-agnostic: the task is defined entirely by DATASET + MODALITY (+ TASK/SOURCE for the
# checkpoint name). See train/finetune_pick.sh for a task wrapper.
#
# GR00T FinetuneConfig defaults: lr 1e-4, warmup 0.05, weight_decay 1e-5, max_steps 10000,
# 40-step action chunk, tune projector + diffusion head.
#
#   MAX_STEPS=3 SAVE_STEPS=99999 bash train/finetune.sh         # smoke
#   MODALITY=configs/vega_umi_modality.py bash train/finetune.sh # any embodiment
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$HERE"

# --- paths (override via env) ---
VENV="${VENV:-$HERE/.venv}"
BASE_MODEL="${BASE_MODEL:-$HERE/base_model/GR00T-N1.7-3B}"     # via scripts/download_base.sh
DATASET="${DATASET:-<data>/data/pick}"          # teleop data server (override per host)
MODALITY="${MODALITY:-$HERE/configs/vega_pick_modality.py}"   # per-task modality config
# default OUTDIR follows NAMING.md: ckpt_<task><source><datarange>_<model><traindate>
# set DATA_RANGE (e.g. 260624-260702) to make the default name complete
TASK="${TASK:-pick}"; SOURCE="${SOURCE:-teleop}"; DATA_RANGE="${DATA_RANGE:-unknown}"
CKPT_NAME="ckpt_${TASK}${SOURCE}${DATA_RANGE}_groot3b$(date +%y%m%d)"
OUTDIR="${OUTDIR:-/data2/$USER/dexmate/checkpoints/$CKPT_NAME}"   # checkpoint store (override per host)
BACKBONE="${GROOT_BACKBONE_PATH:-$HERE/backbone/cosmos-reason2-2b}"
FFMPEG_LIBS="${FFMPEG_LIBS:-$HERE/ffmpeg_libs}"

# --- run config: 6 GPUs; gbs=60 is nearest-to-stock(64) divisible by num_gpus ---
NUM_GPUS="${NUM_GPUS:-6}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-60}"
MAX_STEPS="${MAX_STEPS:-10000}"       # GR00T stock default
SAVE_STEPS="${SAVE_STEPS:-1000}"      # GR00T stock default
MASTER_PORT="${MASTER_PORT:-29520}"

export PYTHONPATH="$HERE"             # only this bundle on the path
export GROOT_BACKBONE_PATH="$BACKBONE"
export GROOT_HF_LOCAL_FIRST=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH="$FFMPEG_LIBS:${LD_LIBRARY_PATH:-}"
# wandb: online if WANDB_API_KEY is set, offline otherwise
if [ -z "${WANDB_API_KEY:-}" ]; then
  export WANDB_MODE="offline"
  echo "[groot train] WANDB_API_KEY not set — training without wandb"
else
  export WANDB_MODE="online"
fi
export WANDB_DIR="${WANDB_DIR:-$OUTDIR}"
export TMPDIR="${TMPDIR:-/tmp}"; mkdir -p "$TMPDIR" "$OUTDIR"

echo "[groot train] base=$BASE_MODEL"
echo "[groot train] dataset=$DATASET gpus=$NUM_GPUS gbs=$GLOBAL_BATCH_SIZE max_steps=$MAX_STEPS out=$OUTDIR"

LAUNCH=( gr00t/experiment/launch_finetune.py
  --base_model_path "$BASE_MODEL"
  --dataset_path "$DATASET"
  --embodiment_tag new_embodiment
  --modality_config_path "$MODALITY"
  --num_gpus "$NUM_GPUS"
  --output_dir "$OUTDIR"
  --global_batch_size "$GLOBAL_BATCH_SIZE"
  --max_steps "$MAX_STEPS"
  --save_steps "$SAVE_STEPS"
  --save_total_limit 5
  --learning_rate 1e-4
  --warmup_ratio 0.05
  --weight_decay 1e-5
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-2}" )   # lower to 1 on many-GPU video runs (RAM OOM)

if [ "$NUM_GPUS" = "1" ]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  "$VENV/bin/python" "${LAUNCH[@]}"; rc=$?
else
  "$VENV/bin/python" -m torch.distributed.run --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" "${LAUNCH[@]}"; rc=$?
fi
echo "[groot train] train rc=$rc"

# --- auto-eval: render GT-vs-model Genesis videos from the final checkpoint ---
# Runs after every successful train. Opt out with AUTO_EVAL=0. Needs the genesis env
# (eval/requirements-genesis.txt) + URDF (scripts/download_urdf.sh); skips with a note if absent.
if [ "$rc" = "0" ] && [ "${AUTO_EVAL:-1}" = "1" ]; then
  CKPT="$(ls -d "$OUTDIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)"
  if [ -n "$CKPT" ]; then
    # training writes processor/ + experiment_cfg/ at the output root; the renderer needs them
    # inside the checkpoint dir it loads.
    [ -d "$OUTDIR/processor" ]      && [ ! -e "$CKPT/processor" ]      && cp -r "$OUTDIR/processor"      "$CKPT/processor"
    [ -d "$OUTDIR/experiment_cfg" ] && [ ! -e "$CKPT/experiment_cfg" ] && cp -r "$OUTDIR/experiment_cfg" "$CKPT/experiment_cfg"
    echo "[groot train] auto-eval: rendering episodes '${EVAL_EPISODES:-0 1}' from $CKPT"
    for ep in ${EVAL_EPISODES:-0 1}; do
      VENV="$VENV" GROOT_BACKBONE_PATH="$BACKBONE" FFMPEG_LIBS="$FFMPEG_LIBS" \
        MODEL="$CKPT" DATASET="$DATASET" EPISODE="$ep" \
        OUT="$OUTDIR/eval/gt_vs_groot_pick_ep${ep}" \
        bash "$HERE/eval/render_gt_vs_model.sh" || echo "[groot train] auto-eval ep$ep skipped (genesis env/URDF not set up?)"
    done
    echo "[groot train] auto-eval videos -> $OUTDIR/eval/"
  fi
fi
exit "$rc"
