#!/usr/bin/env bash
# run_baseline.sh <groot|pi05|tvla|dp|act> <task> <gpu|gpu,gpu,...> [--tactile]
# tvla: Tactile-VLA (pi0.5 + MLP tactile prefix token, finetune/tactile_vla.py), always tactile, output runs/<task>/tactile_vla
# Several GPUs: the TOTAL batch stays at the recipe value (split across GPUs); DP/ACT use $LOADER_WORKERS_PER_GPU loaders per GPU.
# Train (config/recipe.env) -> confirm the final step was reached -> strip resume-only optimizer state
# (evaluated weights untouched) -> open-loop rollout on held-out episodes -> task metrics table.
# Output: $XB_ROOT/runs/<task>/<method>[_tactile]/{ckpt,train.log,rollouts,rollout.log}, runs/<task>/metrics.txt
# XB_SMOKE_STEPS=N: N-step test run under runs/_smoke/ (no metrics).
. "$(dirname "$0")/../config/env.sh"
M=$1; T=$2; GPU=$3; TAC=""; [ "${4:-}" = --tactile ] && TAC=--tactile
D=$XB_ROOT/data/$T; [ -f $D/READY ] || { echo "task $T not prepared (data_prep/prepare_task.sh $T)"; exit 1; }
NG=$(echo $GPU | tr ',' '\n' | wc -l)
[ $M = groot ] && [ $NG -gt 1 ] && { echo "GR00T runs on 1 GPU only (recipe: 1 GPU, batch 16; on 4 GPUs it is 3x slower and ran out of memory)" >&2; exit 1; }; G0=${GPU%%,*}; PORT=$((29600 + G0))   # distinct port per job
RUN=$M${TAC:+_tactile}; [ $M = tvla ] && { [ -n "$TAC" ] && { echo "tvla always uses tactile: drop --tactile" >&2; exit 1; }; RUN=tactile_vla; }
RR=$XB_ROOT/runs${XB_SMOKE_STEPS:+/_smoke}/$T; R=$RR/$RUN; mkdir -p $R
export TOKENIZERS_PARALLELISM=false GROOT_HF_LOCAL_FIRST=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # all models are in the local HF cache
case $M in
  groot) STEPS=$GROOT_STEPS;; pi05|tvla) STEPS=$PI05_STEPS;; dp) STEPS=$DP_STEPS;; act) STEPS=$ACT_STEPS;;
  *) echo "method: groot|pi05|tvla|dp|act"; exit 1;; esac
SAVE=$SAVE_STEPS; [ -n "${XB_SMOKE_STEPS:-}" ] && { STEPS=$XB_SMOKE_STEPS; SAVE=$XB_SMOKE_STEPS; }
[ $M = pi05 -o $M = tvla ] && [ ! -f "$XB_PI05_BASE/model.safetensors" ] && { echo "pi0.5 base model not found: $XB_PI05_BASE" >&2; exit 1; }
LOG=$R/train.log; xb_log "$T/$RUN gpu$GPU: train start ($STEPS steps)"
if [ $M = groot ]; then
  MOD=$XB_CODE/config/modality/xhand${TAC:+_tactile}.py
  CUDA_VISIBLE_DEVICES=$GPU NUM_GPUS=$NG MASTER_PORT=$PORT GLOBAL_BATCH_SIZE=$GROOT_BATCH MAX_STEPS=$STEPS SAVE_STEPS=$SAVE AUTO_EVAL=0 \
    DATALOADER_NUM_WORKERS=$GROOT_WORKERS DATASET=$D/gr00t MODALITY=$MOD OUTDIR=$R/ckpt \
    bash $GR00T_DIR/train/finetune.sh > $LOG 2>&1
else
  PY=$XB_PY; P=$M; [ $M = pi05 -o $M = tvla ] && PY=$XB_PY_PI; [ $M = dp ] && P=diffusion
  case $M in pi05|tvla) B=$PI05_BATCH C=$PI05_CHUNK W=$PI05_WORKERS;; dp) B=$DP_BATCH C=$DP_CHUNK W=$DP_WORKERS;; act) B=$ACT_BATCH C=$ACT_CHUNK W=$ACT_WORKERS;; esac
  XTRA=""; [ $M = tvla ] && XTRA="--tactile-history $TVLA_HISTORY"
  LAUNCH="$PY -u"
  if [ $NG -gt 1 ]; then   # LeRobot's batch size is per process: keep the total at the recipe value
    [ $((B % NG)) = 0 ] || { echo "batch $B not divisible by $NG GPUs" >&2; exit 1; }
    B=$((B / NG)); [ $M = pi05 -o $M = tvla ] && W=$(( (W + NG - 1) / NG )) || W=$LOADER_WORKERS_PER_GPU
    LAUNCH="$(dirname $PY)/accelerate launch --num_processes $NG --main_process_port $PORT"
  fi
  CUDA_VISIBLE_DEVICES=$GPU XB_DATASET=$D/lerobot $LAUNCH $XB_CODE/finetune/train_lerobot.py --policy $P \
    --episodes $D/split/train.json --out $R/ckpt --steps $STEPS --batch-size $B --chunk-size $C --save-freq $SAVE \
    --num-workers $W --seed $SEED $TAC $XTRA > $LOG 2>&1
fi
step=$(tr '\r' '\n' < $LOG | grep -oE "[0-9]+/$STEPS([^0-9]|$)" | cut -d/ -f1 | sort -n | tail -1)   # highest step reached
[ "$step" = "$STEPS" ] || { xb_log "$T/$RUN: TRAIN FAILED at step ${step:-0} (see $LOG)"; exit 1; }
# LeRobot falls back to random weights (with only a warning) when the base model cannot be loaded
grep -q "Returning model without loading pretrained weights" $LOG && { xb_log "$T/$RUN: FAILED, base weights were not loaded (see $LOG)"; exit 1; }
if [ $M = groot ]; then
  for c in $R/ckpt/checkpoint-*; do [ -d "$c" ] || continue
    [ -f $R/ckpt/model.safetensors.index.json ] && cmp -s $R/ckpt/model-00001-of-00003.safetensors $c/model-00001-of-00003.safetensors \
      && rm -r "$c" || xb_log "$T/$RUN: kept $c (final weights not verified)"; done
  CUDA_VISIBLE_DEVICES=$G0 PYTHONPATH=$GR00T_DIR $GR00T_DIR/.venv/bin/python -u $XB_CODE/eval/rollout_gr00t.py --dataset $D/gr00t \
    --episodes $D/split/heldout.json --ckpt $R/ckpt $TAC --modality-config $MOD --outdir $R/rollouts --device cuda:0 > $R/rollout.log 2>&1
else
  for t in $R/ckpt/checkpoints/*/training_state; do [ -d "$t" ] && [ -f "$(dirname $t)/pretrained_model/model.safetensors" ] && rm -r "$t"; done
  CUDA_VISIBLE_DEVICES=$G0 XB_DATASET=$D/lerobot $PY -u $XB_CODE/eval/rollout_lerobot.py --policy $P $TAC \
    --ckpt $R/ckpt/checkpoints/last/pretrained_model --dataset $D/lerobot --episodes $D/split/heldout.json \
    --outdir $R/rollouts > $R/rollout.log 2>&1
fi
n=$(ls $R/rollouts/*.npz 2>/dev/null | wc -l); want=$(python3 -c "import json;print(len(json.load(open('$D/split/heldout.json'))))")
[ "$n" = "$want" ] && xb_log "$T/$RUN: trained ($(du -sh $R/ckpt | cut -f1)), eval $n/$want" || { xb_log "$T/$RUN: EVAL INCOMPLETE $n/$want (see $R/rollout.log)"; exit 1; }
[ -z "${XB_SMOKE_STEPS:-}" ] && ( flock $RR/.metrics.lock $XB_PY $XB_CODE/eval/metrics.py $RR > $RR/metrics.txt 2>&1 )
xb_log "$T/$RUN: DONE"
