#!/usr/bin/env bash
# finetune_vitacformer.sh <task> <gpu|gpu,gpu,...> [--no-tactile] [port]
# ViTacFormer from ImageNet ResNet-18 (config/recipe.env; total batch split across GPUs, torchrun/DDP)
# -> final checkpoint check -> held-out rollout -> metrics.
# Output: $XB_ROOT/runs/<task>/vitacformer[_notac]/{ckpt,train.log,rollouts,rollout.log}. XB_SMOKE_STEPS=N for a test run.
. "$(dirname "$0")/../config/env.sh"
[ $# -ge 2 ] || { sed -n 2p "$0"; exit 1; }
T=$1; GPUS=$2; TAC=""; G0=${GPUS%%,*}; PORT=$((29700 + G0))   # distinct port per job
for a in "${@:3}"; do case $a in --no-tactile) TAC=--no-tactile;; *) PORT=$a;; esac; done
D=$XB_ROOT/data/$T; [ -f $D/READY ] || { echo "task $T not prepared (data_prep/prepare_task.sh $T)"; exit 1; }
[ -f "$VTF_DIR/train_xhand.py" ] || { echo "missing ViTacFormer checkout with the XHand adapter: $VTF_DIR" >&2; exit 1; }
RUN=vitacformer${TAC:+_notac}; RR=$XB_ROOT/runs${XB_SMOKE_STEPS:+/_smoke}/$T; R=$RR/$RUN; mkdir -p $R
STEPS=${XB_SMOKE_STEPS:-$VTF_STEPS}; SAVE=${XB_SMOKE_STEPS:-$SAVE_STEPS}; NP=$(echo $GPUS | tr ',' '\n' | wc -l)
[ $((VTF_BATCH % NP)) = 0 ] || { echo "batch $VTF_BATCH not divisible by $NP GPUs" >&2; exit 1; }
xb_log "$T/$RUN gpus$GPUS: train start ($STEPS steps)"
CUDA_VISIBLE_DEVICES=$GPUS $(dirname $XB_PY)/torchrun --nproc_per_node $NP --master_port $PORT $VTF_DIR/train_xhand.py \
  --dataset $D/lerobot --episodes $D/split/train.json --out $R/ckpt --steps $STEPS --batch-size $VTF_BATCH --lr $VTF_LR \
  --chunk $VTF_CHUNK --save-freq $SAVE --num-workers $VTF_WORKERS_PER_GPU --seed $SEED $TAC > $R/train.log 2>&1
grep -q "^step $STEPS/$STEPS " $R/train.log && [ -f $R/ckpt/policy_last.ckpt ] || { xb_log "$T/$RUN: TRAIN FAILED (see $R/train.log)"; exit 1; }
CUDA_VISIBLE_DEVICES=$G0 VTF_DIR=$VTF_DIR $XB_PY -u $XB_CODE/eval/rollout_vitacformer.py --ckpt $R/ckpt/policy_last.ckpt \
  --dataset $D/lerobot --episodes $D/split/heldout.json --outdir $R/rollouts > $R/rollout.log 2>&1
n=$(ls $R/rollouts/*.npz 2>/dev/null | wc -l); want=$(python3 -c "import json;print(len(json.load(open('$D/split/heldout.json'))))")
[ "$n" = "$want" ] && xb_log "$T/$RUN: trained ($(du -sh $R/ckpt | cut -f1)), eval $n/$want" || { xb_log "$T/$RUN: EVAL INCOMPLETE $n/$want (see $R/rollout.log)"; exit 1; }
[ -z "${XB_SMOKE_STEPS:-}" ] && ( flock $RR/.metrics.lock $XB_PY $XB_CODE/eval/metrics.py $RR > $RR/metrics.txt 2>&1 )
xb_log "$T/$RUN: DONE"
