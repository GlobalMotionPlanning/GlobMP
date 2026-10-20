#!/usr/bin/env bash
# finetune_rdp.sh <task> <gpu|gpu,gpu,...> [--no-tactile]
# Reactive Diffusion Policy (RDP_DIR, config/recipe.env): stage 1 Asymmetric Tokenizer (first GPU) -> stage 2 Latent
# Diffusion Policy (accelerate over all GPUs; the TOTAL batch stays at the recipe value) -> final-step check -> strip
# resume-only state (evaluated EMA weights untouched) -> open-loop rollout on held-out episodes -> task metrics table.
# Output: $XB_ROOT/runs/<task>/rdp/{ckpt/{at,ldp},train.log,rollouts,rollout.log}. XB_SMOKE_STEPS=N: N steps per stage
# under runs/_smoke/ (no metrics). Tactile is part of the method (fast AT decoder + slow obs); --no-tactile is the ablation:
# tactile inputs replaced by zeros in both stages, architecture unchanged, output runs/<task>/rdp_notac.
. "$(dirname "$0")/../config/env.sh"
[ $# -ge 2 ] || { sed -n 2p "$0"; exit 1; }
T=$1; GPU=$2; NOTAC=""; [ "${3:-}" = --no-tactile ] && NOTAC=1; D=$XB_ROOT/data/$T; [ -f $D/READY ] || { echo "task $T not prepared (data_prep/prepare_task.sh $T)"; exit 1; }
NG=$(echo $GPU | tr ',' '\n' | wc -l); G0=${GPU%%,*}; PORT=$((29700 + G0))   # distinct port per job
B=$RDP_BATCH; [ $((B % NG)) = 0 ] || { echo "batch $B not divisible by $NG GPUs" >&2; exit 1; }
RR=$XB_ROOT/runs${XB_SMOKE_STEPS:+/_smoke}/$T; RUN=rdp${NOTAC:+_notac}; R=$RR/$RUN; mkdir -p $R/ckpt
AT_STEPS=${XB_SMOKE_STEPS:-$RDP_AT_STEPS}; STEPS=${XB_SMOKE_STEPS:-$RDP_STEPS}
export PYTHONPATH=$RDP_DIR:$RDP_SITE HYDRA_FULL_ERROR=1 WANDB_MODE=disabled WANDB_DIR=$R/ckpt TOKENIZERS_PARALLELISM=false
ARGS="task.dataset_path=$D/lerobot task.episodes=$D/split/train.json training.seed=$SEED${NOTAC:+ +task.dataset.zero_tactile=true}"
LOG=$R/train.log; AT=$R/ckpt/at/checkpoints/latest.ckpt; CK=$R/ckpt/ldp/checkpoints/latest.ckpt
xb_log "$T/$RUN gpu$GPU: train start (AT $AT_STEPS + LDP $STEPS steps)"
CUDA_VISIBLE_DEVICES=$G0 $XB_PY -u $RDP_DIR/train.py --config-name=train_at_xhand $ARGS training.max_steps=$AT_STEPS \
  dataloader.batch_size=$RDP_AT_BATCH hydra.run.dir=$R/ckpt/at > $LOG 2>&1
grep -q "\[train\] done $AT_STEPS/$AT_STEPS" $LOG && [ -f $AT ] || { xb_log "$T/$RUN: AT TRAIN FAILED (see $LOG)"; exit 1; }
LAUNCH="$XB_PY -u"; [ $NG -gt 1 ] && LAUNCH="$(dirname $XB_PY)/accelerate launch --multi_gpu --num_processes $NG --main_process_port $PORT"
CUDA_VISIBLE_DEVICES=$GPU $LAUNCH $RDP_DIR/train.py --config-name=train_ldp_xhand $ARGS at_load_dir=$AT training.max_steps=$STEPS \
  dataloader.batch_size=$((B / NG)) dataloader.num_workers=$RDP_WORKERS_PER_GPU hydra.run.dir=$R/ckpt/ldp >> $LOG 2>&1
grep -q "\[train\] done $STEPS/$STEPS" $LOG && [ -f $CK ] || { xb_log "$T/$RUN: LDP TRAIN FAILED (see $LOG)"; exit 1; }
$XB_PY -c "import sys, os, torch, dill
p = torch.load(sys.argv[1], pickle_module=dill, map_location='cpu', weights_only=False)
for k in ('model', 'optimizer'): p['state_dicts'].pop(k, None)
torch.save(p, sys.argv[1] + '.tmp', pickle_module=dill); os.replace(sys.argv[1] + '.tmp', sys.argv[1])" $CK
CUDA_VISIBLE_DEVICES=$G0 $XB_PY -u $XB_CODE/eval/rollout_rdp.py --ckpt $CK --at-ckpt $AT --dataset $D/lerobot \
  --episodes $D/split/heldout.json --outdir $R/rollouts > $R/rollout.log 2>&1
n=$(ls $R/rollouts/*.npz 2>/dev/null | wc -l); want=$(python3 -c "import json;print(len(json.load(open('$D/split/heldout.json'))))")
[ "$n" = "$want" ] && xb_log "$T/$RUN: trained ($(du -sh $R/ckpt | cut -f1)), eval $n/$want" || { xb_log "$T/$RUN: EVAL INCOMPLETE $n/$want (see $R/rollout.log)"; exit 1; }
[ -z "${XB_SMOKE_STEPS:-}" ] && ( flock $RR/.metrics.lock $XB_PY $XB_CODE/eval/metrics.py $RR > $RR/metrics.txt 2>&1 )
xb_log "$T/$RUN: DONE"
