#!/usr/bin/env bash
# finetune_trex.sh <task> <gpus a,b,c,d> [--tactile] [port]
# T-Rex XHand post-train from midtrain (config/recipe.env) -> final checkpoint check -> held-out rollout -> metrics.
# Output: $XB_ROOT/runs/<task>/trex[_tactile]/{ckpt,train.log,rollouts,rollout.log}. XB_SMOKE_STEPS=N for a test run.
# Optional: XB_TREX_VQ=<tactile VQ-VAE .pt> (default T-Rex/tactile_vqvae/checkpoints/xhand) and XB_TREX_TAG=<name> (run dir trex_tactile_<name>).
. "$(dirname "$0")/../config/env.sh"
[ $# -ge 2 ] || { sed -n 2p "$0"; exit 1; }
T=$1; GPUS=$2; TAC=0; PORT=29530
for a in "${@:3}"; do case $a in --tactile) TAC=1;; *) PORT=$a;; esac; done
D=$XB_ROOT/data/$T; [ -f $D/READY ] || { echo "task $T not prepared"; exit 1; }
VQ=${XB_TREX_VQ:-$TREX_DIR/tactile_vqvae/checkpoints/xhand/latest.pt}
for f in "$XB_QWEN" "$XB_TREX_MIDTRAIN" $( [ $TAC = 1 ] && echo $VQ ); do [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }; done
RUN=trex; [ $TAC = 1 ] && RUN=trex_tactile${XB_TREX_TAG:+_$XB_TREX_TAG}; RR=$XB_ROOT/runs${XB_SMOKE_STEPS:+/_smoke}/$T; R=$RR/$RUN; mkdir -p $R
STEPS=${XB_SMOKE_STEPS:-$TREX_STEPS}; NP=$(echo $GPUS | tr ',' '\n' | wc -l)
if [ $TAC = 1 ]; then TA="--use_tactile_vec 1 --use_tactile_vqvae 1 --vqvae_ckpt $VQ --tacf6_dim $TREX_TACF_DIM"
else TA="--use_tactile_vec 0 --use_tactile_vqvae 0 --auto_vqvae 0 --tacf6_dim $TREX_TACF_DIM"; fi
xb_log "$T/$RUN gpus$GPUS: train start ($STEPS steps)"
( cd $TREX_DIR/scripts && PYTHONPATH=$TREX_DIR WANDB_MODE=offline CUDA_VISIBLE_DEVICES=$GPUS TOKENIZERS_PARALLELISM=false \
  $(dirname $XB_PY)/accelerate launch --config_file ../config/sft_qwen.yaml --num_processes $NP --main_process_port $PORT \
  train.py --model_path $XB_QWEN --data_format lerobot --lerobot_root $D/trex42_train \
  --max_steps $STEPS --save_freq 1000000 --seed $SEED --action_dim $TREX_ACTION_DIM --action_chunk $TREX_CHUNK \
  --train_bsz_per_gpu $TREX_BATCH_PER_GPU --learning_rate $TREX_LR --min_lr_ratio 0 --weight_decay 0 --gradient_accumulation_steps 1 \
  --output_dir $R/ckpt --log_dir $R/ckpt --experiment_name trex_xhand --run_name $RUN \
  --use_robot_state 0 --use_tactile_deform 0 $TA \
  --tactile_intermediate_size 1536 --training_stage 2 --tactile_loss_weight 1.0 \
  --cascaded_total_steps 10 --cascaded_split_step 6 --cascaded_tactile_dropout 0.1 --cascaded_loss_weight 1.0 \
  --resume_checkpoint $XB_TREX_MIDTRAIN --resume_source midtrain \
  --use_flare 1 --n_flare_tokens_per_frame 4 --n_flare_steps 8 --flare_loss_weight 0.5 \
  --flare_frame_stride 4 --flare_layer_index -1 --image_size 384 288 --val_ratio 0 ) > $R/train.log 2>&1
CK=$(ls -d $R/ckpt/*/*/checkpoint-*-$STEPS 2>/dev/null | tail -1)
[ -n "$CK" ] && [ -f $CK/model.pt ] || { xb_log "$T/$RUN: TRAIN FAILED (no checkpoint-*-$STEPS; see $R/train.log)"; exit 1; }
CUDA_VISIBLE_DEVICES=${GPUS%%,*} PYTHONPATH=$TREX_DIR $XB_PY -u $XB_CODE/eval/rollout_trex.py \
  --ckpt $CK --dataset $D/trex42_heldout --outdir $R/rollouts --tactile $TAC > $R/rollout.log 2>&1
n=$(ls $R/rollouts/*.npz 2>/dev/null | wc -l); want=$(python3 -c "import json;print(len(json.load(open('$D/split/heldout.json'))))")
[ "$n" = "$want" ] && xb_log "$T/$RUN: trained ($(du -sh $CK | cut -f1)), eval $n/$want" || { xb_log "$T/$RUN: EVAL INCOMPLETE $n/$want"; exit 1; }
[ -z "${XB_SMOKE_STEPS:-}" ] && ( flock $RR/.metrics.lock $XB_PY $XB_CODE/eval/metrics.py $RR > $RR/metrics.txt 2>&1 )
xb_log "$T/$RUN: DONE"
