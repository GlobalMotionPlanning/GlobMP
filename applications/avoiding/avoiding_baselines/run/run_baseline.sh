#!/usr/bin/env bash
# run_baseline.sh <policy> [gpu] [repeats]      (policies: config/policies.tsv)
# One baseline on avoiding-v0 with the shared planner (AB_WEIGHTS), 12 constraints.
# Flags per policy are DiRecT's own eval_all.sbatch; D3IL_HARD=1 turns on the three
# keep-outs upstream ships commented out, which is the setting Table 3 reports.
. "$(dirname "$0")/../config/env.sh"
P=$1; GPU=${2:-0}; R=${3:-$REPEAT}
row=$(ab_policy "$P"); [ -n "$row" ] || { echo "unknown policy $P (config/policies.tsv)" >&2; exit 1; }
EXPNAME=$(printf '%s' "$row" | cut -f3)
[ -f "$AB_WEIGHTS" ] || { echo "planner weights not found: $AB_WEIGHTS" >&2; exit 1; }
[ -f "$AB_RUNS/avoiding-v0/dynamics/linear_model.npz" ] || {
  echo "no dynamics model: run data_prep/fit_dynamics.sh" >&2; exit 1; }
# eval.py resolves the planner by convention, not by a flag:
#   <log-folder>/<env>/diffusion/<diffusion-exp-name>/model_ema_<cp>.pth
mkdir -p "$AB_RUNS/avoiding-v0/diffusion/x0pred"
ln -sf "$AB_WEIGHTS" "$AB_RUNS/avoiding-v0/diffusion/x0pred/model_ema_$CP.pth"

COMMON="--diffusion-exp-name x0pred --diffusion-cp $CP --seed $SEED \
  --random-repeat $R --batch-size 1 --replan-steps $REPLAN \
  --n-sampling-steps $STEPS --obstacle-margin $MARGIN \
  --pd-kp 0.0 --pd-kd 0.0 --pd-ki 0.0 --no-early-stop"

case "$P" in
  # Unconstrained, but still --constraints: that flag selects the constraint set
  # used for SCORING as well as for guidance (eval.py:581). Without it the
  # baseline is measured against nothing and reads safe. Upstream's
  # eval_all.sbatch omits it here; the runs behind Table 3 do not.
  no)        ARGS="--guidance-method no --constraints" ;;
  gradient)  ARGS="--guidance-method gradient --constraints --guidance-lr 0.05 --guidance-steps 5 \
                   --gradient-start 0.0 --obstacle-weight 100.0 --dynamics-weight 10.0" ;;
  classifier) ARGS="--guidance-method classifier --constraints --guidance-lr 0.05 \
                   --gradient-start 0.0 --obstacle-weight 100.0 --dynamics-weight 10.0" ;;
  safediffuser-ros) ARGS="--guidance-method safediffuser --constraints --n-sampling-steps $STEPS_LONG \
                   --cbf-algorithm RoS --cbf-coefficient 1.0" ;;
  safediffuser-res) ARGS="--guidance-method safediffuser --constraints --n-sampling-steps $STEPS_LONG \
                   --cbf-algorithm ReS --cbf-coefficient 1.0 --cbf-slack-weight 1000.0 --cbf-relaxation-end 0.9" ;;
  safediffuser-tvs) ARGS="--guidance-method safediffuser --constraints --n-sampling-steps $STEPS_LONG \
                   --cbf-algorithm TVS --cbf-coefficient 1.0 --cbf-sigmoid-bias 0.9 \
                   --cbf-sigmoid-input 256.0 --cbf-sigmoid-output 1.0" ;;
  # The two Lagrangian methods need --guidance-last-steps 50; at the default they
  # crash the simulator. Table 3 brackets their rows for exactly this reason.
  primal-dual) ARGS="--guidance-method primal-dual --constraints --n-sampling-steps $STEPS_LONG \
                   --pd-learning-rate 0.456 --no-pd-use-cbfs --guidance-last-steps 50" ;;
  augmented-lagrangian) ARGS="--guidance-method augmented-lagrangian --constraints --n-sampling-steps $STEPS_LONG \
                   --pd-learning-rate 0.456 --pd-initial-penalty 1e-6 --pd-final-penalty 1e-5 \
                   --no-pd-use-cbfs --guidance-last-steps 50" ;;
  projection) ARGS="--guidance-method projection --constraints --dynamics-relaxation 0 --projection-start 0.5" ;;
  direct)    ARGS="--guidance-method direct --constraints --dynamics-relaxation 0 \
                   --pred-control-weight 1.0 --pred-objective '' --pred-objective-scale 10.0 \
                   --guidance-last-steps 16" ;;
  *) echo "no flag set for $P" >&2; exit 1 ;;
esac

cd "$DIRECT_DIR" && ab_log "$P -> $EXPNAME gpu $GPU ($R episodes, D3IL_HARD=$HARD)" && \
  D3IL_HARD=$HARD CUDA_VISIBLE_DEVICES=$GPU eval "$AB_PY eval.py $COMMON $ARGS \
    --exp-name $EXPNAME --device cuda:0 --log-folder $AB_RUNS"
