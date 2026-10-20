#!/bin/bash
# =============================================================================
# SCENARIO-1 PILOT -- prove the whole pipeline on ONE cell before spending days
# =============================================================================
#
# The hard row needs every learned baseline REGENERATED, because the maps changed:
# new training trajectories, then new weights, then evaluation. That is ~56 SMD models
# (8 maps x 7 team sizes) plus per-map models for the other baselines. Launching that
# blind is how a three-day run dies on hour two, so this runs one cell end to end:
#
#     scenario -> generate trajectories -> train -> evaluate -> score
#
# Every stage checkpoints, so a rerun skips what already succeeded, and every stage is
# VERIFIED rather than assumed: a stage that produces no output fails loudly instead of
# letting the next stage run on nothing.
#
# The pilot cell defaults to crossroomhard-3: a map that did not exist an hour ago, at an
# inflated geometry, with the smallest team. That exercises the new env class, the new
# scenarios entry, the OBS_SCALE plumbing through data generation AND training, and the
# separate hard-row model directory -- i.e. everything that is new.
#
# Training length is cut to SMD_TRAIN_STEPS (default 3000 vs the real 60000) because the
# point is to prove the plumbing, not to produce a usable model. The real run drops the
# override.
#
# USAGE
#   bash run_scenario1_pilot.sh                       # crossroomhard-3, 3000 steps
#   TAG=corridorbayhard-9 bash run_scenario1_pilot.sh
#   SMD_TRAIN_STEPS=60000 bash run_scenario1_pilot.sh  # full-length single cell
# =============================================================================
set -uo pipefail

BASE="${SMD_BASE:?set SMD_BASE}"
SR="$BASE/Diffusion-MRMP/scripts/repro"
DM="$BASE/Diffusion-MRMP/data_trained_models"
DT="$BASE/Diffusion-MRMP/data_trajectories"
LOGS="$BASE/logs_pilot"
TAG="${TAG:-crossroomhard-3}"
export SMD_TRAIN_STEPS="${SMD_TRAIN_STEPS:-3000}"
# evaluation dominates the pilot otherwise: 10 instances x ~6 min at N=3 is an hour
export SMD_N_TEST="${SMD_N_TEST:-2}"
export BOX_COVER_CELL="${BOX_COVER_CELL:-0.0625}"

# ---- resource allocation (added for this machine: 2x EPYC 7502 / 8x A100-40GB) --------
# Scripts call get_torch_device('cuda'), which takes cuda:0. On a shared box that is
# whatever GPU is physically first -- here GPUs 0,1,2,6 are another user's job at ~99%.
# CUDA_VISIBLE_DEVICES remaps, so cuda:0 becomes the GPU named here and nothing else.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
# BLAS/OMP default to one thread per core (128), so every extra process oversubscribes
# and the stages spend their time in the scheduler. Pin to the cpuset we were given.
export SMD_THREADS="${SMD_THREADS:-16}"
export OMP_NUM_THREADS="$SMD_THREADS"
export MKL_NUM_THREADS="$SMD_THREADS"
export OPENBLAS_NUM_THREADS="$SMD_THREADS"
export NUMEXPR_NUM_THREADS="$SMD_THREADS"
export TORCH_NUM_THREADS="$SMD_THREADS"
# A100 is sm_80: TF32 matmuls are a free ~2x on the UNet with no accuracy cost here.
export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-1}"
# keeps the 40 GB card from fragmenting across the long generation stage
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$LOGS"

log(){ echo "[$(date '+%F %T')] $*"; }
fail(){ echo "[$(date '+%F %T')] FAIL: $*" >&2; exit 1; }

set +u; source "$BASE/miniconda3/etc/profile.d/conda.sh"; conda activate smd; set -u
cd "$SR" || fail "cannot cd $SR"

# ---- 0. resolve the cell and export its geometry -----------------------------------
read -r SUBDIR OBS RR < <(python3 - "$TAG" <<'PYEOF'
import sys
sys.path.insert(0, '.')
from scenarios import get_scenario
c = get_scenario(sys.argv[1])
print(c['dataset_subdir'], c.get('obs_scale', 1.0), c.get('robot_r', 0.05))
PYEOF
) || fail "unknown tag $TAG"
export OBS_SCALE="$OBS" ROBOT_R="$RR"
log "cell $TAG"
log "  dataset_subdir $SUBDIR"
log "  OBS_SCALE=$OBS  ROBOT_R=$RR  train_steps=$SMD_TRAIN_STEPS  test_instances=$SMD_N_TEST"

# ---- 1. geometry sanity: is this map actually traversable at this setting? ----------
python3 - "$TAG" <<'PYEOF' || fail "geometry check failed"
import sys
sys.path.insert(0, '.')
import numpy as np, torch
from scenarios import get_scenario
from generate_via_pp import build_reference
from tune_hard_settings import r_crit
c = get_scenario(sys.argv[1])
env, robot, task = build_reference(c, {'device': 'cpu', 'dtype': torch.float32})
G = 340; xs = np.linspace(-1, 1, G); X, Y = np.meshgrid(xs, xs)
pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], -1), dtype=torch.float32)
with torch.no_grad():
    S = env.compute_sdf(pts.reshape(1, -1, 2)).reshape(G, G).numpy()
S = np.minimum(S, np.minimum(1 - np.abs(X), 1 - np.abs(Y)))   # wall counts
r = float(robot.radius)
free = float((S > r).mean()); rc = r_crit(S)
print(f"  geometry: free {100*free:.0f}%   r_crit {rc:.3f} = {rc/r:.1f}x robot r (r={r})")
if free < 0.02:
    print("  !! map has almost no free space at this setting"); sys.exit(1)
# REACHABILITY GATE. Three criteria were tried before this one:
#   r_crit >= r        -- too strict (RUN_NOTE section 2): room/shelf/corridorbay all sit
#                         below the planner's 0.105 and are fine.
#   no gate            -- too lax: let crossroom x2.20 through, whose free space is four
#                         sealed quadrants. PP then fails ~99% and no data can fix it.
#   one component      -- also too strict: shipped Basic is 95% + pockets, and shipped
#                         Dense is genuinely bisected (49/35) yet is the benchmark's own
#                         baseline. Rejecting those is the false positive section 2 warns of.
# What predicts the collapse is the chance two random free points are mutually reachable,
# P_same = sum(component_fraction^2); P_same^3 approximates a solvable 3-robot instance.
# Measured: crossroom x2.20 -> P^3 = 0.016 against an observed 1.0% PP yield.
# Judge it RELATIVE to the same map at 1.0, so inherently-hard maps keep their baseline.
from scipy import ndimage
def _p_same(_S, _clr):
    _f = _S > _clr
    _l, _n = ndimage.label(_f)
    if _n == 0: return 0.0
    _p = ndimage.sum(_f, _l, range(1, _n + 1)) / _f.sum()
    return float((_p ** 2).sum())
clearance = r + 0.055                       # PlanningTask(obstacle_cutoff_margin=0.05)
p_hard = _p_same(S, clearance)
import importlib as _il, os as _os
_os.environ['OBS_SCALE'] = '1.0'
_gp = _il.reload(__import__('generate_via_pp'))
_c = dict(c); _c['obs_scale'] = 1.0
_e, _r2, _t = _gp.build_reference(_c, {'device': 'cpu', 'dtype': torch.float32})
with torch.no_grad():
    _S = _e.compute_sdf(pts.reshape(1, -1, 2)).reshape(G, G).numpy()
_S = np.minimum(_S, np.minimum(1 - np.abs(X), 1 - np.abs(Y)))
p_ship = _p_same(_S, clearance)
print(f"  reachability: P_same {p_hard:.2f} (shipped {p_ship:.2f})   "
      f"P^3 {p_hard**3:.3f} vs {p_ship**3:.3f}")
if p_ship > 0 and (p_hard ** 3) < 0.5 * (p_ship ** 3):
    print(f"  !! this scale collapses reachability to {100*(p_hard/p_ship)**3:.0f}% of the shipped map.")
    print( "     Most start/goal pairs land in mutually unreachable regions, so generation")
    print( "     fails regardless of model quality. Lower obs_scale for this map.")
    sys.exit(1)
PYEOF

# ---- 2a. single-agent trajectories (what the multi-agent generator is built from) ----
# Missed on the first pass: generate_via_pp does not plan from scratch, it composes
# SINGLE-AGENT trajectories. A hard cell needs its OWN single-agent data -- reusing the
# shipped set would train on paths planned through space that is no longer free.
SA_KEY="$(python3 - "$TAG" <<'PYEOF'
import sys; sys.path.insert(0, '.')
from scenarios import get_scenario
print(get_scenario(sys.argv[1])['scenario'])
PYEOF
)"
SA_DIR="$DT/singleagent_${SA_KEY}_$(python3 - "$SA_KEY" <<'PYEOF'
import sys; sys.path.insert(0, '.')
from generate_singleagent_data import SCENARIOS as SA, ROBOT_ID
print(f"{SA[sys.argv[1]]['env_id']}-{ROBOT_ID}")
PYEOF
)"
# Every stage below skips when its artifacts exist, which is what makes a rerun cheap --
# but nothing recorded the geometry they were built at, so an artifact trained at one
# obstacle scale was silently reused at another. That happened: dense's models were
# trained at scale 1.8 and the map has since moved to its published 1.0, and only the
# file dates gave it away. Each artifact directory now carries .obs_scale, and a
# mismatch is fatal rather than a silent reuse.
stamp_scale () {   # stamp_scale <dir>
  [ -d "$1" ] || return 0
  printf '%s\n' "$OBS_SCALE" > "$1/.obs_scale"
}
check_scale () {   # check_scale <dir> <stage label>
  [ -d "$1" ] || return 0
  if [ -f "$1/.obs_scale" ]; then
    was="$(cat "$1/.obs_scale")"
    if [ "$was" != "$OBS_SCALE" ]; then
      log "  FATAL [$2]: $1 was built at OBS_SCALE=$was, this run is $OBS_SCALE."
      log "         Move it aside and rebuild; reusing it would plan in different geometry."
      exit 3
    fi
  elif [ "$(ls -A "$1" 2>/dev/null | wc -l)" -gt 0 ]; then
    # Non-empty and unstamped: refuse. Stamping it here would record THIS run's scale on
    # an artifact built by some earlier run, which is exactly the mismatch the stamp
    # exists to catch. Every artifact present when stamping was introduced was stamped
    # with its map's scale at that point, so an unstamped non-empty directory now means
    # something arrived by another route and its geometry is unverified.
    log "  FATAL [$2]: $1 is not empty and carries no .obs_scale stamp."
    log "         Verify the scale it was built at, write it to $1/.obs_scale, and rerun."
    exit 3
  else
    stamp_scale "$1"
  fi
}
check_scale "$SA_DIR" "1/7"

if [ -d "$SA_DIR" ] && [ "$(ls -A "$SA_DIR" 2>/dev/null | wc -l)" -gt 0 ]; then
  log "  [1/7] single-agent data exists ($(ls "$SA_DIR" | wc -l) tasks) -- skipping"
else
  log "  [1/7] generating single-agent data for $SA_KEY ..."
  # Stage 1 is pure geometry on 64x2 tensors -- it picks 'cuda' if a card is visible, but
  # the kernel-launch queue behind another tenant's saturated A100 makes that ~60x SLOWER
  # than CPU here (measured: 2.8 vs 176 contexts/s). Hide the GPU for this stage only;
  # stages 2/3/5/6 do real diffusion work and keep it.
  CUDA_VISIBLE_DEVICES="" python3 -u generate_singleagent_data.py "$SA_KEY" > "$LOGS/${TAG}_sa.log" 2>&1 \
    || fail "single-agent generation crashed -- see $LOGS/${TAG}_sa.log"
  n=$(ls "$SA_DIR" 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] || fail "single-agent generation produced nothing in $SA_DIR"
  log "  [1/7] ok -- $n tasks"
fi
stamp_scale "$SA_DIR"   # stamp on creation, not on next visit

# ---- 2b. TRAIN the single-agent model --------------------------------------------
# Missed again on the second pass, and it cost the full 40-minute generation budget:
# generate_via_pp does not compose raw trajectories, it samples from a TRAINED
# single-agent diffusion model. Without it every attempt dies on a missing args.yaml --
# measured 527,921 identical failures in 2400 s, zero demonstrations.
SA_MODEL="$DM/singleagent_${SA_KEY}_$(python3 - "$SA_KEY" <<'PYEOF'
import sys; sys.path.insert(0, '.')
from generate_singleagent_data import SCENARIOS as SA, ROBOT_ID
print(f"{SA[sys.argv[1]]['env_id']}-{ROBOT_ID}")
PYEOF
)"
# args.yaml is written BEFORE the first training step, so its presence proves nothing --
# this is the briefing's "checkpoint-existence is not checkpoint-completeness" trap, which
# train_smd_all.py solves with is_really_trained(). A run killed seconds in leaves
# args.yaml + checkpoints/iter_000000 and would be silently accepted as trained.
SA_ITER=$(ls "$SA_MODEL/checkpoints" 2>/dev/null | grep -oE 'iter_[0-9]+' | grep -oE '[0-9]+' \
          | sort -n | tail -1); SA_ITER=${SA_ITER:-0}
check_scale "$SA_MODEL" "2/7"
if [ -f "$SA_MODEL/args.yaml" ] && [ "$SA_ITER" -gt 0 ]; then
  log "  [2/7] single-agent model exists (iter $SA_ITER) -- skipping"
else
  log "  [2/7] training single-agent model for $SA_KEY ..."
  SMD_TRAIN_STEPS="$SMD_TRAIN_STEPS" python3 -u train_singleagent_all.py "$SA_KEY" \
    > "$LOGS/${TAG}_satrain.log" 2>&1 \
    || fail "single-agent training crashed -- see $LOGS/${TAG}_satrain.log"
  [ -f "$SA_MODEL/args.yaml" ] || fail "single-agent training left no args.yaml in $SA_MODEL"
  log "  [2/7] ok"
fi
stamp_scale "$SA_MODEL"   # stamp on creation, not on next visit

# ---- 3. generate multi-agent trajectories ------------------------------------------
PPDIR="$DT/pp_$SUBDIR"   # generate_via_pp writes under a pp_ prefix, not $SUBDIR
# likewise a killed generator leaves a handful of demos; require a meaningful count
PP_MIN="${SMD_PP_MIN_KEEP:-50}"
check_scale "$PPDIR" "3/7"
if [ -d "$PPDIR" ] && [ "$(ls -A "$PPDIR" 2>/dev/null | wc -l)" -ge "$PP_MIN" ]; then
  log "  [3/7] multi-agent data exists ($(ls "$PPDIR" | wc -l) entries) -- skipping"
else
  log "  [3/7] generating multi-agent trajectories ..."
  python3 -u generate_via_pp.py "$TAG" > "$LOGS/${TAG}_gen.log" 2>&1 &
  GPID=$!
  # generate_via_pp catches per-attempt exceptions and retries. A FATAL error (a missing
  # model, a bad path) therefore repeats until the time budget expires rather than
  # stopping -- 527,921 identical failures over 40 minutes on the previous run. Abort as
  # soon as the same exception has repeated with nothing saved.
  while kill -0 $GPID 2>/dev/null; do
    sleep 20
    nerr=$(grep -c "raised" "$LOGS/${TAG}_gen.log" 2>/dev/null); nerr=${nerr:-0}
    nok=$(ls "$PPDIR" 2>/dev/null | wc -l)
    if [ "$nerr" -gt 200 ] && [ "$nok" -eq 0 ]; then
      kill -KILL $GPID 2>/dev/null
      echo "  first error was:"; grep -m1 "raised" "$LOGS/${TAG}_gen.log" | sed 's/^/    /'
      fail "generation is failing on every attempt ($nerr errors, 0 saved) -- aborted early"
    fi
  done
  wait $GPID || fail "generation crashed -- see $LOGS/${TAG}_gen.log"
  n=$(ls "$PPDIR" 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] || fail "generation produced no data in $PPDIR"
  log "  [3/7] ok -- $n entries"
fi
stamp_scale "$PPDIR"   # stamp on creation, not on next visit

# ---- 3b. merge datasets -------------------------------------------------------------
# The evaluated models live in data_trained_models/merged_* and train on
# data_trajectories/merged_<subdir>, which merge_datasets.py builds from the from-scratch
# set <subdir> AND the PP set pp_<subdir>. Training straight off the PP output is what
# crashed the last run: train_smd_all.py reads <subdir>, which a new map does not have.
# For a new map the merge is PP-only, which is fine -- it renumbers into the contiguous
# 0..N-1 layout TrajectoryDatasetBase requires.
MERGED_DATA="$DT/merged_$SUBDIR"
check_scale "$MERGED_DATA" "4/7"
if [ -d "$MERGED_DATA" ] && [ "$(ls -A "$MERGED_DATA" 2>/dev/null | wc -l)" -gt 0 ]; then
  log "  [4/7] merged dataset exists ($(ls "$MERGED_DATA" | wc -l) tasks) -- skipping"
else
  log "  [4/7] merging datasets ..."
  python3 -u merge_datasets.py "$TAG" > "$LOGS/${TAG}_merge.log" 2>&1 \
    || fail "merge crashed -- see $LOGS/${TAG}_merge.log"
  m=$(ls "$MERGED_DATA" 2>/dev/null | wc -l)
  [ "$m" -gt 0 ] || fail "merge produced nothing in $MERGED_DATA"
  log "  [4/7] ok -- $m tasks"
fi
stamp_scale "$MERGED_DATA"   # stamp on creation, not on next visit

# ---- 4. train the composite model ---------------------------------------------------
CKPT="$DM/merged_$SUBDIR/checkpoints"
if [ -f "$CKPT/ema_model_current_state_dict.pth" ]; then
  log "  [5/7] model exists -- skipping"
else
  log "  [5/7] training ($SMD_TRAIN_STEPS steps) ..."
  python3 -u train_smd_merged_all.py "$TAG" > "$LOGS/${TAG}_train.log" 2>&1 \
    || fail "training crashed -- see $LOGS/${TAG}_train.log"
  [ -f "$CKPT/ema_model_current_state_dict.pth" ] || fail "training left no checkpoint in $CKPT"
  log "  [5/7] ok -- $(ls "$CKPT" | wc -l) checkpoint files"
fi

# ---- 4. evaluate --------------------------------------------------------------------
log "  [6/7] evaluating ..."
python3 -u evaluate_smd_C.py "$TAG" > "$LOGS/${TAG}_eval.log" 2>&1 \
  || fail "evaluation crashed -- see $LOGS/${TAG}_eval.log"

# ---- 5. report ----------------------------------------------------------------------
log "  [7/7] result:"
python3 - "$TAG" "$DM" <<'PYEOF' || fail "no result file written"
import glob, json, os, sys
sys.path.insert(0, '.')
from scenarios import get_scenario
c = get_scenario(sys.argv[1])
d = os.path.join(sys.argv[2], f"merged_{c['model_id']}")
if not os.path.isdir(d):
    d = os.path.join(sys.argv[2], c['model_id'])
fs = sorted(glob.glob(os.path.join(d, 'eval_results_C_*.json')))
if not fs:
    print(f"  no eval_results_C_*.json under {d}"); sys.exit(1)
r = json.load(open(fs[-1]))
print(f"    S={r.get('success_rate')}  L={r.get('avg_path_length_over_success')}  "
      f"A={r.get('metric_A')}  C={r.get('metric_C_robot')}  n={r.get('n_instances')}")
PYEOF
log "=== PILOT PASSED for $TAG ==="
