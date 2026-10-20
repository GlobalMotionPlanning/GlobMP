#!/bin/bash
# MDOC preliminary grid: 8 maps x 7 team sizes, MDOC shipped defaults.
# MDOC is CPU-only and training-free -- no GPU, no diffusion checkpoint.
# Velocity regimes are a SCORING axis for MDOC (it has no v_max input), so one run
# per (map, n) yields all three regime columns; run_mdoc_cell.py already reports
# max_step_disp + vel_violation_frac against regime --vreg.
#
#   usage: sweep_mdoc_grid.sh <out_dir> <instances> <cores> [maps...]
#   e.g.   sweep_mdoc_grid.sh /path/out 10 112-117 empty basic dense corridor
set -u
OUT=${1:?out_dir}; INST=${2:-10}; CORES=${3:-112-117}; shift 3 || true
MAPS=${*:-"empty basic dense corridor room shelf corridorbay crossroom"}
MDOC=${MRMP_ROOT:?source config/env.sh first}/MDOC
PY=${PY_DIRECT:?source config/env.sh first}
mkdir -p "$OUT"; cd "$MDOC" || exit 1
export CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
for M in $MAPS; do
  for N in 3 6 9 12 15 18 21; do
    tag="${M}_n${N}"
    [ -s "$OUT/$tag.json" ] && { echo "skip $tag"; continue; }
    t0=$(date +%s.%N)
    timeout 3600 taskset -c "$CORES" $PY scripts/inference/run_mdoc_cell.py \
        --map "$M" --n "$N" --instances "$INST" \
        --samples 64 --dsteps 100 --horizon 64 \
        --hp PP --rl 180 --no-compile \
        --out "$OUT/$tag.json" > "$OUT/$tag.log" 2>&1
    echo "$tag rc=$? wall=$(echo "$(date +%s.%N)-$t0"|bc)"
  done
done
