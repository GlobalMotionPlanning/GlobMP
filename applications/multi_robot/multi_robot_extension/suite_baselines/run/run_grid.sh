#!/bin/bash
# run_grid.sh [workers] [gpu-list] : the 8 methods x 8 maps x N in 3..21 x 3 caps (1232 cells: 7 methods x 168 plus MMD-CBS x 56; MMD-CBS has no velocity
# limit and is planned at one cap). About 92 GPU-hours, over half of it MDOC and a quarter DiRecT; ~13 h at 7 workers.
# Resumable: the queue is $GRID_STATE and a rerun picks up only what is still pending. Do not run two dispatchers
# against one state file (give a second stream its own GRID_STATE). One result per cell in $GRID_OUT
# (Diffusion Policy: $DP_WORK/results). run/alloc_resources.py assigns GPU, CPU set and threads per worker; its
# NUMA table (NODE_CPUS, GPU_NODE) describes the machine the paper ran on: adapt it to yours.
. "$(dirname "$0")/../config/env.sh"
set -eu
J="${1:-4}"; GPUS="${2:-0}"
cd "$MRMP_ROOT/run"
"$PY_SMD" grid.py plan --methods cd,direct,final_projection,mdoc,mmd_cbs,pcd_lb,pcd_shd,diffusion_policy
exec "$PY_SMD" -u grid.py run -j "$J" --gpus "$GPUS"
