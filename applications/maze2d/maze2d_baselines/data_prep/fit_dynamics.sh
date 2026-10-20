#!/usr/bin/env bash
# fit_dynamics.sh : least-squares linear dynamics of the point mass from the D4RL dataset (about 1 min, CPU).
# Every baseline imposes it. Output: runs/dynamics/linear_model.npz (expect R^2 = 0.9999, spectral radius 0.9990).
. "$(dirname "$0")/../config/env.sh"
cd "$DIRECT_DIR" && mb_log "fit dynamics" && \
  "$MB_PY" fit_dynamics.py --config configs/config.yaml --seed "$SEED" --log-folder "$MB_RUNS"
