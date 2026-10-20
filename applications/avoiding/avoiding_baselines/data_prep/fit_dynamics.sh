#!/usr/bin/env bash
# fit_dynamics.sh : linear dynamics from the 96 expert episodes (seconds, CPU).
# Every constrained baseline imposes it.
#
# fit_dynamics.py writes to logs/<env>/dynamics relative to its own directory and
# takes no output flag, so the result is copied to where eval.py looks for it:
# <log-folder>/<env>/dynamics/linear_model.npz.
. "$(dirname "$0")/../config/env.sh"
cd "$DIRECT_DIR" && ab_log "fit dynamics" && "$AB_PY" fit_dynamics.py --env avoiding-v0 || exit 1
mkdir -p "$AB_RUNS/avoiding-v0/dynamics"
cp "$DIRECT_DIR/logs/avoiding-v0/dynamics/linear_model.npz" "$AB_RUNS/avoiding-v0/dynamics/"
ab_log "dynamics -> $AB_RUNS/avoiding-v0/dynamics/linear_model.npz"
