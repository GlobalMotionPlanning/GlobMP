#!/usr/bin/env bash
# finetune_pi05.sh <task> <gpu|gpu,gpu,...> [--tactile]   (recipe: config/recipe.env; outputs: runs/<task>/pi05[_tactile])
[ $# -ge 2 ] || { sed -n 2p "$0"; exit 1; }
exec bash "$(dirname "$0")/run_baseline.sh" pi05 "$@"
