#!/usr/bin/env bash
# finetune_groot.sh <task> <gpu|gpu,gpu,...> [--tactile]   (recipe: config/recipe.env; outputs: runs/<task>/groot[_tactile])
[ $# -ge 2 ] || { sed -n 2p "$0"; exit 1; }
exec bash "$(dirname "$0")/run_baseline.sh" groot "$@"
