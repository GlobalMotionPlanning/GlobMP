#!/usr/bin/env bash
# finetune_tvla.sh <task> <gpu|gpu,gpu,...>   Tactile-VLA (recipe: config/recipe.env pi0.5 + TVLA_HISTORY; outputs: runs/<task>/tactile_vla)
[ $# -ge 2 ] || { sed -n 2p "$0"; exit 1; }
exec bash "$(dirname "$0")/run_baseline.sh" tvla "$@"
