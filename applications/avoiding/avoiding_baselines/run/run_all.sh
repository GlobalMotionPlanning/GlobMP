#!/usr/bin/env bash
# run_all.sh [gpu] : every baseline of Table 3, 100 episodes each.
# A failed run is logged and the sweep continues.
. "$(dirname "$0")/../config/env.sh"
GPU=${1:-0}; FAILED=0
while IFS=$'\t' read -r P NAME EXPNAME; do
  [[ $P == \#* || -z $P ]] && continue
  bash "$AB_CODE/run/run_baseline.sh" "$P" "$GPU" && ab_log "[OK] $P" || { ab_log "[FAIL] $P"; FAILED=$((FAILED+1)); }
done < "$AB_CODE/config/policies.tsv"
ab_log "sweep done, failures: $FAILED"
