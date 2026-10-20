#!/usr/bin/env bash
# run_all.sh [gpu] : every baseline on both settings (about 1.5 h at 3 repeats; SafeDiffuser-ReS costs ~487 s per plan).
# A failed run is logged and the sweep continues.
. "$(dirname "$0")/../config/env.sh"
GPU=${1:-0}; FAILED=0
for S in $SETTINGS; do
  while IFS=$'\t' read -r P NAME; do
    [[ $P == \#* ]] && continue
    bash "$MB_CODE/run/run_baseline.sh" "$P" "$S" "$GPU" && mb_log "[OK] $S/$P" || { mb_log "[FAIL] $S/$P"; FAILED=$((FAILED+1)); }
  done < "$MB_CODE/config/policies.tsv"
done
mb_log "sweep done, failures: $FAILED"
