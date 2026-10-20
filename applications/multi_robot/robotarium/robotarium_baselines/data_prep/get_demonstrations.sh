#!/bin/bash
# get_demonstrations.sh : unpack the paper's single-robot demonstrations (data_prep/demonstrations/*.tar.zst, 744 MB,
# about 4 GB unpacked) into SMD/data_trajectories/, where tools/wire_artifacts.sh looks for them. The planners rebuild
# their normalizer from these at start-up, and the shipped models were trained with exactly this data: use them to
# reproduce the paper's cells (data_prep/generate_data.sh draws a new, unseeded set; see the README).
. "$(dirname "$0")/../config/env.sh"
set -eu
S=$RB_CODE/data_prep/demonstrations; T=$MRMP_ROOT/SMD/data_trajectories
Z=$(command -v zstd || true); [ -n "$Z" ] || Z=$(dirname "$PY_SMD")/zstd       # the smd env ships zstd
[ -x "$Z" ] || { echo "zstd not found (apt install zstd, or create the smd env first)"; exit 1; }
(cd "$S" && sha256sum -c --quiet SHA256SUMS) || { echo "checksum mismatch in $S (incomplete git lfs pull?)"; exit 1; }
mkdir -p "$T"
for a in "$S"/*.tar.zst; do
  n=$(basename "$a" .tar.zst)
  [ -e "$T/$n" ] && { echo "  $T/$n exists (an earlier unpack or generate_data.sh run): move it away first"; exit 1; }
  "$Z" -dc "$a" | tar -xf - -C "$T"
  rb_log "unpacked $n ($(ls "$T/$n" | wc -l) contexts)"
done
