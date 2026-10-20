#!/usr/bin/env bash
# Fetch a prepared DexMate Vega PICK dataset (LeRobot v2.0). Alternative to building it
# from raw teleop with scripts/prepare_data.py. Set DATA_URL to a share link, or DATA_SRC
# to a local path to copy/symlink from.
#
#   DATA_URL=<https-or-s3-link-to-tar> bash scripts/download_data.sh
#   DATA_SRC=/path/to/pick             bash scripts/download_data.sh   # local copy
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${DEST:-<data>/data/pick}"
mkdir -p "$(dirname "$DEST")"

if [ -n "${DATA_SRC:-}" ]; then
  echo "[data] linking $DATA_SRC -> $DEST"
  ln -sfn "$DATA_SRC" "$DEST"
elif [ -n "${DATA_URL:-}" ]; then
  echo "[data] downloading $DATA_URL"
  tmp="$(mktemp -d)"; curl -L "$DATA_URL" -o "$tmp/data.tar"
  mkdir -p "$DEST"; tar -xf "$tmp/data.tar" -C "$DEST" --strip-components=1; rm -rf "$tmp"
else
  echo "Set DATA_URL=<link> or DATA_SRC=<local path>. Expected a LeRobot v2.0 dataset with"
  echo "  meta/{info,episodes,tasks,modality,stats}.json  data/chunk-000/*.parquet  videos/..."
  exit 1
fi

echo "[data] ready at $DEST"
echo "  DATASET=$DEST bash train/finetune_pick.sh"
