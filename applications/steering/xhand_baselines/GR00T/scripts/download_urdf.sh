#!/usr/bin/env bash
# Fetch the DexMate URDF (robot model used by the Genesis GT-vs-model renderer).
# Meshes are large and not committed to git (see .gitignore: assets/). Set URDF_SRC to a
# local checkout to symlink, or URDF_REPO to a git URL to clone.
#
#   URDF_SRC=/path/to/dexmate-urdf bash scripts/download_urdf.sh    # local
#   URDF_REPO=<git-url>            bash scripts/download_urdf.sh    # clone
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${DEST:-$HERE/assets/dexmate-urdf}"
mkdir -p "$(dirname "$DEST")"

if [ -n "${URDF_SRC:-}" ]; then
  echo "[urdf] linking $URDF_SRC -> $DEST"; ln -sfn "$URDF_SRC" "$DEST"
elif [ -n "${URDF_REPO:-}" ]; then
  echo "[urdf] cloning $URDF_REPO -> $DEST"; git clone --depth 1 "$URDF_REPO" "$DEST"
else
  echo "Set URDF_SRC=<local dexmate-urdf checkout> or URDF_REPO=<git url>."
  echo "Expected file after setup:"
  echo "  \$VEGA_URDF = $DEST/robots/humanoid/vega_1p/vega_1p_gripper.urdf"
  exit 1
fi
echo "[urdf] ready. Export before rendering:"
echo "  export VEGA_URDF=$DEST/robots/humanoid/vega_1p/vega_1p_gripper.urdf"
