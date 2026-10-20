#!/bin/bash
# Stage 0: conda, if the machine has none.  Skipped if conda is already on PATH or at $CONDA_BASE.  Installs
# Miniforge (conda-forge only, the channel of config/envs/*.yml, so no Anaconda terms-of-service prompt) to
# $HOME/miniconda3, which is what config/env.sh expects by default.
set -eu
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
if [ -x "$CONDA_BASE/bin/conda" ]; then echo "conda already at $CONDA_BASE"; exit 0; fi
if command -v conda >/dev/null 2>&1; then echo "conda already on PATH: $(command -v conda)"; exit 0; fi
I=/tmp/miniforge-installer.sh
echo "downloading miniforge for $(uname -m) ..."
curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh" -o "$I"
bash "$I" -b -p "$CONDA_BASE"
rm -f "$I"
"$CONDA_BASE/bin/conda" --version
echo "installed at $CONDA_BASE.  Now:  CONDA_BASE=$CONDA_BASE bash tools/create_envs.sh"
