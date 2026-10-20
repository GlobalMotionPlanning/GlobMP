#!/bin/bash
# create_envs.sh [env ...] : the conda envs of config/envs/ (default: direct mdoc smd) as the paper's machine had them:
# the conda packages from <env>.yml (conda-forge), then the exact pip packages from <env>-pip.txt with --no-deps.
# Afterwards: tools/install_editable.sh (the forks). The Robotarium package needs only direct and smd, and no forks.
set -eu
cd "$(dirname "$0")/.."
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
for e in ${*:-direct mdoc smd}; do
  [ -d "$CONDA_BASE/envs/$e" ] && { echo "$CONDA_BASE/envs/$e exists: remove it or skip $e"; exit 1; }
  "$CONDA_BASE/bin/conda" env create -q -n $e -f config/envs/$e.yml
  PYTHONNOUSERSITE=1 "$CONDA_BASE/envs/$e/bin/python" -m pip install -q --no-deps -r config/envs/$e-pip.txt
  PYTHONNOUSERSITE=1 "$CONDA_BASE/envs/$e/bin/python" -c "import torch; print('$e: torch', torch.__version__, 'cuda', torch.cuda.is_available())"
done
