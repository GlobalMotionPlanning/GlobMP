#!/bin/bash
# install_editable.sh : install this package's forks into the three envs in editable mode, as the paper's envs had them
# (after creating the envs from config/envs/*.yml). The forks import each other by package name (mmd, torch_robotics,
# mp_baselines, experiment_launcher, mdoc, smd), so an env still pointing at another checkout runs that checkout's code.
. "$(dirname "$0")/../config/env.sh"
set -eu
R=$MRMP_ROOT
inst(){ py=$1; shift; for p in "$@"; do "$py" -m pip install -q --no-deps --no-build-isolation -e "$R/$p"; echo "  $(basename $(dirname $(dirname $py))): $p"; done; }
# direct: DiRecT, CD, Final Proj., MMD-CBS, PCD, Diffusion Policy (torch_robotics from MDOC, as in the paper's env)
inst "$PY_DIRECT" DiRecT/multi_robot DiRecT/multi_robot/deps/experiment_launcher DiRecT/multi_robot/deps/motion_planning_baselines \
                  MDOC MDOC/deps/torch_robotics
inst "$PY_MDOC"   MDOC MDOC/deps/experiment_launcher MDOC/deps/motion_planning_baselines MDOC/deps/torch_robotics
inst "$PY_SMD"    SMD SMD/deps/experiment_launcher SMD/deps/motion_planning_baselines SMD/deps/torch_robotics
