#!/bin/bash
# check.sh : the guards. Run before trusting any table built from this machine's results.
. "$(dirname "$0")/../config/env.sh"
set -u
cd "$MRMP_ROOT/eval"
echo "== every scale table agrees =="            ; "$PY_SMD" check_scales.py          || true
echo "== every artifact built where evaluated ==" ; "$PY_SMD" check_artifact_scales.py || true
echo "== every cell ran at the cap it claims ==" ; "$PY_SMD" check_regimes.py         || true
echo "== the instance set passes the planners' own admission test =="
PYTHONPATH="$MRMP_ROOT/DiRecT/multi_robot/deps/torch_robotics" \
  "$PY_DIRECT" "$MRMP_ROOT/data_prep/validate_shared_instances.py" || true
