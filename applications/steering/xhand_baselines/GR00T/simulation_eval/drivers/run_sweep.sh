#!/usr/bin/env bash
# CLOSED-LOOP box-position sweep: the policy's own actions drive the robot in Genesis.
# Starts the policy server once (model loaded once) and reuses it across scenarios, then
# stitches a green=pick / red=miss gallery.
#
#   ENGINE=crom CKPT=<dir> MODEL_VENV=<venv> BACKBONE=<cosmos> GENESIS_VENV=<.venv-genesis> \
#   START_NPZ=<rollout npz with gt[L,22]> VEGA_URDF=<urdf> bash drivers/run_sweep.sh
set -u
SB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GENESIS_VENV="${GENESIS_VENV:?set GENESIS_VENV to the .venv-genesis python venv}"
START_NPZ="${START_NPZ:?set START_NPZ to a rollout npz containing gt[L,22] (the start pose)}"
export VEGA_URDF="${VEGA_URDF:?set VEGA_URDF to vega_1p_gripper.urdf}"
TASK="${TASK:-pick: grasp the white box and lift it clear of the cart.}"
OUTDIR="${OUTDIR:-$SB/out/cl_sweep}"
SOCK="${SOCK:-/tmp/simbench_policy.sock}"
MAXCH="${MAXCH:-90}"
mkdir -p "$OUTDIR"

# scenarios: label|box_dx|box_dy  (metres, offset from the nominal trained box pose)
SCEN=("near|-0.08|0.0" "nominal|0.0|0.0" "far|0.12|0.0" "farther|0.22|0.0" "shifted-left|0.0|0.15")

# start the server if not already up
if [ ! -f "$SOCK.ready" ]; then
  echo "[sweep] starting policy server..."
  SOCK="$SOCK" bash "$SB/drivers/run_server.sh" > "$OUTDIR/server.log" 2>&1 &
  for i in $(seq 1 70); do [ -f "$SOCK.ready" ] && break; sleep 5; done
fi
[ -f "$SOCK.ready" ] || { echo "[sweep] server never became ready — see $OUTDIR/server.log"; exit 1; }
echo "[sweep] server ready"

JSON="["
for s in "${SCEN[@]}"; do
  IFS='|' read -r label dx dy <<< "$s"
  echo "[sweep] === $label (dx=$dx dy=$dy) ==="
  PYOPENGL_PLATFORM=egl "$GENESIS_VENV/bin/python" "$SB/sim/closedloop_genesis.py" \
    --sock "$SOCK" --start-npz "$START_NPZ" --task "$TASK" \
    --box-dx "$dx" --box-dy "$dy" --max-chunks "$MAXCH" \
    --out "$OUTDIR/$label" > "$OUTDIR/client_$label.log" 2>&1
  grep -E "\[cl\] DONE" "$OUTDIR/client_$label.log" || echo "  (see client_$label.log)"
  JSON="$JSON{\"npz\":\"$OUTDIR/$label.npz\",\"label\":\"$label\"},"
done
JSON="${JSON%,}]"

# shut the server down
"$GENESIS_VENV/bin/python" - "$SOCK" <<'PY'
import socket,struct,pickle,sys
s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); s.connect(sys.argv[1])
d=pickle.dumps({"cmd":"shutdown"}); s.sendall(struct.pack(">I",len(d))+d); s.close()
PY

echo "[sweep] stitching gallery..."
"$GENESIS_VENV/bin/python" "$SB/sim/stitch_gallery.py" --scenarios "$JSON" --out "$OUTDIR/box_sweep"
echo "[sweep] DONE -> $OUTDIR/box_sweep.mp4"
