#!/usr/bin/env python3
"""Blocking request/response policy server for closed-loop-in-sim evaluation.

Loads a CROM-3B (or GR00T) checkpoint ONCE and answers action queries over a Unix socket,
so a Genesis simulator running in its own (incompatible) venv can drive the robot in a tight
observe->act loop:  sim renders 4 cameras + reads joint state -> sends here -> gets an action
chunk back -> applies it -> repeats.

Protocol (length-prefixed pickle):
    request : {"state": float32[22], "images": {cam: uint8[H,W,3]}, "task": str}
              {"cmd": "quit"}
    reply   : {"action": float32[H,22]}

The engine package is imported but NOT modified. A '<sock>.ready' file is touched after the
model is loaded so the sim client can wait for readiness before connecting.
"""
import argparse
import importlib.util
import io
import os
import pickle
import socket
import struct

import numpy as np


def arr2b(a):
    """Portable .npy-bytes encoding (works across numpy 1.x / 2.x venvs)."""
    buf = io.BytesIO(); np.save(buf, np.ascontiguousarray(a)); return buf.getvalue()


def b2arr(b):
    return np.load(io.BytesIO(b))

STATE_SLICES = [("left_arm", 0, 7), ("right_arm", 7, 14), ("torso", 14, 17),
                ("head", 17, 20), ("left_hand", 20, 21), ("right_hand", 21, 22)]
GROUPS = [s[0] for s in STATE_SLICES]
CAMS = ["head_left_rgb", "head_right_rgb", "left_wrist_rgb", "right_wrist_rgb"]
LANG_KEY = "annotation.human.action.task_description"


def recv_msg(conn):
    hdr = b""
    while len(hdr) < 4:
        b = conn.recv(4 - len(hdr))
        if not b:
            return None
        hdr += b
    n = struct.unpack(">I", hdr)[0]
    buf = bytearray()
    while len(buf) < n:
        b = conn.recv(min(1 << 20, n - len(buf)))
        if not b:
            return None
        buf += b
    return pickle.loads(buf)


def send_msg(conn, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack(">I", len(data)) + data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--modality-config", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sock", required=True, help="unix socket path")
    ap.add_argument("--engine", choices=["crom", "gr00t"], default="crom")
    args = ap.parse_args()

    # register EmbodimentTag.NEW_EMBODIMENT via the modality config, exactly like the rollout
    spec = importlib.util.spec_from_file_location("vega_pick_modality", args.modality_config)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

    if args.engine == "crom":
        from crom_3b_engine.data.embodiment_tags import EmbodimentTag
        from crom_3b_engine.policy.crom_3b_engine_policy import CROM3BEnginePolicy as Policy
    else:
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy as Policy
    print(f"[server] loading {args.engine} from {args.ckpt} ...", flush=True)
    policy = Policy(embodiment_tag=EmbodimentTag.NEW_EMBODIMENT, model_path=args.ckpt, device=args.device)
    print("[server] model ready", flush=True)

    def predict(state22, images, task):
        obs = {
            "video": {cam: images[cam][None, None] for cam in CAMS},                # (1,1,H,W,3)
            "state": {g: state22[a:b].astype(np.float32)[None, None] for g, a, b in STATE_SLICES},
            "language": {LANG_KEY: [[task]]},
        }
        action, _ = policy.get_action(obs)
        H = np.asarray(action[GROUPS[0]]).shape[1]
        return np.stack([np.concatenate([np.asarray(action[g])[0, i].reshape(-1) for g in GROUPS])[:22]
                         for i in range(H)]).astype(np.float32)

    if os.path.exists(args.sock):
        os.remove(args.sock)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.sock); srv.listen(1)
    open(args.sock + ".ready", "w").close()
    print(f"[server] listening on {args.sock}", flush=True)
    n = 0; shutdown = False
    try:
        # persistent: serve one client (= one scenario) at a time, keep listening for the
        # next so a whole sweep reuses the single loaded model. 'shutdown' ends the server.
        while not shutdown:
            conn, _ = srv.accept()
            print("[server] client connected", flush=True)
            while True:
                req = recv_msg(conn)
                if req is None or req.get("cmd") == "quit":
                    print("[server] client done", flush=True); break
                if req.get("cmd") == "shutdown":
                    shutdown = True; print("[server] shutdown requested", flush=True); break
                state = b2arr(req["state"])
                images = {k: b2arr(v) for k, v in req["images"].items()}
                act = predict(state, images, req["task"])
                send_msg(conn, {"action": arr2b(act)})
                n += 1
                if n % 20 == 0:
                    print(f"[server] served {n} queries", flush=True)
            conn.close()
    finally:
        srv.close()
        for p in (args.sock, args.sock + ".ready"):
            if os.path.exists(p):
                os.remove(p)
    print(f"[server] shut down after {n} queries", flush=True)


if __name__ == "__main__":
    main()
