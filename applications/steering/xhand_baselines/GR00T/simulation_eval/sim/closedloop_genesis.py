#!/usr/bin/env python3
"""Closed-loop-in-sim rollout: the POLICY's own actions drive the robot in Genesis.

Unlike the open-loop replay (which follows a fixed GT trajectory), here the robot starts from
a fixed pose, the box sits at a scenario position, and each control step:
    read joint state  ->  render the 4 policy cameras  ->  (socket) policy  ->  action chunk
    ->  apply the chunk as absolute joint targets, stepping physics + the squeeze-grasp box
So if the box is placed too far, the policy genuinely reaches short and MISSES -- the failure
is the model's actual behavior, not a script. Scene is deliberately simplified (illustrative).

Talks to policy_server_eval.py over a Unix socket (the model lives in an incompatible venv).
Saves an external-view frame stream + box-lift series + success flag per scenario.
"""
import argparse
import io
import os as _os
import pickle
import socket
import struct
import time

import numpy as np
import genesis as gs


def arr2b(a):
    """Portable .npy-bytes encoding (works across numpy 1.x / 2.x venvs)."""
    buf = io.BytesIO(); np.save(buf, np.ascontiguousarray(a)); return buf.getvalue()


def b2arr(b):
    return np.load(io.BytesIO(b))

from genesis_replay_2way import (URDF, DESK_SIZE, GRASP_GAP, GRASP_NEAR, FPS, build_dof_map)
from genesis_render_policy_cams import CAM_MOUNTS, CAMS, _quat_to_R, HEAD_FOV, WRIST_FOV, CAM_RES

BOX_DIMS = np.array([float(v) for v in _os.environ.get("BOX_DIMS", "0.178,0.229,0.102").split(",")])
# nominal box + desk from the ep459 calibration (the pose the policy was trained around)
NOM_BOX = np.array([0.47, 0.12, 0.99])
DESK_TOP = 0.93


def _np(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


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
    ap.add_argument("--sock", required=True)
    ap.add_argument("--start-npz", required=True, help="npz with gt[L,22]; gt[0] is the start pose")
    ap.add_argument("--task", required=True)
    ap.add_argument("--box-dx", type=float, default=0.0, help="box offset from nominal, metres (+x=farther)")
    ap.add_argument("--box-dy", type=float, default=0.0, help="box offset from nominal, metres (+y=left)")
    ap.add_argument("--exec-horizon", type=int, default=8)
    ap.add_argument("--max-chunks", type=int, default=90)
    ap.add_argument("--success-lift", type=float, default=0.08, help="metres of box lift = success")
    ap.add_argument("--panel-res", type=int, nargs=2, default=[520, 440])
    ap.add_argument("--out", required=True, help="output npz prefix")
    args = ap.parse_args()

    start_pose = np.load(args.start_npz)["gt"][0].astype(np.float32)
    box0 = NOM_BOX + np.array([args.box_dx, args.box_dy, 0.0])
    print(f"[cl] scenario box@{np.round(box0,3).tolist()} (dx={args.box_dx} dy={args.box_dy})", flush=True)

    gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / FPS, substeps=2),
        vis_options=gs.options.VisOptions(background_color=(0.6, 0.6, 0.62),
                                          ambient_light=(0.5, 0.5, 0.5), shadow=True),
        show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    desk_pos = np.array([box0[0] + 0.10, 0.0, DESK_TOP - DESK_SIZE[2] / 2])
    desk = scene.add_entity(gs.morphs.Box(pos=desk_pos.tolist(), size=DESK_SIZE.tolist()),
                            surface=gs.surfaces.Default(color=(0.62, 0.63, 0.66)))
    box = scene.add_entity(gs.morphs.Box(pos=box0.tolist(), size=BOX_DIMS.tolist()),
                           surface=gs.surfaces.Default(color=(0.92, 0.91, 0.88)))
    robot = scene.add_entity(gs.morphs.URDF(file=URDF, fixed=True, pos=(0, 0, 0)))
    polcams = {c: scene.add_camera(res=(CAM_RES, CAM_RES),
                                   fov=(HEAD_FOV if c.startswith("head") else WRIST_FOV), GUI=False)
               for c in CAMS}
    W, H = args.panel_res
    extcam = scene.add_camera(res=(W, H), pos=(-1.10, -1.55, 2.05), lookat=(box0[0], box0[1], DESK_TOP + 0.10), fov=55, GUI=False)
    scene.build()

    dof = build_dof_map(robot); n_dofs = robot.n_dofs
    inv_state_from_q = dof  # state22[i] <- q[dof[i]]

    def full_q(s22):
        q = np.zeros(n_dofs)
        for d_, gidx in enumerate(dof):
            q[gidx] = float(s22[d_])
        return q

    def read_state22():
        q = _np(robot.get_dofs_position())
        return np.array([q[gidx] for gidx in inv_state_from_q], dtype=np.float32)

    gl_l = robot.get_link("L_gripper_l1"); gl_r = robot.get_link("R_gripper_l1")
    parents = {c: robot.get_link(CAM_MOUNTS[c][0]) for c in CAMS}

    def set_pol_cam_poses():
        for c in CAMS:
            _, tloc, fwd = CAM_MOUNTS[c]
            p = _np(parents[c].get_pos()).astype(float)
            R = _quat_to_R(_np(parents[c].get_quat()))
            eye = p + R @ np.array(tloc)
            up = R @ np.array([0.0, 0.0, 1.0]) if c.startswith("head") else np.array([0.0, 0.0, 1.0])
            polcams[c].set_pose(pos=eye.tolist(), lookat=(eye + R @ np.array(fwd)).tolist(), up=up.tolist())

    def render_pol():
        set_pol_cam_poses()
        return {c: _np(polcams[c].render(rgb=True)[0]).astype(np.uint8)[..., :3] for c in CAMS}

    # ---- connect to the policy server ----
    while not _os.path.exists(args.sock + ".ready"):
        time.sleep(0.5)
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    for _ in range(60):
        try:
            conn.connect(args.sock); break
        except OSError:
            time.sleep(0.5)
    print("[cl] connected to policy server", flush=True)

    # ---- init robot + box ----
    robot.set_dofs_position(full_q(start_pose)); scene.step()
    box_pos = box0.copy(); grasped = False; grip_off = None
    ext_frames = []; lifts = []
    box_bottom_floor = DESK_TOP + BOX_DIMS[2] / 2 - 0.005

    def update_box_and_capture():
        nonlocal grasped, grip_off, box_pos
        pl, pr = _np(gl_l.get_pos()), _np(gl_r.get_pos())
        mid = (pl + pr) / 2.0
        gap = float(np.linalg.norm(pl[:2] - pr[:2]))
        if not grasped and gap < BOX_DIMS[1] + GRASP_GAP \
                and np.linalg.norm(mid[:2] - box_pos[:2]) < GRASP_NEAR and abs(mid[2] - box_pos[2]) < 0.25:
            grasped = True; grip_off = box_pos - mid
        if grasped:
            box_pos = mid + grip_off
            box_pos[2] = max(box_pos[2], box_bottom_floor)
        box.set_pos(box_pos.tolist()); box.set_quat([1, 0, 0, 0])
        desk.set_pos(desk_pos.tolist()); desk.set_quat([1, 0, 0, 0])
        extcam.set_pose(pos=(-1.10, -1.55, 2.05), lookat=(box0[0], box0[1], DESK_TOP + 0.10))
        ext_frames.append(_np(extcam.render(rgb=True)[0]).astype(np.uint8))
        lifts.append(float(box_pos[2] - box0[2]))

    update_box_and_capture()  # frame 0

    # ---- closed loop ----
    K = args.exec_horizon
    success = False; stop_reason = "max_chunks"
    chunks_log = []; emit_log = []; t_exec = 0   # ActProbe features (ACM/TCE) come from these
    for c in range(args.max_chunks):
        state22 = read_state22()
        imgs = render_pol()
        send_msg(conn, {"state": arr2b(state22),
                        "images": {c: arr2b(imgs[c]) for c in CAMS}, "task": args.task})
        rep = recv_msg(conn)
        if rep is None:
            stop_reason = "server_dropped"; break
        chunk = b2arr(rep["action"])                # [H,22] absolute joint targets
        emit_log.append(t_exec); chunks_log.append(np.asarray(chunk, np.float32))
        nexec = min(K, len(chunk))
        for i in range(nexec):
            robot.set_dofs_position(full_q(chunk[i])); scene.step()
            update_box_and_capture()
        t_exec += nexec
        peak = max(lifts)
        if peak >= args.success_lift and lifts[-1] >= args.success_lift * 0.6:
            success = True; stop_reason = "grasp_lift"
            # a few extra frames to show the lift holding
            for _ in range(6):
                robot.set_dofs_position(full_q(chunk[min(K, len(chunk)) - 1])); scene.step()
                update_box_and_capture()
            break
        if c % 10 == 0:
            print(f"  chunk {c}/{args.max_chunks} peak_lift={peak*100:.1f}cm grasped={grasped}", flush=True)

    try:
        send_msg(conn, {"cmd": "quit"}); conn.close()
    except OSError:
        pass

    peak = max(lifts) if lifts else 0.0
    print(f"[cl] DONE success={success} reason={stop_reason} peak_lift={peak*100:.1f}cm "
          f"final={lifts[-1]*100:.1f}cm grasped={grasped} frames={len(ext_frames)}", flush=True)
    _os.makedirs(_os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, ext=np.asarray(ext_frames, dtype=np.uint8), lifts=np.asarray(lifts, np.float32),
             success=success, peak_lift=peak, final_lift=lifts[-1] if lifts else 0.0,
             grasped=grasped, box_dx=args.box_dx, box_dy=args.box_dy, reason=stop_reason,
             chunks=np.asarray(chunks_log, np.float32) if chunks_log else np.zeros((0, 1, 22), np.float32),
             chunk_emit_steps=np.asarray(emit_log, np.int64),
             chunk_task=args.task, exec_horizon=K)
    print(f"[cl] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
