#!/usr/bin/env python3
"""2-panel GT | <model> comparison for the Vega PICK task.

Scene mirrors the real setup: a freestanding gray desk in front of the robot with a
white box on it. The box is driven kinematically by a bimanual squeeze-grasp rule:
when both gripper links close around the box footprint it attaches to their midpoint
and follows the lift. Panels are captioned with live box lift in cm and time-synced
by lift progress so GT and model lift in lockstep.

Geometry is env-overridable per episode: DESK_TOP_Z, DESK_X, BOX_X, BOX_Y, CAM_*.
"""
import argparse
from pathlib import Path
import numpy as np
import genesis as gs
import imageio.v2 as imageio
import cv2
import os as _os

URDF = _os.environ.get("VEGA_URDF") or _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "dexmate-urdf/robots/humanoid/vega_1p/vega_1p_gripper.urdf")
_LEFT_ARM = [f"L_arm_j{i}" for i in range(1, 8)]
_RIGHT_ARM = [f"R_arm_j{i}" for i in range(1, 8)]
_TORSO = ["torso_j1", "torso_j2", "torso_j3"]
_HEAD = ["head_j1", "head_j2", "head_j3"]
DATA_JOINTS = _LEFT_ARM + _RIGHT_ARM + _TORSO + _HEAD + ["L_gripper_j1", "R_gripper_j1"]

# scene geometry (matches the real desk/box; override per-episode via env)
DESK_TOP_Z = float(_os.environ.get("DESK_TOP_Z", 0.72))
DESK_X = float(_os.environ.get("DESK_X", 0.62))               # desk center ahead of robot
DESK_SIZE = np.array([0.60, 1.60, DESK_TOP_Z])                # solid block from the floor
BOX_SIZE = np.array([0.22, 0.30, 0.20])                       # white pick box (x depth, y width, z height)
BOX_START = np.array([float(_os.environ.get("BOX_X", 0.55)),
                      float(_os.environ.get("BOX_Y", 0.0)),
                      DESK_TOP_Z + BOX_SIZE[2] / 2])
GRASP_GAP = float(_os.environ.get("GRASP_GAP", 0.14))         # grippers this close beyond box width == squeeze
GRASP_NEAR = float(_os.environ.get("GRASP_NEAR", 0.24))       # midpoint must be near the box to grasp
FPS = 30


def build_dof_map(robot):
    j2dof = {}
    for j in robot.joints:
        idl = getattr(j, "dofs_idx_local", None)
        if idl is not None and len(idl) > 0:
            j2dof[j.name] = int(idl[0])
    return [j2dof[n] for n in DATA_JOINTS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, nargs=2, default=[560, 460])
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--npz", required=True, help="npz with gt[L,22] + pred[L,22]")
    ap.add_argument("--label", required=True, help="label for the model panel")
    ap.add_argument("--color", type=int, nargs=3, default=[90, 90, 255])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(args.npz)
    gt = d["gt"]; pred = d["pred"]
    L = min(len(gt), len(pred))
    trajs = [("GT", (90, 220, 90), gt),
             (args.label, tuple(args.color), pred)]
    W, H = args.res

    gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / FPS, substeps=2),
        vis_options=gs.options.VisOptions(background_color=(0.6, 0.6, 0.62),
                                          ambient_light=(0.5, 0.5, 0.5), shadow=True),
        show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    desk = scene.add_entity(gs.morphs.Box(pos=[DESK_X, 0.0, DESK_SIZE[2] / 2], size=DESK_SIZE.tolist()),
                            surface=gs.surfaces.Default(color=(0.62, 0.63, 0.66)))   # gray desk (pinned per-frame)
    box = scene.add_entity(gs.morphs.Box(pos=BOX_START.tolist(), size=BOX_SIZE.tolist()),
                           surface=gs.surfaces.Default(color=(0.92, 0.91, 0.88)))    # white box
    robot = scene.add_entity(gs.morphs.URDF(file=URDF, fixed=True, pos=(0, 0, 0)))
    cam = scene.add_camera(res=(W, H),
                           pos=(float(_os.environ.get("CAM_X", 1.85)),
                                float(_os.environ.get("CAM_Y", -1.15)),
                                float(_os.environ.get("CAM_Z", 1.90))),
                           lookat=(float(_os.environ.get("LOOK_X", 0.55)),
                                   float(_os.environ.get("LOOK_Y", 0.0)),
                                   float(_os.environ.get("LOOK_Z", 0.85))),
                           fov=int(_os.environ.get("CAM_FOV", 55)), GUI=False)
    scene.build()
    dof = build_dof_map(robot)
    n_dofs = robot.n_dofs

    def full_q(s22):
        q = np.zeros(n_dofs)
        for d_, gidx in enumerate(dof):
            q[gidx] = float(s22[d_])
        return q

    def _np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    gl_l = robot.get_link("L_gripper_l1")
    gl_r = robot.get_link("R_gripper_l1")
    head_cam_link = robot.get_link("head_l3")   # ZED is fixed-mounted here (genesis merges fixed links)
    VIEW = _os.environ.get("VIEW", "back")      # back | headcam | ext

    def _quat_to_R(q):
        w, x, y, z = [float(v) for v in q]
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

    CAM_OFF = np.array([float(v) for v in _os.environ.get("HEADCAM_OFF", "0.06,0,0.06").split(",")])
    CAM_FWD = np.array([float(v) for v in _os.environ.get("HEADCAM_FWD", "1,0,-0.1").split(",")])

    def set_headcam():
        p = _np(head_cam_link.get_pos()).astype(float)
        R = _quat_to_R(_np(head_cam_link.get_quat()))
        eye = p + R @ CAM_OFF
        cam.set_pose(pos=eye.tolist(), lookat=(eye + R @ CAM_FWD).tolist())

    def set_backcam(look):
        # slightly behind-left of the robot, above shoulder height
        eye = np.array([float(_os.environ.get("BACK_X", -1.10)),
                        float(_os.environ.get("BACK_Y", -1.55)),
                        float(_os.environ.get("BACK_Z", 2.05))])
        cam.set_pose(pos=eye.tolist(), lookat=look.tolist())

    # ---- per-episode auto-calibration from the GT trajectory (deep check) ----
    # the bimanual squeeze reveals the true box: min gripper gap = box width,
    # grasp midpoint = box center, desk top = box bottom.
    park = np.array([5.0, 5.0, 0.1])
    box.set_pos(park.tolist()); desk.set_pos((park + [0, 2, 0]).tolist())
    gaps, mids = [], []
    for t in range(0, L, 2):
        robot.set_dofs_position(full_q(gt[t])); scene.step()
        pl, pr = _np(gl_l.get_pos()), _np(gl_r.get_pos())
        gaps.append(float(np.linalg.norm(pl[:2] - pr[:2])))
        mids.append((pl + pr) / 2.0)
    gaps = np.asarray(gaps); mids = np.asarray(mids)
    box_w = float(np.clip(np.percentile(gaps, 3), 0.16, 0.45))
    g_idx = int(np.argmax(gaps < box_w + 0.02)) if (gaps < box_w + 0.02).any() else int(np.argmin(gaps))
    box_c = mids[g_idx]
    # real box: 7 x 9 x 4 inches (depth x width x height) unless overridden
    cal_box_size = np.array([float(v) for v in _os.environ.get("BOX_DIMS", "0.178,0.229,0.102").split(",")])
    if abs(box_w - cal_box_size[1]) > 0.06:
        print(f"[calib] note: measured squeeze gap {box_w:.2f}m vs box width {cal_box_size[1]:.2f}m", flush=True)
    cal_desk_top = float(box_c[2] - cal_box_size[2] / 2)
    cal_box_start = np.array([float(box_c[0]), float(box_c[1]), cal_desk_top + cal_box_size[2] / 2])
    desk_pos = np.array([cal_box_start[0] + 0.10, 0.0, cal_desk_top - DESK_SIZE[2] / 2])
    print(f"[calib] box_w={box_w:.2f}m  box@({cal_box_start[0]:.2f},{cal_box_start[1]:.2f},{cal_box_start[2]:.2f})  desk_top={cal_desk_top:.2f}m  grasp_frame~{g_idx*2}", flush=True)

    def render_traj(traj):
        # Box driven KINEMATICALLY by a bimanual squeeze rule: static on the desk until
        # both gripper links close around its footprint, then it rides their midpoint.
        robot.set_dofs_position(full_q(traj[0])); scene.step()
        frames, dzs = [], []
        desk.set_pos(desk_pos.tolist()); desk.set_quat([1, 0, 0, 0])
        step = max(1, L // args.max_frames)
        grasped = False
        grip_off = None
        bp = cal_box_start.copy()
        for t in range(0, L, step):
            robot.set_dofs_position(full_q(traj[t]))
            scene.step()
            pl, pr = _np(gl_l.get_pos()), _np(gl_r.get_pos())
            mid = (pl + pr) / 2.0
            gap = float(np.linalg.norm(pl[:2] - pr[:2]))
            if not grasped \
                    and gap < max(box_w, cal_box_size[1]) + GRASP_GAP \
                    and np.linalg.norm(mid[:2] - bp[:2]) < GRASP_NEAR \
                    and abs(mid[2] - bp[2]) < 0.25:
                grasped = True
                grip_off = bp - mid
            if grasped:
                bp = mid + grip_off
                bp[2] = max(bp[2], cal_desk_top + cal_box_size[2] / 2 - 0.005)  # never below desk top
            try:
                box.set_pos(bp.tolist()); box.set_quat([1, 0, 0, 0])
                desk.set_pos(desk_pos.tolist()); desk.set_quat([1, 0, 0, 0])
            except Exception:
                pass
            dzs.append(float(bp[2] - cal_box_start[2]))
            if VIEW == "headcam":
                set_headcam()
            elif VIEW == "back":
                set_backcam(np.array([cal_box_start[0], cal_box_start[1], cal_box_start[2] + 0.10]))
            rgb, *_ = cam.render(rgb=True)
            frames.append(_np(rgb).astype(np.uint8))
        return frames, dzs, grasped

    panels = []
    for name, color, tr in trajs:
        print(f"[2way] rendering {name} ...", flush=True)
        fr, dzs, grasped = render_traj(tr)
        print(f"[2way]   {name}: grasped={grasped}  final box lift {dzs[-1]*100:+.1f}cm  peak {max(dzs)*100:+.1f}cm", flush=True)
        panels.append((name, color, fr, dzs))

    def label(img, name, cm, color):
        out = img.copy(); h, w = img.shape[:2]
        cv2.rectangle(out, (0, 0), (w, 32), (0, 0, 0), -1)
        cv2.putText(out, name, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
        cv2.rectangle(out, (0, h - 30), (w, h), (0, 0, 0), -1)
        cv2.putText(out, f"box lifted: {cm:+.0f} cm", (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 2, cv2.LINE_AA)
        return out

    # Time-synchronize panels by LIFT progress: GT sets the reference timeline; the model
    # panel shows the frame whose lift is closest to GT's at each output frame.
    def lift_end(dzs):
        m = max((abs(v) for v in dzs), default=0.0)
        if m < 0.03:
            return len(dzs)
        e = next((i for i, v in enumerate(dzs) if abs(v) >= 0.97 * m), len(dzs))
        return min(len(dzs), e + 2)
    N = int(args.fps * 7)
    gt_idx = np.linspace(0, max(1, lift_end(panels[0][3]) - 1), N).astype(int)
    gt_lift = [panels[0][3][i] for i in gt_idx]
    def lift_start(dzs):
        v = np.asarray(dzs)
        nz = np.flatnonzero(v > 0.01)
        return int(nz[0]) if len(nz) else len(v)
    g_start = lift_start(panels[0][3])
    norm = []
    for k, (name, color, fr, dzs) in enumerate(panels):
        if k == 0:
            idx = gt_idx
        else:
            dv = np.asarray(dzs); pmax = float(dv.max()) if len(dv) else 0.0
            m_start = lift_start(dzs)
            idx = []
            for j, gi in enumerate(gt_idx):
                if gi < g_start or g_start == 0:
                    # pre-lift: proportional time so the panel keeps moving
                    idx.append(int(round(gi / max(1, g_start) * m_start)))
                else:
                    # lift phase: match by lift progress, never before the model's own grasp
                    p = gt_lift[j]
                    cand = int(np.argmin(np.abs(dv[m_start:] - min(p, pmax)))) + m_start if m_start < len(dv) else len(dv) - 1
                    idx.append(cand)
            idx = np.asarray(idx)
        idx = np.clip(idx, 0, len(fr) - 1)
        norm.append((name, color, [fr[i] for i in idx], [dzs[i] for i in idx]))
    out_frames = [np.concatenate([label(fr[i], name, dzs[i] * 100.0, color)
                                  for name, color, fr, dzs in norm], axis=1) for i in range(N)]
    print(f"[2way] time-synced to {N} frames/panel", flush=True)

    gif = Path(args.out + ".gif"); mp4 = Path(args.out + ".mp4")
    imageio.mimsave(gif, out_frames[::2], fps=max(1, args.fps // 2), loop=0)
    try:
        imageio.mimsave(mp4, out_frames, fps=args.fps, quality=8)
    except Exception as e:
        print(f"[2way] mp4 skipped ({e})")
    print(f"[done] {mp4} ({len(out_frames)} frames)")


if __name__ == "__main__":
    main()
