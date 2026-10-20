#!/usr/bin/env python3
"""N-panel Genesis replay: GT | model@dataset-frames | model@genesis-frames | ...

Generalizes genesis_replay_2way to an arbitrary list of joint trajectories rendered side by
side in the SAME calibrated pick scene (gray desk + white box, bimanual-squeeze kinematics,
per-episode auto-calibration from the GT squeeze). Panels are time-synced by lift progress.

Reads ONE npz that carries gt[L,22] plus one or more prediction arrays named pred, pred2,
pred3, ... (or explicit --keys). Labels/colors are given per prediction; GT is always panel 0.
"""
import argparse
from pathlib import Path
import numpy as np
import genesis as gs
import imageio.v2 as imageio
import cv2
import os as _os

from genesis_replay_2way import (URDF, DATA_JOINTS, DESK_SIZE, BOX_SIZE, BOX_START,
                                  GRASP_GAP, GRASP_NEAR, DESK_X, FPS, build_dof_map)


def _np(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, nargs=2, default=[520, 440])
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--npz", required=True, help="npz with gt[L,22] + pred/pred2/... arrays")
    ap.add_argument("--keys", nargs="+", default=None,
                    help="prediction array keys in the npz (default: auto pred,pred2,...)")
    ap.add_argument("--labels", nargs="+", required=True, help="one label per prediction key")
    ap.add_argument("--colors", nargs="+", required=True,
                    help="one 'R,G,B' per prediction key")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(args.npz)
    gt = d["gt"]
    keys = args.keys or [k for k in ["pred", "pred2", "pred3", "pred4"] if k in d.files]
    assert len(keys) == len(args.labels) == len(args.colors), \
        f"keys({len(keys)})/labels/colors length mismatch"
    L = min([len(gt)] + [len(d[k]) for k in keys])
    trajs = [("GT", (90, 220, 90), gt[:L])]
    for k, lab, col in zip(keys, args.labels, args.colors):
        c = tuple(int(v) for v in col.split(","))
        trajs.append((lab, c, d[k][:L]))
    W, H = args.res

    gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / FPS, substeps=2),
        vis_options=gs.options.VisOptions(background_color=(0.6, 0.6, 0.62),
                                          ambient_light=(0.5, 0.5, 0.5), shadow=True),
        show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    desk = scene.add_entity(gs.morphs.Box(pos=[DESK_X, 0.0, DESK_SIZE[2] / 2], size=DESK_SIZE.tolist()),
                            surface=gs.surfaces.Default(color=(0.62, 0.63, 0.66)))
    box = scene.add_entity(gs.morphs.Box(pos=BOX_START.tolist(), size=BOX_SIZE.tolist()),
                           surface=gs.surfaces.Default(color=(0.92, 0.91, 0.88)))
    robot = scene.add_entity(gs.morphs.URDF(file=URDF, fixed=True, pos=(0, 0, 0)))
    cam = scene.add_camera(res=(W, H),
                           pos=(1.85, -1.15, 1.90), lookat=(0.55, 0.0, 0.85),
                           fov=55, GUI=False)
    scene.build()
    dof = build_dof_map(robot); n_dofs = robot.n_dofs

    def full_q(s22):
        q = np.zeros(n_dofs)
        for d_, gidx in enumerate(dof):
            q[gidx] = float(s22[d_])
        return q

    gl_l = robot.get_link("L_gripper_l1"); gl_r = robot.get_link("R_gripper_l1")

    # ---- per-episode calibration from GT squeeze (same as 2-way) ----
    park = np.array([5.0, 5.0, 0.1])
    box.set_pos(park.tolist()); desk.set_pos((park + [0, 2, 0]).tolist())
    gaps, mids = [], []
    for t in range(0, L, 2):
        robot.set_dofs_position(full_q(gt[t])); scene.step()
        pl, pr = _np(gl_l.get_pos()), _np(gl_r.get_pos())
        gaps.append(float(np.linalg.norm(pl[:2] - pr[:2]))); mids.append((pl + pr) / 2.0)
    gaps = np.asarray(gaps); mids = np.asarray(mids)
    box_w = float(np.clip(np.percentile(gaps, 3), 0.16, 0.45))
    g_idx = int(np.argmax(gaps < box_w + 0.02)) if (gaps < box_w + 0.02).any() else int(np.argmin(gaps))
    box_c = mids[g_idx]
    cal_box_size = np.array([float(v) for v in _os.environ.get("BOX_DIMS", "0.178,0.229,0.102").split(",")])
    cal_desk_top = float(box_c[2] - cal_box_size[2] / 2)
    cal_box_start = np.array([float(box_c[0]), float(box_c[1]), cal_desk_top + cal_box_size[2] / 2])
    desk_pos = np.array([cal_box_start[0] + 0.10, 0.0, cal_desk_top - DESK_SIZE[2] / 2])

    def set_backcam(look):
        eye = np.array([-1.10, -1.55, 2.05])
        cam.set_pose(pos=eye.tolist(), lookat=look.tolist())

    def render_traj2(traj):
        robot.set_dofs_position(full_q(traj[0])); scene.step()
        frames, dzs = [], []
        step = max(1, L // args.max_frames)
        grasped = False; grip_off = None; bp = cal_box_start.copy()
        for t in range(0, L, step):
            robot.set_dofs_position(full_q(traj[t])); scene.step()
            pl, pr = _np(gl_l.get_pos()), _np(gl_r.get_pos())
            mid = (pl + pr) / 2.0
            gap = float(np.linalg.norm(pl[:2] - pr[:2]))
            if not grasped and gap < max(box_w, cal_box_size[1]) + GRASP_GAP \
                    and np.linalg.norm(mid[:2] - bp[:2]) < GRASP_NEAR and abs(mid[2] - bp[2]) < 0.25:
                grasped = True; grip_off = bp - mid
            if grasped:
                bp = mid + grip_off
                bp[2] = max(bp[2], cal_desk_top + cal_box_size[2] / 2 - 0.005)
            box.set_pos(bp.tolist()); box.set_quat([1, 0, 0, 0])
            desk.set_pos(desk_pos.tolist()); desk.set_quat([1, 0, 0, 0])
            set_backcam(np.array([cal_box_start[0], cal_box_start[1], cal_box_start[2] + 0.10]))
            rgb, *_ = cam.render(rgb=True)
            frames.append(_np(rgb).astype(np.uint8))
            dzs.append(float(bp[2] - cal_box_start[2]))
        return frames, dzs, grasped

    panels = []
    for name, color, tr in trajs:
        print(f"[nway] rendering {name} ...", flush=True)
        fr, dzs, grasped = render_traj2(tr)
        print(f"[nway]   {name}: grasped={grasped} final lift {dzs[-1]*100:+.1f}cm peak {max(dzs)*100:+.1f}cm", flush=True)
        panels.append((name, color, fr, dzs))

    def label(img, name, cm, color):
        out = img.copy(); h, w = img.shape[:2]
        cv2.rectangle(out, (0, 0), (w, 32), (0, 0, 0), -1)
        cv2.putText(out, name, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        cv2.rectangle(out, (0, h - 30), (w, h), (0, 0, 0), -1)
        cv2.putText(out, f"box lifted: {cm:+.0f} cm", (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 2, cv2.LINE_AA)
        return out

    def lift_end(dzs):
        m = max((abs(v) for v in dzs), default=0.0)
        if m < 0.03:
            return len(dzs)
        e = next((i for i, v in enumerate(dzs) if abs(v) >= 0.97 * m), len(dzs))
        return min(len(dzs), e + 2)

    def lift_start(dzs):
        v = np.asarray(dzs); nz = np.flatnonzero(v > 0.01)
        return int(nz[0]) if len(nz) else len(v)

    N = int(args.fps * 7)
    gt_idx = np.linspace(0, max(1, lift_end(panels[0][3]) - 1), N).astype(int)
    gt_lift = [panels[0][3][i] for i in gt_idx]
    g_start = lift_start(panels[0][3])
    norm = []
    for k, (name, color, fr, dzs) in enumerate(panels):
        if k == 0:
            idx = gt_idx
        else:
            dv = np.asarray(dzs); pmax = float(dv.max()) if len(dv) else 0.0
            m_start = lift_start(dzs); idx = []
            for j, gi in enumerate(gt_idx):
                if gi < g_start or g_start == 0:
                    idx.append(int(round(gi / max(1, g_start) * m_start)))
                else:
                    p = gt_lift[j]
                    cand = int(np.argmin(np.abs(dv[m_start:] - min(p, pmax)))) + m_start if m_start < len(dv) else len(dv) - 1
                    idx.append(cand)
            idx = np.asarray(idx)
        idx = np.clip(idx, 0, len(fr) - 1)
        norm.append((name, color, [fr[i] for i in idx], [dzs[i] for i in idx]))
    out_frames = [np.concatenate([label(fr[i], name, dzs[i] * 100.0, color)
                                  for name, color, fr, dzs in norm], axis=1) for i in range(N)]
    print(f"[nway] time-synced to {N} frames/panel across {len(norm)} panels", flush=True)

    gif = Path(args.out + ".gif"); mp4 = Path(args.out + ".mp4")
    imageio.mimsave(gif, out_frames[::2], fps=max(1, args.fps // 2), loop=0)
    try:
        imageio.mimsave(mp4, out_frames, fps=args.fps, quality=8)
    except Exception as e:
        print(f"[nway] mp4 skipped ({e})")
    print(f"[done] {mp4} ({len(out_frames)} frames, {len(norm)} panels)")


if __name__ == "__main__":
    main()
