#!/usr/bin/env python3
"""2-panel GT | <model> replay for the POINT task.

Unlike the box-push replay (genesis_replay_2way.py) there is no manipulated object:
the robot stands at the table and points with the right arm. Object positions were
never recorded, so the scene shows the real table geometry only; correctness of the
pointed-at object is judged on the real robot / camera footage. Each panel is
captioned with the instruction given to the model.
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

# real point scene: large metal table in front of the stationary robot
TABLE_TOP_Z = float(_os.environ.get("TABLE_TOP_Z", 0.72))
TABLE_CENTER = np.array([float(_os.environ.get("TABLE_X", 0.95)), 0.0, TABLE_TOP_Z - 0.02])
TABLE_SIZE = np.array([0.90, 1.70, 0.04])
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
    ap.add_argument("--task", required=True, help="instruction shown as the bottom caption")
    ap.add_argument("--color", type=int, nargs=3, default=[90, 90, 255])
    ap.add_argument("--real-video", default=None,
                    help="episode camera mp4 — shown as a leading REAL panel (objects visible)")
    ap.add_argument("--objects", default=None,
                    help="json {name: [x, y]} from estimate_point_objects.py — rendered as "
                         "colored blocks/marker at ESTIMATED positions")
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
    scene.add_entity(gs.morphs.Box(pos=TABLE_CENTER.tolist(), size=TABLE_SIZE.tolist(), fixed=True),
                     surface=gs.surfaces.Default(color=(0.62, 0.64, 0.66)))   # metal table
    if args.objects:
        import json as _json
        OBJ_COLOR = {"red_block": (0.85, 0.10, 0.10), "green_block": (0.10, 0.55, 0.15),
                     "blue_block": (0.10, 0.15, 0.45), "yellow_block": (0.90, 0.80, 0.10),
                     "red_marker": (0.90, 0.35, 0.40)}
        # true shapes measured from the 260708 footage: red/green stand upright,
        # blue bar + yellow brick lie flat, marker is a whiteboard pen lying flat
        OBJ_SIZE = {"red_block": (0.04, 0.04, 0.07), "green_block": (0.04, 0.04, 0.05),
                    "blue_block": (0.12, 0.05, 0.03), "yellow_block": (0.08, 0.045, 0.03)}
        for name, spec in _json.load(open(args.objects)).items():
            ox, oy = (spec["pos"] if isinstance(spec, dict) else spec)
            yaw = spec.get("yaw", 0.0) if isinstance(spec, dict) else 0.0
            if name == "red_marker":
                if isinstance(spec, dict) and spec.get("standing"):
                    marker_pose = dict(pos=[ox, oy, TABLE_TOP_Z + 0.06], euler=(0, 0, 0))
                else:
                    marker_pose = dict(pos=[ox, oy, TABLE_TOP_Z + 0.010], euler=(0, 90, yaw))
                scene.add_entity(gs.morphs.Cylinder(radius=0.010, height=0.12, fixed=True,
                                                    **marker_pose),
                                 surface=gs.surfaces.Default(color=OBJ_COLOR[name]))
            else:
                sx, sy, sz = OBJ_SIZE.get(name, (0.05, 0.05, 0.05))
                scene.add_entity(gs.morphs.Box(pos=[ox, oy, TABLE_TOP_Z + sz / 2],
                                               size=(sx, sy, sz), fixed=True,
                                               euler=(0, 0, yaw)),
                                 surface=gs.surfaces.Default(color=OBJ_COLOR.get(name, (0.5, 0.5, 0.5))))
    robot = scene.add_entity(gs.morphs.URDF(file=URDF, fixed=True, pos=(0, 0, 0)))
    # over-the-shoulder 3/4 view from the robot's left-rear: shows the robot AND where over
    # the table the right arm points.
    cam = scene.add_camera(res=(W, H),
                           pos=(float(_os.environ.get("CAM_X", -0.9)),
                                float(_os.environ.get("CAM_Y", 1.35)),
                                float(_os.environ.get("CAM_Z", 1.95))),
                           lookat=(float(_os.environ.get("LOOK_X", 0.95)),
                                   float(_os.environ.get("LOOK_Y", -0.15)),
                                   float(_os.environ.get("LOOK_Z", 0.75))),
                           fov=int(_os.environ.get("CAM_FOV", 52)), GUI=False)
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

    def render_traj(traj):
        robot.set_dofs_position(full_q(traj[0])); scene.step()
        frames = []
        step = max(1, L // args.max_frames)
        for t in range(0, L, step):
            robot.set_dofs_position(full_q(traj[t]))
            scene.step()
            rgb, *_ = cam.render(rgb=True)
            frames.append(_np(rgb).astype(np.uint8))
        return frames

    panels = []
    for name, color, tr in trajs:
        print(f"[point] rendering {name} ...", flush=True)
        panels.append((name, color, render_traj(tr)))

    def label(img, name, color):
        out = img.copy(); h, w = img.shape[:2]
        cv2.rectangle(out, (0, 0), (w, 32), (0, 0, 0), -1)
        cv2.putText(out, name, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
        if args.objects and name != "REAL (head cam)":
            cv2.putText(out, "objects: estimated", (w - 205, 23), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.rectangle(out, (0, h - 30), (w, h), (0, 0, 0), -1)
        cv2.putText(out, f'"{args.task}"', (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 2, cv2.LINE_AA)
        return out

    n = min(len(fr) for _, _, fr in panels)
    if args.real_video:
        real = imageio.mimread(args.real_video, memtest=False)
        idx = np.linspace(0, len(real) - 1, n).astype(int)
        rh, rw = panels[0][2][0].shape[:2]
        real_fr = [cv2.resize(np.asarray(real[i])[..., :3], (rw, rh)) for i in idx]
        panels.insert(0, ("REAL (head cam)", (240, 240, 90), real_fr))
    out_frames = [np.concatenate([label(fr[i], name, color)
                                  for name, color, fr in panels], axis=1) for i in range(n)]

    gif = Path(args.out + ".gif"); mp4 = Path(args.out + ".mp4")
    imageio.mimsave(gif, out_frames[::2], fps=max(1, args.fps // 2), loop=0)
    try:
        imageio.mimsave(mp4, out_frames, fps=args.fps, quality=8)
    except Exception as e:
        print(f"[point] mp4 skipped ({e})")
    print(f"[done] {mp4} ({len(out_frames)} frames)")


if __name__ == "__main__":
    main()
