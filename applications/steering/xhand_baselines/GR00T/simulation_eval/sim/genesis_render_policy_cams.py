#!/usr/bin/env python3
"""Stage 0 of the 'Genesis frame as input' pipeline.

Replays a GT joint trajectory in the SAME Genesis pick scene as genesis_replay_2way
(gray desk + white box, bimanual-squeeze box kinematics), but instead of rendering an
external view it mounts the FOUR policy cameras at their real URDF links and renders the
four observation streams the policy actually consumes:

    head_left_rgb  <- zed_left_camera  (fixed on head_l3)
    head_right_rgb <- zed_right_camera (fixed on head_l3)
    left_wrist_rgb <- L_camera_link    (fixed on L_arm_l7)
    right_wrist_rgb<- R_camera_link    (fixed on R_arm_l7)

Genesis merges fixed links, so each camera pose is derived per-frame as
    world = parent_link_world @ T_fixed_offset            (offsets extracted from the URDF)

Output is a LeRobot v2.0 mini-dataset (parquet with GT state/action + the 4 rendered mp4s
+ meta) so the UNCHANGED rollout_to_npz.py can run the policy on Genesis frames exactly as
it does on the real dataset -- the only variable that changes between column 2 and column 3
is the image source (real camera mp4 vs Genesis render).

    GT state comes from the rollout npz (key 'gt', [L,22]) so col2 and col3 share the
    identical state trajectory and length.

Env: VEGA_URDF, plus the same DESK_*/BOX_*/CAM geometry envs as genesis_replay_2way.
"""
import argparse
import json
import os as _os
from pathlib import Path

import numpy as np
import genesis as gs
import imageio.v2 as imageio

# reuse the calibrated scene + joint mapping from the 2-way renderer
from genesis_replay_2way import (URDF, DATA_JOINTS, DESK_SIZE, BOX_SIZE,
                                  GRASP_GAP, GRASP_NEAR, FPS, build_dof_map)

CAMS = ["head_left_rgb", "head_right_rgb", "left_wrist_rgb", "right_wrist_rgb"]
# (parent link, translation in parent frame, optical-Z forward in parent frame) from URDF
CAM_MOUNTS = {
    "head_left_rgb":  ("head_l3", [0.0365,  0.025, 0.0984], [1.0, 0.0, 0.0]),
    "head_right_rgb": ("head_l3", [0.0365, -0.025, 0.0984], [1.0, 0.0, 0.0]),
    "left_wrist_rgb": ("L_arm_l7", [0.2652, -0.0322, 0.0745], [0.996, 0.0, -0.087]),
    "right_wrist_rgb":("R_arm_l7", [0.2652,  0.0318, 0.0745], [0.996, 0.0, -0.087]),
}
HEAD_FOV = int(_os.environ.get("HEAD_FOV", "78"))
WRIST_FOV = int(_os.environ.get("WRIST_FOV", "95"))
CAM_RES = int(_os.environ.get("CAM_RES", "256"))


def _np(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def _quat_to_R(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="rollout npz with gt[L,22] (the trajectory to replay)")
    ap.add_argument("--out-dataset", required=True, help="dir to write the LeRobot mini-dataset")
    ap.add_argument("--task", required=True, help="task instruction to store in meta/episodes.jsonl")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=200)
    args = ap.parse_args()

    gt = np.load(args.npz)["gt"]
    L = len(gt)
    print(f"[gencam] replaying gt[{L},{gt.shape[1]}] from {args.npz}", flush=True)

    gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / FPS, substeps=2),
        vis_options=gs.options.VisOptions(background_color=(0.6, 0.6, 0.62),
                                          ambient_light=(0.5, 0.5, 0.5), shadow=True),
        show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    desk = scene.add_entity(gs.morphs.Box(pos=[0.62, 0.0, DESK_SIZE[2] / 2], size=DESK_SIZE.tolist()),
                            surface=gs.surfaces.Default(color=(0.62, 0.63, 0.66)))
    box = scene.add_entity(gs.morphs.Box(pos=[0.55, 0.0, DESK_SIZE[2] / 2 + 0.1], size=BOX_SIZE.tolist()),
                           surface=gs.surfaces.Default(color=(0.92, 0.91, 0.88)))
    robot = scene.add_entity(gs.morphs.URDF(file=URDF, fixed=True, pos=(0, 0, 0)))
    cams = {c: scene.add_camera(res=(CAM_RES, CAM_RES),
                                fov=(HEAD_FOV if c.startswith("head") else WRIST_FOV), GUI=False)
            for c in CAMS}
    scene.build()

    dof = build_dof_map(robot)
    n_dofs = robot.n_dofs

    def full_q(s22):
        q = np.zeros(n_dofs)
        for d_, gidx in enumerate(dof):
            q[gidx] = float(s22[d_])
        return q

    gl_l = robot.get_link("L_gripper_l1")
    gl_r = robot.get_link("R_gripper_l1")
    parents = {c: robot.get_link(CAM_MOUNTS[c][0]) for c in CAMS}

    # ---- calibrate the box from the GT squeeze, identical to the 2-way renderer ----
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
    cal_box_size = np.array([float(v) for v in _os.environ.get("BOX_DIMS", "0.178,0.229,0.102").split(",")])
    cal_desk_top = float(box_c[2] - cal_box_size[2] / 2)
    cal_box_start = np.array([float(box_c[0]), float(box_c[1]), cal_desk_top + cal_box_size[2] / 2])
    desk_pos = np.array([cal_box_start[0] + 0.10, 0.0, cal_desk_top - DESK_SIZE[2] / 2])
    print(f"[gencam] box_w={box_w:.2f} box@{np.round(cal_box_start,2).tolist()} desk_top={cal_desk_top:.2f}", flush=True)

    def set_cam_poses():
        for c in CAMS:
            pl_, tloc, fwd = CAM_MOUNTS[c]
            p = _np(parents[c].get_pos()).astype(float)
            R = _quat_to_R(_np(parents[c].get_quat()))
            eye = p + R @ np.array(tloc)
            look = eye + R @ np.array(fwd)
            up = R @ np.array([0.0, 0.0, 1.0]) if c.startswith("head") else np.array([0.0, 0.0, 1.0])
            cams[c].set_pose(pos=eye.tolist(), lookat=look.tolist(), up=up.tolist())

    # ---- replay + render the 4 policy views, with the same squeeze box kinematics ----
    robot.set_dofs_position(full_q(gt[0])); scene.step()
    desk.set_pos(desk_pos.tolist()); desk.set_quat([1, 0, 0, 0])
    step = max(1, L // args.max_frames)
    keep_idx = list(range(0, L, step))
    grasped = False; grip_off = None; bp = cal_box_start.copy()
    streams = {c: [] for c in CAMS}
    for t in keep_idx:
        robot.set_dofs_position(full_q(gt[t]))
        scene.step()
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
        set_cam_poses()
        for c in CAMS:
            rgb, *_ = cams[c].render(rgb=True)
            streams[c].append(_np(rgb).astype(np.uint8)[..., :3])
        if len(streams[CAMS[0]]) % 25 == 0:
            print(f"  rendered {len(streams[CAMS[0]])}/{len(keep_idx)} frames", flush=True)

    Lr = len(keep_idx)
    # ---- write LeRobot v2.0 mini-dataset: GT state/action + 4 rendered mp4s ----
    import pandas as pd
    ds = Path(args.out_dataset)
    (ds / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (ds / "meta").mkdir(parents=True, exist_ok=True)
    st = gt[keep_idx].astype(np.float64)
    df = pd.DataFrame({
        "observation.state": list(st),
        "action": list(st),
        "timestamp": (np.arange(Lr) / FPS).astype(np.float64),
        "annotation.human.action.task_description": np.zeros(Lr, np.int64),
        "task_index": np.zeros(Lr, np.int64),
        "episode_index": np.full(Lr, args.episode, np.int64),
        "index": np.arange(Lr, dtype=np.int64),
        "next.reward": np.zeros(Lr, np.float64),
        "next.done": np.r_[np.zeros(Lr - 1, bool), True],
    })
    df.to_parquet(ds / "data" / "chunk-000" / f"episode_{args.episode:06d}.parquet")
    for c in CAMS:
        vd = ds / "videos" / "chunk-000" / f"observation.images.{c}"
        vd.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(vd / f"episode_{args.episode:06d}.mp4", streams[c], fps=FPS, quality=8)
    feat = {
        "observation.state": {"dtype": "float32", "shape": [22], "names": None},
        "action": {"dtype": "float32", "shape": [22], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "next.reward": {"dtype": "float32", "shape": [1], "names": None},
        "annotation.human.action.task_description": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
    }
    for c in CAMS:
        feat[f"observation.images.{c}"] = {
            "dtype": "video", "shape": [CAM_RES, CAM_RES, 3], "names": ["height", "width", "channel"],
            "video_info": {"video.fps": float(FPS), "video.codec": "h264",
                           "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False}}
    info = {"codebase_version": "v2.0", "robot_type": "vega_genesis_rendered", "fps": FPS,
            "total_episodes": args.episode + 1, "total_frames": Lr, "chunks_size": 1000,
            "splits": {"train": f"0:{args.episode + 1}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": feat}
    json.dump(info, open(ds / "meta" / "info.json", "w"), indent=2)
    with open(ds / "meta" / "episodes.jsonl", "w") as f:
        f.write(json.dumps({"episode_index": args.episode, "tasks": [args.task], "length": Lr}) + "\n")
    json.dump({"task_index": 0, "task": args.task}, open(ds / "meta" / "tasks.jsonl", "w"))
    # frame indices into the ORIGINAL gt trajectory, so the 3-way merge can align the
    # (possibly subsampled) genesis stream 1:1 with the full-rate dataset rollout.
    np.save(ds / "meta" / "keep_idx.npy", np.asarray(keep_idx, dtype=np.int64))
    print(f"[gencam] DONE: {Lr} frames x 4 cams (keep_idx {keep_idx[0]}..{keep_idx[-1]} step {step}) -> {ds}", flush=True)


if __name__ == "__main__":
    main()
