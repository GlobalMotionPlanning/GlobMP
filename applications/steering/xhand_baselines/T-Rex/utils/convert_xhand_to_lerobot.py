"""Convert raw XHand1 teleop (DexMate Vega-1) -> T-Rex LeRobot v3.0 dataset.

Raw layout (one dir per episode, as staged from the teleop rig):
    <src>/episode_*/
        episode_*.h5     left/right_arm_current_pose [N,4,4], *_arm_target_pose [N,4,4],
                         *_hand_joint_positions [N,12], *_hand_target_joint_positions [N,12],
                         *_hand_tactile_force [N,5,3]
        episode_*_head_left_rgb.mp4, *_left_wrist.mp4, *_right_wrist.mp4 (+ head_right_rgb, unused)

Same action semantics as convert_inlab_to_lerobot.py (delta-base eef arm, absolute
hand targets, chunk 16, stride 1) at the XHand layout: 42-D = 2 x (9 eef + 12 hand),
tactile [10,3], no deform videos.  Images are stored at --image_size (the size the
trainer resizes to anyway).

Episode indices refer to the bench order: sorted episode dirs that have 4 mp4 + h5,
so `--episodes` accepts the same ids as bench/handover/{train,heldout}.json.

Usage:
  python utils/convert_xhand_to_lerobot.py --src RAW --output_root OUT \
      --episodes_json bench/handover/train.json --instruction "..."
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import cv2
import h5py
import numpy as np

from utils.lerobot_common import (
    ACTION_DIM, FRAME_STRIDE,
    KEY_HEAD, KEY_WRIST_R, KEY_WRIST_L, KEY_STATE, KEY_ACTION, KEY_ACTION_ABS, KEY_TACF6,
    pose_matrix_to_9d, build_action_chunk, build_trex_features, NormStatsAccumulator,
    NORM_STATS_FILENAME,
)


def bench_episodes(src):
    """Episode dirs in bench order (same filter as bench/handover/convert_xhand.py)."""
    eps = sorted(glob.glob(f"{src}/episode_*"))
    return [e for e in eps if len(glob.glob(e + "/*.mp4")) == 4 and glob.glob(e + "/*.h5")]


def read_video(path, size):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if size is not None:
            f = cv2.resize(f, size, interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def load_episode(ep_dir, size):
    name = os.path.basename(ep_dir)
    with h5py.File(os.path.join(ep_dir, f"{name}.h5"), "r") as f:
        s_l_pose = f["left_arm_current_pose"][:]
        s_r_pose = f["right_arm_current_pose"][:]
        a_l_pose = f["left_arm_target_pose"][:]
        a_r_pose = f["right_arm_target_pose"][:]
        s_l_hnd = f["left_hand_joint_positions"][:]
        s_r_hnd = f["right_hand_joint_positions"][:]
        a_l_hnd = f["left_hand_target_joint_positions"][:]
        a_r_hnd = f["right_hand_target_joint_positions"][:]
        t_l = f["left_hand_tactile_force"][:]      # [N,5,3]
        t_r = f["right_hand_tactile_force"][:]
    head = read_video(os.path.join(ep_dir, f"{name}_head_left_rgb.mp4"), size)
    wl = read_video(os.path.join(ep_dir, f"{name}_left_wrist.mp4"), size)
    wr = read_video(os.path.join(ep_dir, f"{name}_right_wrist.mp4"), size)

    states = np.concatenate([pose_matrix_to_9d(s_l_pose), s_l_hnd,
                             pose_matrix_to_9d(s_r_pose), s_r_hnd], axis=1)       # [N,42]
    abs_targets = np.concatenate([pose_matrix_to_9d(a_l_pose), a_l_hnd,
                                  pose_matrix_to_9d(a_r_pose), a_r_hnd], axis=1)  # [N,42]
    assert states.shape[1] == ACTION_DIM, states.shape
    n = min(len(states), len(head), len(wl), len(wr))
    return dict(n=n, n_h5=len(states), n_vid=(len(head), len(wl), len(wr)),
                head=head, wl=wl, wr=wr,
                s_l_pose=s_l_pose, s_r_pose=s_r_pose, a_l_pose=a_l_pose, a_r_pose=a_r_pose,
                a_l_hnd=a_l_hnd, a_r_hnd=a_r_hnd, states=states, abs_targets=abs_targets,
                tac=np.concatenate([t_l, t_r], axis=1).astype(np.float32))                # [N,10,3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--episodes_json", required=True, help="JSON list of bench episode ids.")
    ap.add_argument("--repo_id", default="")
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--image_size", type=int, nargs=2, default=[384, 288], metavar=("W", "H"))
    ap.add_argument("--stats_from", default="",
                    help="Copy trex_norm_stats.json from this dataset instead of computing it "
                         "(held-out sets must use the training stats).")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # lazy

    eps = bench_episodes(args.src)
    ids = json.load(open(args.episodes_json))
    print(f"{len(eps)} bench episodes in {args.src}; converting {len(ids)}: {ids[:8]}...", flush=True)

    W, H = args.image_size
    features = build_trex_features(head_shape=(3, H, W), include_wrist=True, wrist_shape=(3, H, W),
                                   include_tactile=True, deform_shape=None)
    ds = LeRobotDataset.create(repo_id=args.repo_id or os.path.basename(args.output_root.rstrip("/")),
                               fps=args.fps, features=features, root=args.output_root,
                               robot_type="vega1_xhand1", use_videos=True, vcodec="h264")

    acc = NormStatsAccumulator()
    index = []
    for k, bid in enumerate(ids):
        d = load_episode(eps[bid], (W, H))
        N = d["n"]
        for i in range(N):
            chunk = build_action_chunk(d["s_l_pose"], d["a_l_pose"], d["a_l_hnd"],
                                       d["s_r_pose"], d["a_r_pose"], d["a_r_hnd"], i, N)
            tgt = min(i + FRAME_STRIDE - 1, N - 1)
            ds.add_frame({
                "task": args.instruction,
                KEY_HEAD: d["head"][i], KEY_WRIST_R: d["wr"][i], KEY_WRIST_L: d["wl"][i],
                KEY_STATE: d["states"][i].astype(np.float32),
                KEY_ACTION: chunk,
                KEY_ACTION_ABS: d["abs_targets"][tgt].astype(np.float32),
                KEY_TACF6: d["tac"][i],
            })
            acc.add_frame(chunk, d["states"][i], d["tac"][i])
        ds.save_episode()
        acc.add_episode_tracking(d["states"][:N], d["abs_targets"][:N])
        index.append({"lerobot_episode": k, "bench_episode": bid,
                      "source": os.path.basename(eps[bid]), "frames": N,
                      "h5_frames": d["n_h5"], "video_frames": list(d["n_vid"])})
        print(f"  [{k + 1}/{len(ids)}] bench ep {bid} {os.path.basename(eps[bid])}: {N} frames "
              f"(h5 {d['n_h5']}, video {d['n_vid']})", flush=True)

    ds.finalize()
    if args.stats_from:
        dst = os.path.join(args.output_root, NORM_STATS_FILENAME)
        shutil.copy(os.path.join(args.stats_from, NORM_STATS_FILENAME), dst)
        print(f">>> stats copied from {args.stats_from}")
    else:
        print(f">>> stats: {acc.write(args.output_root)}")
    with open(os.path.join(args.output_root, "meta", "bench_episodes.json"), "w") as f:
        json.dump(index, f, indent=1)
    print(f">>> done: {args.output_root}")


if __name__ == "__main__":
    main()
