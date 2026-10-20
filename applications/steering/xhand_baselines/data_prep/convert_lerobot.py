#!/usr/bin/env python3
"""Today's XHand teleop (raw .h5 + 4 mp4 per episode) -> LeRobot v3.0, the schema the bench trains on.

THIS IS A DIFFERENT EMBODIMENT FROM THE T-REX BASELINES, and that is the whole point of the run:

    T-Rex  : 58-D state = left_arm 7 | left_hand 22 | right_arm 7 | right_hand 22   (Sharpa Wave)
    XHand  : 38-D state = left_arm 7 | left_hand 12 | right_arm 7 | right_hand 12
    tactile: T-Rex 60-D (10 fingertips x 6 F/T)  vs  XHand 30-D (5 fingers x 3 per hand)

So every baseline's state/action projection has to be rebuilt; this is transfer to a new hand,
not a continuation. Layout is kept arm|hand|arm|hand so the arm block still lines up with the
pretrained one -- only the hand slices are new.

state  : *_arm_joint_positions        + *_hand_joint_positions
action : *_arm_target_dofs            + *_hand_target_joint_positions   (commanded, as T-Rex does)
tactile: *_hand_tactile_force (5,3) flattened per hand -> 30

Videos are copied, not re-encoded: h5 and all four streams already agree frame-for-frame at 30 fps.
"""
import glob, json, os, shutil, sys
import h5py, numpy as np, pyarrow as pa, pyarrow.parquet as pq

SRC = os.environ["SRC"]          # raw session: episode_*/ with .h5, 4 .mp4, result.json
OUT = os.environ["OUT"]
TASK = os.environ["TASK"]
CAMS = ["head_left_rgb", "head_right_rgb", "left_wrist", "right_wrist"]
FPS = 30
SL = {"left_arm": (0, 7), "left_hand": (7, 19), "right_arm": (19, 26), "right_hand": (26, 38)}


def load(h5):
    with h5py.File(h5, "r") as h:
        n = h["timestamp"].shape[0]
        st = np.concatenate([h["left_arm_joint_positions"][:], h["left_hand_joint_positions"][:],
                             h["right_arm_joint_positions"][:], h["right_hand_joint_positions"][:]], 1)
        ac = np.concatenate([h["left_arm_target_dofs"][:], h["left_hand_target_joint_positions"][:],
                             h["right_arm_target_dofs"][:], h["right_hand_target_joint_positions"][:]], 1)
        tf = np.concatenate([h["left_hand_tactile_force"][:].reshape(n, -1),
                             h["right_hand_tactile_force"][:].reshape(n, -1)], 1)
    return st.astype(np.float32), ac.astype(np.float32), tf.astype(np.float32), n


def main():
    eps = sorted(glob.glob(f"{SRC}/episode_*"))
    good = [e for e in eps if len(glob.glob(e + "/*.mp4")) == 4 and glob.glob(e + "/*.h5")]
    print(f"{len(eps)} episodes, {len(good)} with all 4 cameras + h5", flush=True)
    for d in ["data/chunk-000", "meta/episodes/chunk-000"] + [f"videos/observation.images.{c}/chunk-000" for c in CAMS]:
        os.makedirs(f"{OUT}/{d}", exist_ok=True)

    rows, recs, gidx, kept = [], [], 0, 0
    for e in good:
        h5 = glob.glob(e + "/*.h5")[0]
        st, ac, tf, n = load(h5)
        vids = {c: [m for m in glob.glob(e + "/*.mp4") if m.endswith(f"_{c}.mp4")] for c in CAMS}
        if any(not v for v in vids.values()):
            print(f"  skip {os.path.basename(e)}: camera name mismatch"); continue
        for c in CAMS:
            shutil.copy2(vids[c][0], f"{OUT}/videos/observation.images.{c}/chunk-000/file-{kept:03d}.mp4")
        rows.append(pa.table({
            "observation.state": pa.array(list(st), pa.list_(pa.float32(), 38)),
            "action": pa.array(list(ac), pa.list_(pa.float32(), 38)),
            "observation.tactile_force": pa.array(list(tf), pa.list_(pa.float32(), 30)),
            "timestamp": pa.array(np.arange(n, dtype=np.float32) / FPS),
            "frame_index": pa.array(np.arange(n), pa.int64()),
            "episode_index": pa.array([kept] * n, pa.int64()),
            "index": pa.array(np.arange(gidx, gidx + n), pa.int64()),
            "task_index": pa.array([0] * n, pa.int64())}))
        succ = None
        rj = os.path.join(e, "result.json")
        if os.path.exists(rj):
            try: succ = bool(json.load(open(rj)).get("success"))
            except Exception: pass
        recs.append({"episode_index": kept, "tasks": [TASK], "length": int(n),
                     # lerobot resolves a global frame index to its episode through these
                     # absolute bounds; without them _get_query_indices raises KeyError.
                     "dataset_from_index": gidx, "dataset_to_index": gidx + int(n),
                     "source_episode": os.path.basename(e), "success": succ,
                     "data/chunk_index": 0, "data/file_index": kept,
                     **{f"videos/observation.images.{c}/chunk_index": 0 for c in CAMS},
                     **{f"videos/observation.images.{c}/file_index": kept for c in CAMS},
                     # one file per episode, so each episode spans its whole video
                     **{f"videos/observation.images.{c}/from_timestamp": 0.0 for c in CAMS},
                     **{f"videos/observation.images.{c}/to_timestamp": float(n) / FPS for c in CAMS}})
        gidx += n; kept += 1
        if kept % 25 == 0: print(f"  {kept}/{len(good)}", flush=True)

    for i, t in enumerate(rows):
        pq.write_table(t, f"{OUT}/data/chunk-000/file-{i:03d}.parquet")
    pq.write_table(pa.Table.from_pylist(recs), f"{OUT}/meta/episodes/chunk-000/file-000.parquet")
    # indexed BY the task string, exactly as the T-Rex release is -- lerobot resolves tasks by
    # string lookup, and an unindexed table makes a tensor reach pi05's tokenizer
    import pandas as pd
    pd.DataFrame({"task": [TASK], "task_index": [0]}).set_index("task")[["task_index"]] \
      .to_parquet(f"{OUT}/meta/tasks.parquet")
    feats = {"observation.state": {"dtype": "float32", "shape": [38]},
             "action": {"dtype": "float32", "shape": [38]},
             "observation.tactile_force": {"dtype": "float32", "shape": [30]},
             "timestamp": {"dtype": "float32", "shape": [1]}, "frame_index": {"dtype": "int64", "shape": [1]},
             "episode_index": {"dtype": "int64", "shape": [1]}, "index": {"dtype": "int64", "shape": [1]},
             "task_index": {"dtype": "int64", "shape": [1]}}
    for c in CAMS:
        feats[f"observation.images.{c}"] = {"dtype": "video", "shape": [360, 640, 3],
            "names": ["height", "width", "channel"],
            "info": {"video.fps": FPS, "video.height": 360, "video.width": 640, "video.channels": 3,
                     "video.codec": "h264", "video.pix_fmt": "yuv420p", "has_audio": False}}
    json.dump({"codebase_version": "v3.0", "robot_type": "vega1_xhand", "fps": FPS,
               "total_episodes": kept, "total_frames": gidx, "total_tasks": 1,
               "total_videos": kept * len(CAMS), "chunks_size": 1000,
               "splits": {"train": f"0:{kept}"},
               "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
               "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
               "features": feats}, open(f"{OUT}/meta/info.json", "w"), indent=1)
    print(f"done: {kept} episodes, {gidx} frames ({gidx/FPS/3600:.2f} h) -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
