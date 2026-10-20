#!/usr/bin/env python3
"""Second conversion of today's XHand teleop: GR00T-flavoured LeRobot v2.

GR00T reads a different dialect from the lerobot trainers (ACT/DP/pi05) that the v3 dataset
feeds: v2.0 layout, one parquet + one mp4 per episode under episode_XXXXXX names, plus a
meta/modality.json that slices the flat state/action vectors into named groups. So today's data
needs converting twice -- same frames, two dialects.

Groups follow the XHand layout, NOT T-Rex's: left_hand/right_hand are 12-DoF here, not 22.
Tactile is carried as its own state group so a --tactile variant can include or drop it without
another conversion.
"""
import glob, json, os, shutil
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

SRC = os.environ["SRC"]          # the LeRobot v3 dataset from convert_lerobot.py
OUT = os.environ["OUT"]
CAMS = ["head_left_rgb", "head_right_rgb", "left_wrist", "right_wrist"]
FPS = 30
SL = [("left_arm", 0, 7), ("left_hand", 7, 19), ("right_arm", 19, 26), ("right_hand", 26, 38)]

info = json.load(open(f"{SRC}/meta/info.json"))
meta = pq.read_table(f"{SRC}/meta/episodes/chunk-000/file-000.parquet").to_pylist()
meta.sort(key=lambda r: r["episode_index"])
TASK = meta[0]["tasks"][0]
for d in ["data/chunk-000", "meta"] + [f"videos/chunk-000/observation.images.{c}" for c in CAMS]:
    os.makedirs(f"{OUT}/{d}", exist_ok=True)

gidx, recs = 0, []
for r in meta:
    i = r["episode_index"]
    t = pq.read_table(f"{SRC}/data/chunk-000/file-{i:03d}.parquet")
    n = t.num_rows
    st = np.asarray(t["observation.state"].to_pylist(), np.float32)
    ac = np.asarray(t["action"].to_pylist(), np.float32)
    tac = np.asarray(t["observation.tactile_force"].to_pylist(), np.float32)
    pq.write_table(pa.table({
        "observation.state": pa.array(list(np.concatenate([st, tac], 1)), pa.list_(pa.float32(), 68)),
        "action": pa.array(list(ac), pa.list_(pa.float32(), 38)),
        "timestamp": pa.array(np.arange(n, dtype=np.float32) / FPS),
        "annotation.human.action.task_description": pa.array([0] * n, pa.int64()),
        "task_index": pa.array([0] * n, pa.int64()),
        "episode_index": pa.array([i] * n, pa.int64()),
        "index": pa.array(np.arange(gidx, gidx + n), pa.int64()),
        "next.reward": pa.array(np.zeros(n, np.float32)),
        "next.done": pa.array([False] * (n - 1) + [True])}),
        f"{OUT}/data/chunk-000/episode_{i:06d}.parquet")
    for c in CAMS:   # same bytes as the v3 video: hard link (one copy on disk), copy across filesystems
        s, d = (f"{SRC}/videos/observation.images.{c}/chunk-000/file-{i:03d}.mp4",
                f"{OUT}/videos/chunk-000/observation.images.{c}/episode_{i:06d}.mp4")
        try: os.link(s, d)
        except OSError: shutil.copy2(s, d)
    recs.append({"episode_index": i, "tasks": [TASK], "length": int(n)})
    gidx += n
    if (i + 1) % 25 == 0: print(f"  {i+1}/{len(meta)}", flush=True)

with open(f"{OUT}/meta/episodes.jsonl", "w") as f:
    for r in recs: f.write(json.dumps(r) + "\n")
with open(f"{OUT}/meta/tasks.jsonl", "w") as f:
    f.write(json.dumps({"task_index": 0, "task": TASK}) + "\n")
state = {k: {"start": a, "end": b} for k, a, b in SL}
state["tactile"] = {"start": 38, "end": 68}      # tactile rides in state; drop it for the -tac run
json.dump({"state": state,
           "action": {k: {"start": a, "end": b} for k, a, b in SL},
           "video": {c: {"original_key": f"observation.images.{c}"} for c in CAMS},
           "annotation": {"human.action.task_description": {}}},
          open(f"{OUT}/meta/modality.json", "w"), indent=2)
feats = {"observation.state": {"dtype": "float32", "shape": [68]},
         "action": {"dtype": "float32", "shape": [38]}}
for c in CAMS:
    feats[f"observation.images.{c}"] = {"dtype": "video", "shape": [360, 640, 3],
        "names": ["height", "width", "channel"],
        "info": {"video.fps": FPS, "video.height": 360, "video.width": 640, "video.channels": 3,
                 "video.codec": "h264", "video.pix_fmt": "yuv420p", "has_audio": False}}
for k, dt in [("timestamp", "float32"), ("next.reward", "float32"), ("next.done", "bool"),
              ("annotation.human.action.task_description", "int64"), ("task_index", "int64"),
              ("episode_index", "int64"), ("index", "int64")]:
    feats[k] = {"dtype": dt, "shape": [1]}
json.dump({"codebase_version": "v2.0", "robot_type": "vega1_xhand", "fps": FPS,
           "total_episodes": len(recs), "total_frames": gidx, "total_tasks": 1,
           "total_videos": len(recs) * len(CAMS), "total_chunks": 1, "chunks_size": 1000,
           "splits": {"train": f"0:{len(recs)}"},
           "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
           "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
           "features": feats}, open(f"{OUT}/meta/info.json", "w"), indent=1)
print(f"done: {len(recs)} episodes, {gidx} frames -> {OUT}", flush=True)
