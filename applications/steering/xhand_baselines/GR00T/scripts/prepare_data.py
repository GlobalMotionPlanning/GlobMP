#!/usr/bin/env python3
"""Prepare LeRobot v2.0 datasets for the DexMate Vega pick-move-scale teleop.

Each raw episode is a self-contained .tar holding:
    <ep>/metadata.json
    <ep>/transitions.pkl          # list[ {timestamp_ns, state{joint_pos,joint_vel}, action{...}, metadata} ]
    <ep>/{head_left_rgb,head_right_rgb,left_wrist_rgb,right_wrist_rgb}/frame_NNNNNN.jpg

Each demo is pick -> base move -> place. The base move is the only time the chassis
velocity {vx,vy,wz} is non-zero, so the episode is segmented on chassis speed:
    pick  = [0 .. move_start)
    move  = [move_start .. move_end)      # dropped
    place = [move_end .. N)
The pick window is written to <out>/pick, the place window to <out>/place.

Output is 22-DOF absolute joints + 4 RGB cams; chassis is excluded from the action.
Video is encoded h264/yuv420p (decoded by the GR00T loader via torchcodec/decord).

Robustness:
  * transient tar read errors are retried (READ_RETRIES / RETRY_SLEEP).
  * RESUMABLE: converted sources are tracked in <out>/processed_sources.json and skipped;
    pick/place stay index-aligned. Re-run the same command to fill any gaps.
  * stats.json is recomputed from all parquet at finalize, so resume stays correct.

Usage:
    SRC_DIRS=<data>/teleop/pick_move_scale_20260622,<data>/teleop/pick_move_scale_20260623 \
      OUT_BASE=<data>/data python scripts/prepare_data.py
    MAX_EPISODES=2 python scripts/prepare_data.py            # smoke test on 2 episodes
"""
import os, sys, json, glob, time, tarfile, tempfile, shutil
import numpy as np

SRC_DIRS = os.environ.get("SRC_DIRS",
    "<data>/teleop/pick_move_scale_20260622,"
    "<data>/teleop/pick_move_scale_20260623").split(",")
OUT_BASE = os.environ.get("OUT_BASE", "<data>/data")
MAX_EPISODES = int(os.environ.get("MAX_EPISODES", "0"))   # 0 = all
READ_RETRIES = int(os.environ.get("READ_RETRIES", "5"))
RETRY_SLEEP = float(os.environ.get("RETRY_SLEEP", "4"))
FPS = 30
CAMS = ["head_left_rgb", "head_right_rgb", "left_wrist_rgb", "right_wrist_rgb"]
GROUPS = ["left_arm", "right_arm", "torso", "head", "left_hand", "right_hand"]  # -> 22 dims
SLICES = [(0, 7), (7, 14), (14, 17), (17, 20), (20, 21), (21, 22)]
JOINT_NAMES = (["left_arm_j%d" % i for i in range(7)] + ["right_arm_j%d" % i for i in range(7)] +
               ["torso_j%d" % i for i in range(3)] + ["head_j%d" % i for i in range(3)] +
               ["left_hand", "right_hand"])
MOVE_REL_THR = float(os.environ.get("MOVE_REL_THR", "0.05"))
MIN_SEG_LEN = int(os.environ.get("MIN_SEG_LEN", "20"))
SKILLS = {
    "pick":  "pick: grasp the white box and lift it clear of the cart.",
    "place": "scale: move the box onto the scale and set it down stable.",
}


def _vec(d, keys):
    return np.concatenate([np.atleast_1d(np.asarray(d[k], dtype=np.float32)).ravel() for k in keys])


def parse_transitions(trans):
    state, action, chvel = [], [], []
    for s in trans:
        state.append(_vec(s["state"]["joint_pos"], GROUPS))
        action.append(_vec({k: s["action"][k]["pos"] for k in GROUPS}, GROUPS))
        c = s["action"]["chassis"]
        chvel.append([float(c["vx"]), float(c["vy"]), float(c["wz"])])
    state = np.asarray(state, np.float32)
    action = np.asarray(action, np.float32)
    chspeed = np.linalg.norm(np.asarray(chvel, np.float32), axis=1)
    return state, action, chspeed


def segment(chspeed):
    mx = float(chspeed.max())
    if mx <= 1e-6:
        raise ValueError("chassis never moves (no detectable move phase)")
    moving = chspeed > MOVE_REL_THR * mx
    idx = np.where(moving)[0]
    move_start, move_end = int(idx.min()), int(idx.max()) + 1
    pick = slice(0, move_start)
    place = slice(move_end, len(chspeed))
    if (pick.stop - pick.start) < MIN_SEG_LEN:
        raise ValueError("pick window too short (%d)" % (pick.stop - pick.start))
    if (place.stop - place.start) < MIN_SEG_LEN:
        raise ValueError("place window too short (%d)" % (place.stop - place.start))
    return pick, place


def extract_with_retry(tar, dest):
    """Extract a .tar to dest, retrying on transient I/O errors. Returns episode dir name."""
    last = None
    for attempt in range(1, READ_RETRIES + 1):
        try:
            with tarfile.open(tar) as tf:
                name = tf.getnames()[0].split("/")[0]
                tf.extractall(dest)
            return name
        except (OSError, tarfile.TarError) as e:
            last = e
            if attempt < READ_RETRIES:
                time.sleep(RETRY_SLEEP * attempt)
                # clear partial extraction before retry
                for x in glob.glob(os.path.join(dest, "*")):
                    shutil.rmtree(x, ignore_errors=True) if os.path.isdir(x) else os.remove(x)
    raise last


def load_frames(ep_root, ep_name, n_steps):
    from PIL import Image
    per_cam = {}
    for cam in CAMS:
        files = sorted(glob.glob(os.path.join(ep_root, ep_name, cam, "frame_*.jpg")),
                       key=lambda p: int(os.path.basename(p)[6:12]))
        if not files:
            raise ValueError("camera %s has no frames" % cam)
        per_cam[cam] = files
    n = min([n_steps] + [len(per_cam[c]) for c in CAMS])
    frames = {cam: np.stack([np.asarray(Image.open(f).convert("RGB"), np.uint8)
                             for f in per_cam[cam][:n]]) for cam in CAMS}
    return frames, n


def encode_mp4(path, frames):
    import av
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h, w = frames.shape[1], frames.shape[2]
    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    stream.options = {"crf": "23"}
    for img in frames:
        for pkt in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


class DatasetWriter:
    def __init__(self, root, task_str):
        self.root, self.task = root, task_str
        os.makedirs(os.path.join(root, "data", "chunk-000"), exist_ok=True)
        for cam in CAMS:
            os.makedirs(os.path.join(root, "videos", "chunk-000",
                        "observation.images.%s" % cam), exist_ok=True)
        # resume existing state
        self.episodes = []
        epf = os.path.join(root, "meta", "episodes.jsonl")
        if os.path.exists(epf):
            self.episodes = [json.loads(l) for l in open(epf)]
        self.ep = len(self.episodes)
        self.gindex = sum(e["length"] for e in self.episodes)

    def add(self, state, action, frames, source):
        import pandas as pd
        n = state.shape[0]
        ts = (np.arange(n) / FPS).astype(np.float64)
        done = np.zeros(n, bool); done[-1] = True
        df = pd.DataFrame({
            "observation.state": list(state.astype(np.float32)),
            "action": list(action.astype(np.float32)),
            "timestamp": ts,
            "annotation.human.action.task_description": np.zeros(n, np.int64),
            "task_index": np.zeros(n, np.int64),
            "episode_index": np.full(n, self.ep, np.int64),
            "index": np.arange(self.gindex, self.gindex + n, dtype=np.int64),
            "next.reward": np.zeros(n, np.float64),
            "next.done": done,
        })
        df.to_parquet(os.path.join(self.root, "data", "chunk-000",
                      "episode_%06d.parquet" % self.ep), index=False)
        for cam in CAMS:
            encode_mp4(os.path.join(self.root, "videos", "chunk-000",
                       "observation.images.%s" % cam, "episode_%06d.mp4" % self.ep), frames[cam])
        self.episodes.append({"episode_index": self.ep, "tasks": [self.task],
                              "length": int(n), "source": source})
        self.gindex += n; self.ep += 1

    def finalize(self, sample_hw):
        import pandas as pd
        if self.ep == 0:
            print("  [%s] no episodes" % self.root); return
        h, w = sample_hw
        feat = {"observation.state": {"dtype": "float32", "shape": [22], "names": JOINT_NAMES},
                "action": {"dtype": "float32", "shape": [22], "names": JOINT_NAMES}}
        for cam in CAMS:
            feat["observation.images.%s" % cam] = {
                "dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "channel"],
                "video_info": {"video.fps": float(FPS), "video.codec": "h264",
                               "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False}}
        for k in ["timestamp", "next.reward"]:
            feat[k] = {"dtype": "float32", "shape": [1], "names": None}
        for k in ["annotation.human.action.task_description", "task_index", "episode_index", "index"]:
            feat[k] = {"dtype": "int64", "shape": [1], "names": None}
        feat["next.done"] = {"dtype": "bool", "shape": [1], "names": None}
        total_frames = sum(e["length"] for e in self.episodes)
        info = {"codebase_version": "v2.0", "robot_type": "vega_1p_gripper", "fps": FPS,
                "total_episodes": self.ep, "total_frames": total_frames, "chunks_size": 1000,
                "splits": {"train": "0:%d" % self.ep},
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": feat}
        meta = os.path.join(self.root, "meta"); os.makedirs(meta, exist_ok=True)
        json.dump(info, open(os.path.join(meta, "info.json"), "w"), indent=2)
        with open(os.path.join(meta, "episodes.jsonl"), "w") as f:
            for e in self.episodes:
                f.write(json.dumps(e) + "\n")
        with open(os.path.join(meta, "tasks.jsonl"), "w") as f:
            f.write(json.dumps({"task_index": 0, "task": self.task}) + "\n")
        modality = {"state": {g: {"start": s, "end": e} for g, (s, e) in zip(GROUPS, SLICES)},
                    "action": {g: {"start": s, "end": e} for g, (s, e) in zip(GROUPS, SLICES)},
                    "video": {cam: {"original_key": "observation.images.%s" % cam} for cam in CAMS},
                    "annotation": {"human.action.task_description": {}}}
        json.dump(modality, open(os.path.join(meta, "modality.json"), "w"), indent=2)
        # recompute stats from all parquet (resume-safe)
        S, A, T, R = [], [], [], []
        for i in range(self.ep):
            df = pd.read_parquet(os.path.join(self.root, "data", "chunk-000", "episode_%06d.parquet" % i))
            S.append(np.stack(df["observation.state"].values))
            A.append(np.stack(df["action"].values))
            T.append(df["timestamp"].values.reshape(-1, 1))
            R.append(df["next.reward"].values.reshape(-1, 1))
        def st(arr):
            a = np.concatenate(arr, 0).astype(np.float64)
            return {"mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
                    "min": a.min(0).tolist(), "max": a.max(0).tolist(),
                    "q01": np.quantile(a, 0.01, 0).tolist(), "q99": np.quantile(a, 0.99, 0).tolist()}
        json.dump({"observation.state": st(S), "action": st(A),
                   "timestamp": st(T), "next.reward": st(R)},
                  open(os.path.join(meta, "stats.json"), "w"), indent=2)
        print("  [%s] %d episodes, %d frames" % (self.root, self.ep, total_frames))


def discover():
    tars, seen = [], set()
    for d in SRC_DIRS:
        for t in sorted(glob.glob(os.path.join(d, "*.tar"))):
            b = os.path.basename(t)
            if b not in seen:
                seen.add(b); tars.append(t)
    return tars


def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    proc_path = os.path.join(OUT_BASE, "processed_sources.json")
    processed = set(json.load(open(proc_path))) if os.path.exists(proc_path) else set()
    tars = discover()
    if MAX_EPISODES:
        tars = tars[:MAX_EPISODES]
    todo = [t for t in tars if os.path.basename(t) not in processed]
    print("discovered=%d already_done=%d todo=%d -> %s" % (
        len(tars), len(tars) - len(todo), len(todo), OUT_BASE))
    writers = {name: DatasetWriter(os.path.join(OUT_BASE, name), task) for name, task in SKILLS.items()}
    skipped, sample_hw = [], None
    tmpbase = os.environ.get("PREP_TMPDIR") or None
    for i, tar in enumerate(todo):
        b = os.path.basename(tar)
        tmp = tempfile.mkdtemp(prefix="ep_", dir=tmpbase)
        try:
            ep_name = extract_with_retry(tar, tmp)
            trans = __import__("pickle").load(open(os.path.join(tmp, ep_name, "transitions.pkl"), "rb"))
            state, action, chspeed = parse_transitions(trans)
            pick, place = segment(chspeed)
            frames, n = load_frames(tmp, ep_name, len(trans))
            state, action = state[:n], action[:n]
            pick = slice(pick.start, min(pick.stop, n)); place = slice(min(place.start, n), n)
            if (pick.stop - pick.start) < MIN_SEG_LEN or (place.stop - place.start) < MIN_SEG_LEN:
                raise ValueError("segment too short after frame alignment")
            if sample_hw is None:
                sample_hw = (frames[CAMS[0]].shape[1], frames[CAMS[0]].shape[2])
            writers["pick"].add(state[pick], action[pick], {c: frames[c][pick] for c in CAMS}, b)
            writers["place"].add(state[place], action[place], {c: frames[c][place] for c in CAMS}, b)
            processed.add(b)
            json.dump(sorted(processed), open(proc_path, "w"))
            print("[%d/%d] %s N=%d pick=%d move=%d place=%d" % (
                i + 1, len(todo), ep_name, n, pick.stop - pick.start,
                place.start - pick.stop, place.stop - place.start), flush=True)
        except Exception as e:
            skipped.append((b, str(e)))
            print("[%d/%d] SKIP %s : %s" % (i + 1, len(todo), b, str(e)[:120]), flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    if sample_hw is None:
        sample_hw = (600, 960)
    for w in writers.values():
        w.finalize(sample_hw)
    print("\n=== DONE this pass. newly_converted=%d skipped=%d total_in_set=%d ===" % (
        len(todo) - len(skipped), len(skipped), writers["pick"].ep))
    for b, why in skipped:
        print("  SKIP", b, "::", why[:120])


if __name__ == "__main__":
    main()
