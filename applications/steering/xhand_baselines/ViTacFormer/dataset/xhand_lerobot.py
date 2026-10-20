"""XHand LeRobot v3 reader for ViTacFormer (stands in for the h5 pipeline of dataset/data_tactile.py).

One sample per frame t of each listed episode, in the layout of the original pipeline:
  image        [4, 3, 224, 320]  4 cameras at t, ImageNet-normalised (train: crop/rotate/colour aug)
  qpos         [6*38]    state at t-15, t-12, ..., t        (history_cnt 6, sample_rate 3)
  action       [100, 38] action t..t+99 minus state t       (RelativeGenerate rel_abs)
  tactile      [18*60]   force t-17..t: abs 30 + rel to t-17 (tac_history_cnt 18 at 30 Hz)
  tactile_next [18, 60]  force t..t+17: abs 30 + rel to t   (future-tactile target)
Out-of-episode indices take the closest frame (padding_value="closest"); is_pad is all False, as in the original.
Low-dim keys are mean/std normalised with stats from the listed (training) episodes (std < 1e-3 -> 1).
"""
import json, glob, random
import cv2, numpy as np, pandas as pd, torch
from torchcodec.decoders import VideoDecoder

CAMS = ["head_left_rgb", "head_right_rgb", "left_wrist", "right_wrist"]   # file names as recorded (left_wrist shows the right hand)
HIST, STRIDE, TAC_HIST, CHUNK = 6, 3, 18, 100
IMG_HW = (224, 320)
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
KEYS = ("qpos", "action", "tactile", "tactile_next")


def augment(img, rng):
    """Original ImageProcess aug: crop 0.98 + rotate +-0.3 deg, then photometric jitter (cv2, uint8 RGB)."""
    h, w = img.shape[:2]
    y, x = rng.integers(0, int(h * 0.02) + 1), rng.integers(0, int(w * 0.02) + 1)
    img = cv2.resize(img[y:y + int(h * 0.98), x:x + int(w * 0.98)], (w, h))
    img = cv2.warpAffine(img, cv2.getRotationMatrix2D((w // 2, h // 2), rng.uniform(-0.3, 0.3), 1.0), (w, h))
    c_mode, b_flag, c_flag, hsv = rng.integers(2), rng.integers(2), rng.integers(2), rng.integers(4)
    beta, alpha, sat, hue = rng.uniform(-20, 20), rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2), rng.integers(-18, 18)
    cv = lambda im, a=1.0, b=0.0: np.clip(im.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    if b_flag: img = cv(img, b=beta)
    if c_mode and c_flag: img = cv(img, a=alpha)
    if hsv:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        if hsv in (1, 3): img[..., 1] = cv(img[..., 1], a=sat)
        if hsv in (2, 3): img[..., 0] = ((img[..., 0].astype(int) + hue) % 180).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    if c_mode and c_flag: img = cv(img, a=alpha)                 # applied twice, as in the original
    return img


class XHandLeRobot(torch.utils.data.Dataset):
    def __init__(self, root, episodes, train=True, stats=None, tactile=True):
        self.root, self.train, self.tactile = root, train, tactile
        info = json.load(open(f"{root}/meta/info.json"))
        self.fps, vpath = info["fps"], info["video_path"]
        meta = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet"))])
        meta = meta.set_index("episode_index")
        self.eps, self.low, self.vid, self.task, self.index = list(episodes), {}, {}, {}, []
        for e in self.eps:
            m = meta.loc[e]
            df = pd.read_parquet(f"{root}/" + info["data_path"].format(chunk_index=m["data/chunk_index"], file_index=m["data/file_index"]))
            df = df[df.episode_index == e].sort_values("frame_index")
            self.low[e] = {k: np.stack(df[c].to_numpy()).astype(np.float32) for k, c in
                           (("state", "observation.state"), ("action", "action"), ("force", "observation.tactile_force"))}
            self.vid[e] = {c: (f"{root}/" + vpath.format(video_key=f"observation.images.{c}",
                                                         chunk_index=m[f"videos/observation.images.{c}/chunk_index"],
                                                         file_index=m[f"videos/observation.images.{c}/file_index"]),
                               int(round(m[f"videos/observation.images.{c}/from_timestamp"] * self.fps))) for c in CAMS}
            self.task[e] = str(m["tasks"][0])
            self.index += [(e, t) for t in range(len(df))]
        self.stats = stats if stats is not None else self.compute_stats()

    def lowdim(self, e, t):
        L = self.low[e]; n = len(L["state"]); at = lambda o: np.clip(t + np.asarray(o), 0, n - 1)
        out = {"qpos": L["state"][at(range(-(HIST - 1) * STRIDE, 1, STRIDE))],
               "action": L["action"][at(range(CHUNK))] - L["state"][t]}
        if self.tactile:
            for k, o in (("tactile", range(-(TAC_HIST - 1), 1)), ("tactile_next", range(TAC_HIST))):
                f = L["force"][at(o)]; out[k] = np.concatenate([f, f - f[0]], 1)
        return out

    def compute_stats(self, every=10):        # original: every 10th sample, per-channel moments over all window rows
        acc = {}
        for e, t in self.index[::every]:
            for k, v in self.lowdim(e, t).items(): acc.setdefault(k, []).append(v)
        stats = {}
        for k, v in acc.items():
            v = np.concatenate(v).astype(np.float64); sd = v.std(0)
            stats[k] = {"mean": v.mean(0).astype(np.float32), "std": np.where(np.abs(sd) < 1e-3, 1, sd).astype(np.float32)}
        return stats

    def frames(self, e, ts):
        """uint8 RGB [len(ts), n_cam, H, W, 3] for frames ts of episode e (sequential decode for sorted ts)."""
        out = []
        for c in CAMS:
            path, off = self.vid[e][c]
            out.append(VideoDecoder(path, dimension_order="NHWC").get_frames_at([off + int(t) for t in ts]).data.numpy())
        return np.stack(out, 1)

    def item(self, e, t, frames=None):
        if frames is None: frames = self.frames(e, [t])[0]
        rng = np.random.default_rng(np.random.randint(2 ** 31)) if self.train else None
        imgs = []
        for im in frames:
            im = cv2.resize(im, IMG_HW[::-1], interpolation=cv2.INTER_LINEAR)
            if self.train: im = augment(im, rng)
            imgs.append(((im.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1))
        low = {k: (v - self.stats[k]["mean"]) / self.stats[k]["std"] for k, v in self.lowdim(e, t).items()}
        s = {"image": torch.from_numpy(np.stack(imgs)), "qpos": torch.from_numpy(low["qpos"].reshape(-1)),
             "action": torch.from_numpy(low["action"]), "is_pad": torch.zeros(CHUNK, dtype=torch.bool)}
        if self.tactile:
            s["tactile"] = torch.from_numpy(low["tactile"].reshape(-1)); s["tactile_next"] = torch.from_numpy(low["tactile_next"])
        return s

    def __len__(self): return len(self.index)
    def __getitem__(self, i): return self.item(*self.index[i])


def worker_init(_):                           # numpy/random are not reseeded per DataLoader worker by torch
    s = torch.initial_seed() % 2 ** 32; np.random.seed(s); random.seed(s); torch.set_num_threads(1); cv2.setNumThreads(1)
