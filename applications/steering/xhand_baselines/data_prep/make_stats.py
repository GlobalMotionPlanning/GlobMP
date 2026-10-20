#!/usr/bin/env python3
"""Compute meta/stats.json for the XHand dataset.

lerobot normalises from these, and the tactile path in finetune/train_lerobot.py indexes stats directly --
without the file, ds.meta.stats is None and any --tactile run dies before the first batch.
Image stats are per-channel over a frame subsample; vector stats are exact over all frames.
"""
import glob, json, os
import numpy as np, pyarrow.parquet as pq
from torchcodec.decoders import VideoDecoder

D = os.environ["OUT"]
info = json.load(open(f"{D}/meta/info.json"))
VEC = [k for k, v in info["features"].items() if v["dtype"] != "video"]
IMG = [k for k, v in info["features"].items() if v["dtype"] == "video"]

cols = {k: [] for k in VEC}
for f in sorted(glob.glob(f"{D}/data/chunk-000/*.parquet")):
    t = pq.read_table(f)
    for k in VEC:
        a = np.asarray(t[k].to_pylist(), dtype=np.float64)
        cols[k].append(a.reshape(len(a), -1))

stats = {}
for k in VEC:
    a = np.concatenate(cols[k], 0)
    stats[k] = {"min": a.min(0).tolist(), "max": a.max(0).tolist(),
                "mean": a.mean(0).tolist(), "std": (a.std(0) + 1e-8).tolist(),
                "count": [int(len(a))],
                **{f"q{q:02d}": np.percentile(a, q, axis=0).tolist() for q in (1, 10, 50, 90, 99)}}
    print(f"  {k:32} {a.shape}")

for k in IMG:
    vs = sorted(glob.glob(f"{D}/videos/{k}/chunk-000/*.mp4"))[:12]
    px = []
    for v in vs:
        d = VideoDecoder(v); n = d.metadata.num_frames
        for i in np.linspace(0, n - 1, 6).astype(int):
            px.append(d[int(i)].permute(1, 2, 0).cpu().numpy().reshape(-1, 3) / 255.0)
    p = np.concatenate(px, 0)
    stats[k] = {"min": p.min(0).reshape(3, 1, 1).tolist(), "max": p.max(0).reshape(3, 1, 1).tolist(),
                "mean": p.mean(0).reshape(3, 1, 1).tolist(), "std": (p.std(0) + 1e-8).reshape(3, 1, 1).tolist(),
                "count": [int(len(p))]}
    print(f"  {k:32} sampled {len(vs)} videos")

json.dump(stats, open(f"{D}/meta/stats.json", "w"))
print(f"wrote {D}/meta/stats.json  ({len(stats)} features)")
