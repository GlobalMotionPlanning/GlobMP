#!/usr/bin/env python3
"""Open-loop chunked rollout of an XHand GR00T checkpoint on held-out episodes.

Deployment shape is the one the T-Rex repro used: at every chunk boundary the policy sees the
REAL observation (4 cameras, 38-D or 68-D state, instruction) and emits a 40-step chunk; the
first CHUNK steps are replayed without re-querying. GR00T predicts absolute 38-D joint targets,
so no IK is needed. GT is the recorded action, as elsewhere in the bench.

  38-D layout: [L_arm 0:7 | L_hand 7:19 | R_arm 19:26 | R_hand 26:38]
  state carries tactile at [38:68]; only the +tactile config names it.
"""
import argparse, importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd
from torchcodec.decoders import VideoDecoder
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

SL = [("left_arm", 0, 7), ("left_hand", 7, 19), ("right_arm", 19, 26), ("right_hand", 26, 38)]
CAMS = ["head_left_rgb", "head_right_rgb", "left_wrist", "right_wrist"]
LANG = "annotation.human.action.task_description"

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", required=True, help="json list of episode indices")
ap.add_argument("--ckpt", required=True); ap.add_argument("--outdir", required=True)
ap.add_argument("--dataset", required=True, help="data/<task>/gr00t")
ap.add_argument("--modality-config", required=True)
ap.add_argument("--device", default="cuda:0"); ap.add_argument("--chunk", type=int, default=16)
ap.add_argument("--tactile", action="store_true")
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("mod", a.modality_config)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

policy = Gr00tPolicy(embodiment_tag=EmbodimentTag.NEW_EMBODIMENT, model_path=a.ckpt, device=a.device)
Path(a.outdir).mkdir(parents=True, exist_ok=True)
eps = json.load(open(a.episodes)) if a.episodes.endswith(".json") else [int(x) for x in a.episodes.split(",")]
meta = {e["episode_index"]: e for e in map(json.loads, open(f"{a.dataset}/meta/episodes.jsonl"))}

for i in eps:
    out = f"{a.outdir}/ep{i:03d}.npz"
    if Path(out).exists():
        print(f"ep {i}: exists, skip", flush=True); continue
    task = meta[i]["tasks"][0]
    df = pd.read_parquet(f"{a.dataset}/data/chunk-000/episode_{i:06d}.parquet")
    S = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    A = np.stack(df["action"].to_numpy()).astype(np.float32)
    dec = {c: VideoDecoder(f"{a.dataset}/videos/chunk-000/observation.images.{c}/episode_{i:06d}.mp4")
           for c in CAMS}
    n = min(len(S), len(A), min(d.metadata.num_frames for d in dec.values()))
    starts = list(range(0, n, a.chunk))
    pred = np.zeros((n, 38), np.float32)
    for s in starts:
        st = {g: S[s][lo:hi][None, None] for g, lo, hi in SL}
        if a.tactile: st["tactile"] = S[s][38:68][None, None]
        obs = {"video": {c: dec[c][s].permute(1, 2, 0).cpu().numpy().astype(np.uint8)[None, None] for c in CAMS},
               "state": st, "language": {LANG: [[task]]}}
        act, _ = policy.get_action(obs)
        H = np.asarray(act["left_arm"]).shape[1]
        ch = np.stack([np.concatenate([np.asarray(act[g])[0, k].reshape(-1) for g, _, _ in SL])
                       for k in range(H)])[:, :38]
        k = min(a.chunk, n - s); pred[s:s + k] = ch[:k]
    gt = A[:n].copy()
    np.savez(out, gt_joints=gt, pred_joints=pred, raw_joints=S[:n, :38], tactile=S[:n, 38:68],
             chunk_starts=np.array(starts), task=np.array(task), episode=i)
    arm = np.r_[0:7, 19:26]; hnd = np.r_[7:19, 26:38]
    print(f"ep {i:3d}  n={n:5d}  arm |err| med={np.median(np.abs(pred[:,arm]-gt[:,arm])):.4f}  "
          f"hand med={np.median(np.abs(pred[:,hnd]-gt[:,hnd])):.4f} rad", flush=True)
print("done")
