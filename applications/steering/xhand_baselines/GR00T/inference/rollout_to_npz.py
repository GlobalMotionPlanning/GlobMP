#!/usr/bin/env python3
"""Open-loop rollout of a trained GR00T baseline over a LeRobot pick episode.

Feeds ground-truth state + the episode's real camera frames to the Gr00tPolicy, records
the predicted first-step action at each timestep, and saves gt/pred [L,22] to an npz —
the input format for the Genesis 2-way GT-vs-model replay.

    PYTHONPATH=<groot> GROOT_BACKBONE_PATH=<cosmos> \
      python inference/rollout_to_npz.py --ckpt <model_dir> --dataset <lerobot_pick> \
             --episode 0 --task "pick: grasp the white box and lift it clear of the cart." \
             --out /tmp/groot_pick_ep0.npz
"""
import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
from torchcodec.decoders import VideoDecoder

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

STATE_SLICES = [("left_arm", 0, 7), ("right_arm", 7, 14), ("torso", 14, 17),
                ("head", 17, 20), ("left_hand", 20, 21), ("right_hand", 21, 22)]
GROUPS = [s[0] for s in STATE_SLICES]
CAMS = ["head_left_rgb", "head_right_rgb", "left_wrist_rgb", "right_wrist_rgb"]
LANG_KEY = "annotation.human.action.task_description"

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--episode", type=int, default=0)
ap.add_argument("--task", default=None,
                help="instruction for the model; default: the episode's OWN task from "
                     "meta/episodes.jsonl (never guess a prompt for a multi-task dataset)")
ap.add_argument("--modality-config", default=str(Path(__file__).resolve().parent.parent / "configs" / "vega_pick_modality.py"))
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--exec-horizon", type=int, default=8,
                help="steps executed per predicted chunk (mirrors deploy; 1 = per-frame chunks, noisy)")
ap.add_argument("--ensemble-stride", type=int, default=0,
                help=">0: ACT-style temporal ensembling — predict a full chunk every N frames and "
                     "average all chunks covering each timestep (overrides --exec-horizon)")
ap.add_argument("--out", required=True)
args = ap.parse_args()

if args.task is None:
    import json as _json
    with open(Path(args.dataset) / "meta" / "episodes.jsonl") as _f:
        for _line in _f:
            _rec = _json.loads(_line)
            if _rec["episode_index"] == args.episode:
                args.task = _rec["tasks"][0]
                break
    if args.task is None:
        raise SystemExit(f"episode {args.episode} not found in {args.dataset}/meta/episodes.jsonl")
    print(f"[rollout] task (from dataset): {args.task!r}")

# register the vega modality config for EmbodimentTag.NEW_EMBODIMENT
spec = importlib.util.spec_from_file_location("vega_pick_modality", args.modality_config)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def load22(df, prefix):
    if prefix in df.columns:
        return np.stack(df[prefix].to_numpy()).astype(np.float32)
    parts = [np.stack(df[f"{prefix}.{c}"].to_numpy()).astype(np.float32).reshape(len(df), -1) for c in GROUPS]
    return np.concatenate(parts, axis=1)


df = pd.read_parquet(f"{args.dataset}/data/chunk-000/episode_{args.episode:06d}.parquet")
state = load22(df, "observation.state")
action_arr = load22(df, "action")
frames = {}
for cam in CAMS:
    dec = VideoDecoder(f"{args.dataset}/videos/chunk-000/observation.images.{cam}/episode_{args.episode:06d}.mp4")
    frames[cam] = [dec[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8) for i in range(dec.metadata.num_frames)]
L = min(len(state), len(action_arr), min(len(frames[c]) for c in CAMS))
print(f"episode {args.episode}: L={L}", flush=True)

# ---- episode deep-check: camera stability (chassis leak) + idle stats + lead trim ----
arm_v = np.abs(np.diff(state[:L, :14], axis=0)).max(axis=1)
idle_frac = float((arm_v < 0.002).mean())
band = [f[::4, ::4].mean(axis=2)[:40] for f in frames[CAMS[0]][:L:15]]  # background band, strided
shifts = []
for a, b in zip(band[:-1], band[1:]):
    A, B = np.fft.fft2(a), np.fft.fft2(b)
    r = np.fft.ifft2(A * np.conj(B) / (np.abs(A * np.conj(B)) + 1e-9))
    dy, dx = np.unravel_index(np.argmax(np.abs(r)), r.shape)
    dy = dy if dy <= r.shape[0] // 2 else dy - r.shape[0]
    dx = dx if dx <= r.shape[1] // 2 else dx - r.shape[1]
    shifts.append(float(np.hypot(dy, dx)))
cam_shift = max(shifts) if shifts else 0.0
moving = np.flatnonzero(arm_v > 0.005)
trim = max(0, int(moving[0]) - 15) if len(moving) else 0
print(f"[check] idle_frac={idle_frac:.2f}  cam_shift_max={cam_shift:.1f}px  trim_lead={trim}", flush=True)
if cam_shift > 2.0:
    print("[check] WARNING: background shifts >2px — possible chassis motion inside this episode", flush=True)
if trim:
    state, action_arr = state[trim:], action_arr[trim:]
    for cam in CAMS:
        frames[cam] = frames[cam][trim:]
    L -= trim
    print(f"[check] trimmed idle lead-in: rollout starts at frame {trim}, L={L}", flush=True)

policy = Gr00tPolicy(embodiment_tag=EmbodimentTag.NEW_EMBODIMENT, model_path=args.ckpt, device=args.device)

def predict_chunk(t):
    obs = {
        "video": {cam: frames[cam][t][None, None] for cam in CAMS},          # (1,1,H,W,3)
        "state": {g: state[t][a:b][None, None] for g, a, b in STATE_SLICES},  # (1,1,D)
        "language": {LANG_KEY: [[args.task]]},
    }
    action, _ = policy.get_action(obs)
    H = np.asarray(action[GROUPS[0]]).shape[1]
    return np.stack([np.concatenate([np.asarray(action[g])[0, i].reshape(-1) for g in GROUPS])[:22]
                     for i in range(H)])

pred = np.zeros((L, 22), dtype=np.float32)
if args.ensemble_stride > 0:
    # ACT-style temporal ensembling: every timestep = mean over all chunks covering it
    S = args.ensemble_stride
    acc = np.zeros((L, 22), dtype=np.float64)
    cnt = np.zeros(L, dtype=np.int64)
    for t in range(0, L, S):
        chunk = predict_chunk(t)
        n = min(len(chunk), L - t)
        acc[t:t + n] += chunk[:n]
        cnt[t:t + n] += 1
        if (t // S) % 10 == 0:
            print(f"  t={t}/{L} (ensemble)", flush=True)
    pred = (acc / np.maximum(cnt, 1)[:, None]).astype(np.float32)
else:
    K = max(1, args.exec_horizon)
    for t in range(0, L, K):
        chunk = predict_chunk(t)
        n = min(K, L - t)
        pred[t:t + n] = chunk[:n]
        if (t // K) % 10 == 0:
            print(f"  t={t}/{L}", flush=True)

# GT panel replays the OBSERVED robot state (what the robot actually did);
# the commanded actions are kept alongside for the open-loop MAE.
gt_state = state[:L]
gt_action = action_arr[:L]
mae = float(np.mean(np.abs(gt_action[:, :14] - pred[:, :14])))
import os as _os
_d = _os.path.dirname(args.out)
if _d:
    _os.makedirs(_d, exist_ok=True)
np.savez(args.out, gt=gt_state, gt_action=gt_action, pred=pred, trim_lead=trim,
         idle_frac=idle_frac, cam_shift_max=cam_shift)
print(f"saved {args.out}: gt{gt_state.shape} pred{pred.shape}  arm MAE vs action={mae:.5f}", flush=True)
