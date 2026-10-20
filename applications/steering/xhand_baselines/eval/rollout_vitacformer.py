#!/usr/bin/env python3
"""Open-loop chunked rollout of a ViTacFormer checkpoint on held-out episodes (npz schema of rollout_gr00t.py).

Observations are built by the training reader (VTF_DIR/dataset/xhand_lerobot.py, no augmentation) with the
checkpoint's training stats. At every CHUNK-th step the policy sees the REAL observation (4 cameras, state
history, tactile history), predicts its 100-step relative chunk with the prior latent (z = 0) and its own
predicted future tactile; absolute targets = chunk + current state, and the first CHUNK are kept. GT is the
recorded action. No temporal ensembling (the bench protocol, not the original deployment).

  VTF_DIR=<ViTacFormer checkout> python eval/rollout_vitacformer.py --ckpt <ckpt/policy_last.ckpt> --dataset <data/task/lerobot> \
      --episodes <split/heldout.json> --outdir <rollouts>
"""
import argparse, json, os, sys, types
from pathlib import Path
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--dataset", required=True)
ap.add_argument("--episodes", required=True); ap.add_argument("--outdir", required=True)
ap.add_argument("--chunk", type=int, default=16); ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
ap.add_argument("--limit", type=int, default=0, help="first N episodes only (testing)")
ap.add_argument("--max-chunks", type=int, default=0, help="first N chunks per episode only (testing)")
a = ap.parse_args()

VTF = os.environ["VTF_DIR"]                              # ViTacFormer checkout + third_party/vitacformer_xhand.patch
sys.path[:0] = [VTF, os.path.join(VTF, "detr")]
try: import IPython  # noqa: F401  (imported, unused, by the model files)
except ImportError: sys.modules["IPython"] = types.SimpleNamespace(embed=None)
from dataset.xhand_lerobot import XHandLeRobot
from policy import ACTPolicy

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
policy = ACTPolicy(ck["policy_config"]); policy.load_state_dict(ck["model"]); policy.to(a.device).eval()
tac = ck["tactile"]
eps = json.load(open(a.episodes)) if a.episodes.endswith(".json") else [int(x) for x in a.episodes.split(",")]
if a.limit: eps = eps[:a.limit]
ds = XHandLeRobot(a.dataset, eps, train=False, stats=ck["stats"], tactile=tac)
am, asd = (torch.as_tensor(ck["stats"]["action"][k], device=a.device) for k in ("mean", "std"))
print(f"[vitacformer] loaded {a.ckpt} step={ck['step']} tactile={tac} chunk={ck['policy_config']['num_queries']}", flush=True)
Path(a.outdir).mkdir(parents=True, exist_ok=True)

for i in eps:
    out = f"{a.outdir}/ep{i:03d}.npz"
    if os.path.exists(out): print(f"ep {i}: exists, skip", flush=True); continue
    L = ds.low[i]; n = len(L["state"]); gt = L["action"]
    starts = list(range(0, n, a.chunk))
    if a.max_chunks: starts = starts[:a.max_chunks]
    frames = ds.frames(i, starts)
    pred = np.zeros_like(gt)
    for s, fr in zip(starts, frames):
        b = {k: v.unsqueeze(0).to(a.device) for k, v in ds.item(i, s, fr).items()}
        with torch.no_grad():
            rel = policy(b["qpos"], b["image"], tactile=b["tactile"] if tac else None)[0] * asd + am
        ch = (rel + torch.as_tensor(L["state"][s], device=a.device)).cpu().numpy(); k = min(a.chunk, n - s)
        pred[s:s + k] = ch[:k]
    np.savez(out, gt_joints=gt, pred_joints=pred, raw_joints=L["state"], tactile=L["force"],
             chunk_starts=np.array(starts), task=np.array(ds.task[i]), episode=i)
    arm = np.r_[0:7, 19:26]; hnd = np.r_[7:19, 26:38]; m = slice(0, min(n, starts[-1] + a.chunk))
    print(f"ep {i:3d}  n={n:5d}  arm |err| med={np.median(np.abs(pred[m][:, arm] - gt[m][:, arm])):.4f}  "
          f"hand med={np.median(np.abs(pred[m][:, hnd] - gt[m][:, hnd])):.4f} rad", flush=True)
print("done")
