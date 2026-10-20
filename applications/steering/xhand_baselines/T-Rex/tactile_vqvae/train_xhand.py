"""Tactile VQ-VAE for XHand1 force (5 fingers x 3 per hand), single GPU.

Same model and loss as tactile_vqvae/train.py (per-finger codes, K=64, W=16), but
reads the XHand teleop h5 files (and the toy-on-shelf LeRobot parquet) directly
instead of the midtrain manifests.  Episodes listed in --exclude (held-out test
episodes) are never seen.

Output: {config, model_state, stats} blob, the format scripts/train.py --vqvae_ckpt loads.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import h5py
import numpy as np
import torch

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from tactile_vqvae.models import TactileVQVAE
from tactile_vqvae.models.tactile_vqvae import TactileVQVAEConfig

NF, D = 5, 3   # fingers per hand, force channels


def load_h5_roots(roots, exclude):
    eps = []
    for root in roots:
        for h5 in sorted(glob.glob(f"{root}/episode_*/*.h5")):
            name = os.path.basename(os.path.dirname(h5))
            if name in exclude:
                continue
            with h5py.File(h5, "r") as f:
                tac = np.concatenate([f["left_hand_tactile_force"][:],
                                      f["right_hand_tactile_force"][:]], axis=1)   # [N,10,3]
            eps.append((f"{os.path.basename(root)}/{name}", tac.astype(np.float32)))
    return eps


def load_parquet_root(root):
    import pyarrow.parquet as pq
    by_ep = {}
    for p in sorted(glob.glob(f"{root}/data/chunk-*/*.parquet")):
        t = pq.read_table(p, columns=["observation.tactile_force", "episode_index"]).to_pydict()
        for tf, e in zip(t["observation.tactile_force"], t["episode_index"]):
            by_ep.setdefault(e, []).append(tf)
    return [(f"{os.path.basename(root)}/ep{e}", np.asarray(v, np.float32).reshape(-1, 10, 3))
            for e, v in sorted(by_ep.items())]


def windows(eps, W, stride):
    """All [W,5,3] windows, per hand, raw."""
    out = []
    for _, tac in eps:
        for s in range(0, len(tac) - W + 1, stride):
            w = tac[s:s + W]
            out.append(w[:, :NF])
            out.append(w[:, NF:])
    return np.stack(out)                                   # [M,W,5,3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5_roots", nargs="+", required=True)
    ap.add_argument("--parquet_roots", nargs="*", default=[])
    ap.add_argument("--exclude_json", default="", help="JSON list of episode dir names to leave out.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--codebook_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min_lr_ratio", type=float, default=0.05)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--val_ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    exclude = set(json.load(open(args.exclude_json))) if args.exclude_json else set()
    eps = load_h5_roots(args.h5_roots, exclude)
    for r in args.parquet_roots:
        eps += load_parquet_root(r)
    n_frames = sum(len(t) for _, t in eps)
    print(f"{len(eps)} episodes, {n_frames} frames (excluded {len(exclude)} held-out)", flush=True)

    # Stats: q01/q99 over all training frames, per (finger, channel) of both hands -> [30].
    flat = np.concatenate([t.reshape(-1, 30) for _, t in eps])
    q01, q99 = np.quantile(flat, 0.01, axis=0), np.quantile(flat, 0.99, axis=0)
    mask = (q99 - q01) > 1e-6
    stats = {"tacf6_min": q01.tolist(), "tacf6_max": q99.tolist(), "tacf6_mask": mask.tolist()}
    print("q01", np.round(q01, 1).tolist(), "\nq99", np.round(q99, 1).tolist(), flush=True)

    # Episode-level val split, then windows.
    perm = rng.permutation(len(eps))
    n_val = max(1, int(len(eps) * args.val_ratio))
    val_eps = [eps[i] for i in perm[:n_val]]
    tr_eps = [eps[i] for i in perm[n_val:]]
    Xtr, Xva = windows(tr_eps, args.window, args.stride), windows(val_eps, args.window, args.stride)

    def norm(x, hand):
        lo = q01.reshape(10, 3)[hand * NF:(hand + 1) * NF]
        hi = q99.reshape(10, 3)[hand * NF:(hand + 1) * NF]
        m = mask.reshape(10, 3)[hand * NF:(hand + 1) * NF]
        return np.where(m, np.clip(2 * (x - lo) / (hi - lo + 1e-8) - 1, -1, 1), x)

    # windows() alternates left/right, so hand = index parity.
    def prep(X):
        hands = np.arange(len(X)) % 2
        Xn = np.empty_like(X)
        Xn[hands == 0] = norm(X[hands == 0], 0)
        Xn[hands == 1] = norm(X[hands == 1], 1)
        mag = np.linalg.norm(X.reshape(len(X), -1), axis=1)
        return torch.from_numpy(Xn.astype(np.float32)), torch.from_numpy(mag.astype(np.float32))

    Ttr, Mtr = prep(Xtr)
    Tva, Mva = prep(Xva)
    tau = float(np.median(Mtr.numpy()[Mtr.numpy() > 0])) if (Mtr > 0).any() else 1.0
    print(f"windows: train {len(Ttr)}  val {len(Tva)}  magnitude tau (median contact) {tau:.1f}", flush=True)

    cfg = TactileVQVAEConfig(window=args.window, in_channels=NF * D, codebook_size=args.codebook_size,
                             granularity="finger", n_fingers=NF, per_finger_dim=D, weight_tau=tau)
    dev = torch.device("cuda")
    model = TactileVQVAE(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(Ttr) / args.batch_size)
    total = args.epochs * steps_per_epoch
    print(f"{args.epochs} epochs x {steps_per_epoch} steps = {total}", flush=True)

    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        model.train()
        order = torch.randperm(len(Ttr))
        for b in range(steps_per_epoch):
            idx = order[b * args.batch_size:(b + 1) * args.batch_size]
            if step < args.warmup_steps:
                lr = args.lr * (step + 1) / args.warmup_steps
            else:
                p = (step - args.warmup_steps) / max(1, total - args.warmup_steps)
                lr = args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p)))
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(Ttr[idx].to(dev), Mtr[idx].to(dev))
            opt.zero_grad(set_to_none=True)
            out["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
        model.eval()
        with torch.no_grad():
            vr, used = [], set()
            for b in range(0, len(Tva), 1024):
                o = model(Tva[b:b + 1024].to(dev), Mva[b:b + 1024].to(dev))
                vr.append(o["recon_loss"].item())
                used |= set(o["indices"].flatten().tolist())
        print(f"ep {ep:2d} step {step} train recon {out['recon_loss'].item():.4f} vq {out['vq_loss'].item():.4f} "
              f"| val recon {np.mean(vr):.4f} codes used {len(used)}/{cfg.codebook_size} "
              f"| {time.time() - t0:.0f}s", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    blob = {"model_state": model.state_dict(), "config": cfg.to_dict(), "stats": stats,
            "step": step, "epoch": args.epochs,
            "data": {"episodes": [n for n, _ in eps], "val_episodes": [n for n, _ in val_eps],
                     "excluded": sorted(exclude)}}
    torch.save(blob, os.path.join(args.output_dir, "latest.pt"))
    print(f">>> saved {args.output_dir}/latest.pt", flush=True)


if __name__ == "__main__":
    main()
