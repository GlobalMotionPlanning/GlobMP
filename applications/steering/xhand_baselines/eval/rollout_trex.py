#!/usr/bin/env python3
"""Open-loop held-out eval of a T-Rex XHand checkpoint (42-D eef+hand), same protocol as the
baseline benches: at frames 0, 16, 32, ... the model sees the real observation and predicts a
16-step chunk; all 16 steps are kept and compared with the dataset's own target chunk.

Runs the repo's inference path (scripts/test.py: model_load + CascadedServer, slow_and_fast)
in-process instead of over ZMQ. XHand has 3 tactile channels per finger, so the one hard-coded
reshape(-1, 6) in test.py is replaced.

Errors are computed in the chunk-start frame. Position and geodesic rotation error are invariant to
that base, so they equal the error of the absolute poses.

Writes <outdir>/ep<local_idx>.npz with pred/gt [n, 42] and the chunk start of every frame.
"""
import argparse, io, json, os, sys, time
import numpy as np, torch
from PIL import Image

REPO = os.environ["TREX_DIR"]                                         # T-Rex + trex_xhand.patch
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/scripts")
import test as T                                                      # noqa: E402
from utils.lerobot_common import (ACTION_CHUNK, ACTION_DIM, KEY_HEAD, KEY_WRIST_R,  # noqa: E402
                                  KEY_WRIST_L, KEY_ACTION, KEY_TACF6)

F6 = 3


def _encode_tactile_f6_xhand(tactile_f6_input, statistic, device):
    if tactile_f6_input is None:
        return None
    arr = np.asarray(tactile_f6_input, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[-1]
    norm = T._normalize(arr.reshape(-1), statistic["tacf6_mask"], statistic["tacf6_min"], statistic["tacf6_max"])
    return torch.tensor(norm.reshape(-1, F6), dtype=torch.bfloat16).unsqueeze(0).to(device)


T._encode_tactile_f6 = _encode_tactile_f6_xhand


def to_png(chw):
    a = (chw.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    b = io.BytesIO(); Image.fromarray(a).save(b, format="PNG"); return b.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint-<epoch>-<step> dir")
    ap.add_argument("--dataset", required=True, help="T-Rex 42-D held-out LeRobot dataset")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tactile", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    ta = json.load(open(f"{a.ckpt}/training_args.json"))
    args = argparse.Namespace(
        checkpoint_path=a.ckpt, base_model_path="", stats_path="", dataset_name="",
        action_dim=ACTION_DIM, action_chunk=ACTION_CHUNK, use_robot_state=0, use_tactile_deform=0,
        use_tactile_vec=a.tactile, tactile_intermediate_size=ta.get("tactile_intermediate_size", 0),
        n_flare_tokens_per_frame=0, n_flare_steps=0, cuda="0", image_size=[384, 288],
        cascaded_total_steps=10, cascaded_split_step=6,
        disable_tactile=0 if a.tactile else 1,   # no-tactile run: training fits only the action expert (full flow)
        use_tactile_code=0, vqvae_codebook_size=64, vqvae_ckpt="", use_tactile_vqvae=0, tacf6_dim=F6)
    assert bool(ta.get("use_tactile_vec", 0)) == bool(a.tactile), "checkpoint/--tactile mismatch"
    model, processor, statistic = T.model_load(args)
    server = T.CascadedServer(args, model, processor, statistic)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id="local", root=a.dataset, video_backend="pyav")
    task = ds.meta.tasks.index[0] if hasattr(ds.meta.tasks, "index") else list(ds.meta.tasks.values())[0]
    ep_idx = np.asarray(ds.hf_dataset["episode_index"]).reshape(-1)
    tac_all = np.asarray(ds.hf_dataset.with_format("numpy")[KEY_TACF6], np.float32).reshape(-1, 10, F6)
    for e in sorted(set(ep_idx.tolist())):
        rows = np.where(ep_idx == e)[0]; n = len(rows)
        gt = np.zeros((n, ACTION_DIM), np.float32); pr = np.zeros((n, ACTION_DIM), np.float32)
        start = np.zeros(n, np.int64); tac_hist = []
        t0 = time.time()
        for t in range(n):
            tac_hist.append(tac_all[rows[t]]); tac_hist = tac_hist[-16:]      # VQ-VAE window W=16, frame rate
            if t % ACTION_CHUNK:
                continue
            s = ds[int(rows[t])]
            payload = {"image_head": to_png(s[KEY_HEAD]), "image_wrist_right": to_png(s[KEY_WRIST_R]),
                       "image_wrist_left": to_png(s[KEY_WRIST_L]), "task_description": task}
            if a.tactile:
                payload["tactile_f6"] = np.stack(tac_hist, 0)              # dense [<=16, 10, 3] window
            out = np.asarray(server.predict("slow_and_fast", payload)["actions"], np.float32)
            chunk_gt = np.asarray(s[KEY_ACTION], np.float32)
            k = min(ACTION_CHUNK, n - t)
            pr[t:t + k] = out[:k]; gt[t:t + k] = chunk_gt[:k]; start[t:t + k] = t
        np.savez(f"{a.outdir}/ep{int(e):03d}.npz", pred=pr, gt=gt, chunk_start=start, episode=int(e))
        print(f"ep {e}: {n} frames, {time.time()-t0:.0f}s", flush=True)
    print("ROLLOUT DONE", flush=True)


if __name__ == "__main__":
    main()
