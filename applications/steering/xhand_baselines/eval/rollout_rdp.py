#!/usr/bin/env python3
"""Open-loop chunked rollout of Reactive Diffusion Policy (Asymmetric Tokenizer + Latent Diffusion Policy)
on held-out episodes, in the npz schema of rollout_gr00t.py so metrics.py scores it unchanged.

Protocol as everywhere in this bench: at every CHUNK-th step s the slow policy (LDP) sees the REAL recorded
observation (3 cameras, state and tactile at s-2 and s) and samples a latent action chunk. The fast part then
runs as RDP's action thread does at every control step: step s+k is decoded by the AT's GRU decoder from that
latent and the RECORDED tactile stream up to s+k (window from s-3, as in training). Kept: steps s..s+CHUNK-1.
Arm joints are predicted relative to the state at s (as in training) and made absolute with the recorded state.
GT is the recorded action. Needs RDP_DIR and RDP_SITE on PYTHONPATH.
"""
import argparse, contextlib, io, json, os
from pathlib import Path
import numpy as np, torch, dill, hydra
from omegaconf import OmegaConf

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="stage-2 (LDP) checkpoint, .../checkpoints/latest.ckpt")
ap.add_argument("--at-ckpt", default=None, help="stage-1 (AT) checkpoint; default: the one the LDP was trained with")
ap.add_argument("--dataset", required=True, help="data/<task>/lerobot")
ap.add_argument("--episodes", required=True); ap.add_argument("--outdir", required=True)
ap.add_argument("--chunk", type=int, default=16); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--max-chunks", type=int, default=0, help="test only")
ap.add_argument("--seed", type=int, default=1000)
a = ap.parse_args()

OmegaConf.register_new_resolver("eval", eval, replace=True)
payload = torch.load(open(a.ckpt, "rb"), pickle_module=dill, map_location="cpu", weights_only=False)
cfg = payload["cfg"]
cfg.policy.at.load_dir = a.at_ckpt or cfg.at_load_dir
policy = hydra.utils.instantiate(cfg.policy)
policy.load_state_dict(payload["state_dicts"]["ema_model" if cfg.training.use_ema else "model"])
policy.set_normalizer(policy.normalizer)          # the AT is not an nn.Module: hand it the loaded normalizer
policy.to(torch.device(a.device)).eval()
R = cfg.task.dataset.obs_temporal_downsample_ratio
To = cfg.n_obs_steps * R                           # window offset: action index To-1 is the current step
print(f"[rdp] loaded {a.ckpt} (ema={cfg.training.use_ema}) at={cfg.policy.at.load_dir}  horizon={cfg.horizon} "
      f"obs_steps={cfg.n_obs_steps}x{R} ddim_steps={policy.num_inference_steps}", flush=True)

eps = json.load(open(a.episodes)) if a.episodes.endswith(".json") else [int(x) for x in a.episodes.split(",")]
if a.limit: eps = eps[:a.limit]
dcfg = OmegaConf.to_container(cfg.task.dataset, resolve=True)
for k in ("at", "use_latent_action_before_vq"): dcfg.pop(k, None)
dcfg.update(_target_="reactive_diffusion_policy.dataset.xhand_lerobot_dataset.XHandLeRobotDataset",
            dataset_path=a.dataset, episodes=eps)
ds = hydra.utils.instantiate(dcfg)                 # held-out episodes; normalization comes from the checkpoint
Path(a.outdir).mkdir(parents=True, exist_ok=True)
dev = lambda d: {k: v.unsqueeze(0).to(a.device) for k, v in d.items()}
quiet = lambda: contextlib.redirect_stdout(io.StringIO())   # predict_from_latent_action prints per call

for e in eps:
    out = f"{a.outdir}/ep{e:03d}.npz"
    if os.path.exists(out): print(f"ep {e}: exists, skip", flush=True); continue
    d = ds.data[e]; gt = d["action"]; n = len(gt); pred = np.zeros_like(gt); starts = []
    torch.manual_seed(a.seed + e)
    for s in range(0, n, a.chunk):
        if a.max_chunks and len(starts) == a.max_chunks: break
        smp = ds.sample(e, s); obs, ext = dev(smp["obs"]), dev(smp["extended_obs"])
        with torch.no_grad():
            lat = policy.predict_action(obs, dataset_obs_temporal_downsample_ratio=R,
                                        return_latent_action=True)["action"][:, 0]        # slow: latent chunk
            for k in range(min(a.chunk, n - s)):                                         # fast: every step
                L = To + k
                with quiet():
                    act = policy.predict_from_latent_action(lat, {kk: v[:, :L] for kk, v in ext.items()}, L, R)["action"]
                pred[s + k] = act[0, -1].float().cpu().numpy()
        pred[s:s + k + 1, ds.rel] += d["observation.state"][s, ds.rel]               # relative arm dims -> absolute
        starts.append(s)
    if a.max_chunks: n = starts[-1] + min(a.chunk, n - starts[-1]); gt, pred = gt[:n], pred[:n]
    np.savez(out, gt_joints=gt, pred_joints=pred, raw_joints=d["observation.state"][:n],
             tactile=d["observation.tactile_force"][:n], chunk_starts=np.array(starts), task=np.array(ds.task[e]), episode=e)
    arm = np.r_[0:7, 19:26]; hnd = np.r_[7:19, 26:38]
    print(f"ep {e:3d}  n={n:5d}  arm |err| med={np.median(np.abs(pred[:, arm] - gt[:, arm])):.4f}  "
          f"hand med={np.median(np.abs(pred[:, hnd] - gt[:, hnd])):.4f} rad", flush=True)
print("done")
