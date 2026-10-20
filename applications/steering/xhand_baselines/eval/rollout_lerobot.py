#!/usr/bin/env python3
"""Open-loop chunked rollout of a lerobot policy (ACT / Diffusion / pi-0.5 / Tactile-VLA) on held-out episodes,
in the same npz schema as rollout_xhand.py so metrics.py and the video pipeline are shared.

Observation format is training's, not a re-implementation: train_lerobot.py's install_dataset_hook
is imported and used to build the dataset, so the video-key stripping, the tactile fold into
observation.state (38 -> 68) and the quantile stats are byte-for-byte what the policy saw. The
saved pre/post-processors are loaded from the checkpoint (they carry OUR dataset stats).

Protocol as everywhere in this bench: at every CHUNK-th step the policy sees the REAL observation
and predicts its action chunk; the first CHUNK actions are kept. GT is the recorded action.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

ap = argparse.ArgumentParser()
ap.add_argument("--policy", choices=["act", "diffusion", "pi05", "tvla"], required=True)
ap.add_argument("--ckpt", required=True, help=".../checkpoints/last/pretrained_model")
ap.add_argument("--dataset", required=True, help="data/<task>/lerobot")
ap.add_argument("--episodes", required=True); ap.add_argument("--outdir", required=True)
ap.add_argument("--tactile", action="store_true"); ap.add_argument("--chunk", type=int, default=16)
ap.add_argument("--device", default="cuda:0"); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

os.environ["XB_DATASET"] = a.dataset
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "finetune"))
import train_lerobot as TA                               # reads XB_DATASET at import
if a.policy == "tvla": import tactile_vla               # registers policy type "tactile_vla"; tactile history via the hook
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors, get_policy_class

eps = json.load(open(a.episodes)) if a.episodes.endswith(".json") else [int(x) for x in a.episodes.split(",")]
if a.limit: eps = eps[:a.limit]
Path(a.outdir).mkdir(parents=True, exist_ok=True)

policy_cfg = PreTrainedConfig.from_pretrained(a.ckpt)
policy_cfg.pretrained_path = a.ckpt
policy = get_policy_class(policy_cfg.type).from_pretrained(a.ckpt, config=policy_cfg)
policy.to(a.device).eval()
pre, post = make_pre_post_processors(policy_cfg, pretrained_path=a.ckpt)
print(f"[{a.policy}] loaded {a.ckpt}  state_dim={policy_cfg.input_features['observation.state'].shape}  "
      f"chunk={getattr(policy_cfg,'chunk_size',getattr(policy_cfg,'horizon',None))} n_action_steps={policy_cfg.n_action_steps}", flush=True)

T = TA.install_dataset_hook(eps, a.tactile)
# ONE dataset over every held-out episode, then per-episode row ranges. The tactile fold-in hook
# patches __getitem__ on the CLASS, so a fresh dataset per episode would wrap it again and stack
# tactile twice (38+30+30 = 98-D, which the 68-D normalizer rejects). One build, one fold.
cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="local/xhand", root=a.dataset, episodes=eps,
                                                video_backend="pyav"),
                          policy=policy_cfg, output_dir=Path("/tmp/_unused"), job_name="rollout",
                          wandb=WandBConfig(enable=False))
ds = T.make_dataset(cfg)
ep_col = np.asarray(ds.hf_dataset["episode_index"])
print(f"[rollout] one dataset, {len(ds)} rows over {len(eps)} episodes; state dim seen by policy = "
      f"{tuple(ds[0]['observation.state'].shape)}", flush=True)
for i in eps:
    out = f"{a.outdir}/ep{i:03d}.npz"
    if os.path.exists(out): print(f"ep {i}: exists, skip", flush=True); continue
    rows = np.nonzero(ep_col == i)[0]; n = len(rows)
    assert n > 0 and rows[-1] - rows[0] == n - 1, f"episode {i} rows not contiguous"
    gt = np.stack(pd.read_parquet(f"{a.dataset}/data/chunk-000/file-{i:03d}.parquet")["action"].to_numpy()).astype(np.float32)[:n]
    pred = np.zeros((n, gt.shape[1]), np.float32); policy.reset()
    for s in range(0, n, a.chunk):
        it = ds[int(rows[s])]
        batch = {k: v.unsqueeze(0) for k, v in it.items() if torch.is_tensor(v) and k.startswith("observation.")}
        batch["task"] = [it.get("task", "")]
        with torch.no_grad():
            obs = pre(batch)
            if a.policy == "diffusion":
                # generate_actions wants state (B, s, d) and images (B, s, n_cam, C, H, W).
                # The dataset already gives state a leading n_obs_steps dim but gives images
                # none, so the two need different handling. Going through predict_action_chunk
                # instead would stack its observation queue on top and add a SECOND time dim to
                # state (4-D vs 3-D image features); the queue exists for stepping an env one
                # action at a time, which is not what an open-loop chunk rollout does.
                from lerobot.utils.constants import OBS_STATE, OBS_IMAGES
                imgs = torch.stack([obs[k] for k in policy.config.image_features], dim=1).unsqueeze(1)
                act = policy.diffusion.generate_actions({OBS_STATE: obs[OBS_STATE], OBS_IMAGES: imgs})
            else:
                act = policy.predict_action_chunk(obs)     # (1, H, A) normalized
            act = post(act)
        ch = act[0].float().cpu().numpy(); k = min(a.chunk, n - s, len(ch))
        pred[s:s + k] = ch[:k, :gt.shape[1]]
    np.savez(out, gt_joints=gt, pred_joints=pred, episode=i, task=np.array(str(batch["task"][0])))
    arm = np.r_[0:7, 19:26]; hnd = np.r_[7:19, 26:38]
    print(f"ep {i:3d}  n={n:5d}  arm |err| med={np.median(np.abs(pred[:,arm]-gt[:,arm])):.4f}  "
          f"hand med={np.median(np.abs(pred[:,hnd]-gt[:,hnd])):.4f} rad", flush=True)
print("done")
