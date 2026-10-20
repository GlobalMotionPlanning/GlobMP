#!/usr/bin/env python3
"""ACT baseline on the T-Rex release, with and without tactile.

Stands in for the paper's ViTacFormer row, described there as an "ACT-style visuo-tactile
imitation policy". ACT is the right vehicle here for three reasons the VLAs cannot match:

  * it has no max_state_dim/max_action_dim cap, so 58-D (or 118-D with tactile) needs no
    projection surgery -- it sizes from the data;
  * it trains from scratch on an ImageNet ResNet-18 backbone, so unlike every VLA on this
    box there is NO question of prior exposure to these episodes;
  * dim_model 512 fits comfortably on 2 GPUs.

Reads the ORIGINAL v3 release directly (lerobot 0.4.4 is a v3 reader; our GR00T roots are
v2 and it refuses them). Two dataset fixes are applied through a make_dataset hook:

  1. the release carries 23 video keys (3 RGB + 20 tactile streams). Decoding all of them
     costs 1.15 s/sample; restricting to the 3 RGB cameras costs 0.07 s -- a 16x speedup.
     video_keys is a derived property, so the restriction is applied to meta.info["features"].
  2. --tactile concatenates observation.tactile_force (60-D) onto observation.state (58-D),
     giving 118-D. This is exactly what the paper does for its pi-0.5 + tactile row:
     "we extend the original state input by concatenating single-step tactile observations".

  XB_DATASET=<data/task/lerobot> python finetune/train_lerobot.py --policy act --episodes <split/train.json> --out <ckpt>
  (normally called by finetune_{pi05,dp,act}.sh)
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path

# DATASET is overridable so the same trainer can fine-tune on a different embodiment. Today's
# XHand teleop is 38-D state (arm 7 + hand 12 per side) with 30-D tactile and a 4th camera,
# against T-Rex's 58-D / 60-D / 3 cameras -- so the camera list is read from the dataset's own
# info.json rather than hard-coded, and the state/action projections are rebuilt by the policy.
RELEASE = os.environ["XB_DATASET"]          # LeRobot v3 root (data/<task>/lerobot)
_info = json.load(open(os.path.join(RELEASE, "meta", "info.json")))
RGB_KEYS = [k for k in _info["features"] if k.startswith("observation.images.")
            and "tactile" not in k]
TACTILE_KEY = "observation.tactile_force"
STATE_KEY = "observation.state"


def install_pi05_load_hook():
    """Two fixes without which pi05 either dies or, worse, trains from random init.

    1. WEIGHTS. pi05_base ships a 32-D action head; our action space is 58-D, so
       action_in_proj.weight and action_out_proj.weight/.bias mismatch. PI05Policy overrides
       from_pretrained and calls load_state_dict(strict=True) inside a bare try/except that
       only PRINTS "Warning: Could not remap state dict keys" and returns the model -- so a
       single mismatched tensor silently leaves all 3.5 B parameters randomly initialized.
       We drop the mismatched keys (they keep their fresh init, the backbone transfers) and
       load non-strict, then assert that the load actually happened.
       The state side needs no surgery: pi05 TOKENIZES state rather than projecting it, so
       max_state_dim carries no weights.

    2. PROCESSORS. pi05_base's policy_preprocessor.json references 'relative_actions_processor',
       a step that does not exist in lerobot 0.4.4's registry, so loading the pretrained
       pipeline raises. We build the pipeline fresh instead -- which is what we want anyway:
       the checkpoint's normalization statistics belong to Physical Intelligence's robot, not
       to this dataset. Fresh construction uses OUR dataset stats.
    """
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    import lerobot.policies.factory as F
    import lerobot.scripts.lerobot_train as T
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    _orig_load = PI05Policy.load_state_dict

    def _filtered_load(self, state_dict, strict=True, assign=False):
        want = {k: v.shape for k, v in self.state_dict().items()}
        dropped = sorted(k for k, v in state_dict.items() if k in want and v.shape != want[k])
        sd = {k: v for k, v in state_dict.items() if k not in dropped}
        missing, unexpected = _orig_load(self, sd, strict=False, assign=assign)
        loaded = len(sd) - len(unexpected)
        print(f"[pi05] loaded {loaded}/{len(state_dict)} checkpoint tensors")
        print(f"[pi05] reinit on shape mismatch ({len(dropped)}): {dropped}")
        print(f"[pi05] missing={len(missing)} unexpected={len(unexpected)}")
        if unexpected:
            print(f"[pi05] UNEXPECTED (no home in the model): {unexpected[:10]}")
        if missing:
            print(f"[pi05] MISSING (left at init): {missing[:10]}")
        # A silent random init is the failure mode this whole hook exists to prevent.
        if loaded < 0.9 * len(state_dict):
            raise RuntimeError(f"[pi05] only {loaded}/{len(state_dict)} tensors loaded -- "
                               f"refusing to train from a mostly-random 3.5 B model")
        return missing, unexpected

    PI05Policy.load_state_dict = _filtered_load

    _orig_mpp = F.make_pre_post_processors

    def _fresh_processors(policy_cfg, pretrained_path=None, **kw):
        if isinstance(policy_cfg, PI05Config) and pretrained_path:
            print("[pi05] building pre/post processors from OUR dataset stats, not the "
                  "checkpoint's pipeline (its 'relative_actions_processor' step, which is "
                  "disabled anyway, does not exist in lerobot 0.4.4)")
            pretrained_path = None
        return _orig_mpp(policy_cfg, pretrained_path, **kw)

    # lerobot_train does `from ...factory import make_pre_post_processors`, so the name it
    # calls is its own module global -- patching only the factory module would miss it.
    F.make_pre_post_processors = _fresh_processors
    T.make_pre_post_processors = _fresh_processors


def install_dataset_hook(episodes, tactile: bool):
    """Patch make_dataset to restrict video keys and optionally fold tactile into state."""
    import lerobot.scripts.lerobot_train as T
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.transforms import ImageTransforms

    # Strip the 20 unused video keys on EVERY metadata instance, before any LeRobotDataset
    # is built. This is not cosmetic. The release declares 23 video streams but ships only
    # 13 dirs (3 RGB + 10 deform); the 10 tactile_*_raw_* streams are absent. With them still
    # declared, LeRobotDataset.__init__ -> _check_cached_episodes_sufficient() finds a missing
    # video file, raises FileNotFoundError, and falls into download() -> snapshot_download(),
    # which burns ~47 min of pure CPU in fnmatch filtering a 1067x23 allow_patterns list
    # against the repo listing -- every process, every run, every DDP rank. Stripping the keys
    # on the metadata makes the check pass on the 3 RGB streams that do exist, so the cached
    # parquet is used directly and __init__ returns in seconds. Same data either way.
    _orig_meta_init = LeRobotDatasetMetadata.__init__

    def _meta_init(self, *args, **kwargs):
        _orig_meta_init(self, *args, **kwargs)
        for k in [k for k in self.info["features"]
                  if k.startswith("observation.images.") and k not in RGB_KEYS]:
            self.info["features"].pop(k)

    LeRobotDatasetMetadata.__init__ = _meta_init

    # lerobot 0.4.4 bug: EpisodeAwareSampler builds its index list from the METADATA's
    # dataset_from/to_index, which are absolute frame indices into the full 5.47 M-row
    # dataset. But a dataset built with episodes=<subset> only holds those episodes'
    # rows, and __getitem__ indexes that filtered table positionally -- so every sampled
    # index is out of range ("Invalid key: 3471090 is out of bounds for size 977409").
    # Only policies exposing drop_n_last_frames use this sampler, which is why ACT and
    # pi05 were unaffected and diffusion died instantly. Map the sampler's indices through
    # the dataset's own absolute->relative table.
    _built = {}

    _OrigSampler = T.EpisodeAwareSampler

    class _RelativeSampler(_OrigSampler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            ds = _built.get("ds")
            amap = getattr(ds, "_absolute_to_relative_idx", None) if ds is not None else None
            if amap is not None:
                before = len(self.indices)
                self.indices = [amap[i] for i in self.indices if i in amap]
                print(f"[hook] EpisodeAwareSampler: {before} absolute -> {len(self.indices)} relative indices")

    T.EpisodeAwareSampler = _RelativeSampler

    def patched(cfg):
        # Build the dataset ourselves rather than calling lerobot's make_dataset, solely so we
        # can pass download_videos=False. With a repo_id set, lerobot treats the hub as
        # authoritative and snapshot-downloads the WHOLE repo (4,386 files / ~100 GB of raw
        # tactile streams we do not use) even though every file we need is already local.
        tf = ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
        meta = LeRobotDatasetMetadata(cfg.dataset.repo_id, root=cfg.dataset.root,
                                      revision=cfg.dataset.revision)
        # Strip the 20 tactile video streams on the METADATA first. delta_timestamps is
        # resolved from meta, so a policy with n_obs_steps>1 (diffusion) would otherwise ask
        # the dataset for columns we are about to remove.
        for k in [k for k in meta.info["features"]
                  if k.startswith("observation.images.") and k not in RGB_KEYS]:
            meta.info["features"].pop(k)
        dts = resolve_delta_timestamps(cfg.policy, meta)
        h = getattr(cfg.policy, "tactile_history", 1)     # tactile_vla only: last h tactile frames
        if h > 1:
            dts = {**(dts or {}), TACTILE_KEY: [(i - h + 1) / meta.fps for i in range(h)]}
        ds = LeRobotDataset(
            cfg.dataset.repo_id, root=cfg.dataset.root, episodes=cfg.dataset.episodes,
            delta_timestamps=dts,
            image_transforms=tf, revision=cfg.dataset.revision,
            video_backend=cfg.dataset.video_backend, tolerance_s=cfg.tolerance_s,
            download_videos=False,
        )
        _built["ds"] = ds
        feats = ds.meta.info["features"]
        dropped = [k for k in feats if k.startswith("observation.images.") and k not in RGB_KEYS]
        for k in dropped:
            feats.pop(k)
        print(f"[hook] video keys {len(dropped)+len(RGB_KEYS)} -> {len(ds.meta.video_keys)}")

        if tactile:
            feats[STATE_KEY] = dict(feats[STATE_KEY])
            _sd = _info["features"][STATE_KEY]["shape"][0]
            _td = _info["features"][TACTILE_KEY]["shape"][0]
            feats[STATE_KEY]["shape"] = [_sd + _td]
            # Normalization stats must grow with the input, or Normalize() broadcasts 58
            # against 118 and dies. Concatenate the tactile stats onto the state stats.
            import numpy as np
            st, tc = ds.meta.stats[STATE_KEY], ds.meta.stats[TACTILE_KEY]
            ds.meta.stats[STATE_KEY] = {
                k: np.concatenate([np.asarray(st[k]).reshape(-1),
                                   np.asarray(tc[k]).reshape(-1)])
                # q* as well as the moments: pi05 normalizes STATE/ACTION with QUANTILES,
                # so a fold that carried only mean/std/min/max would leave q01/q99 at 58-D.
                for k in ("mean", "std", "min", "max", "q01", "q10", "q50", "q90", "q99")
                if k in st and k in tc
            }
            print(f"[hook] state stats {len(np.asarray(st['mean']).reshape(-1))} -> "
                  f"{len(ds.meta.stats[STATE_KEY]['mean'])}")
            inner = ds.__class__.__getitem__

            def with_tactile(self, idx):
                s = inner(self, idx)
                if TACTILE_KEY in s:
                    s[STATE_KEY] = torch.cat([s[STATE_KEY], s[TACTILE_KEY].to(s[STATE_KEY].dtype)], dim=-1)
                return s

            ds.__class__.__getitem__ = with_tactile
            print(f"[hook] state {_sd} + tactile {_td} -> {feats[STATE_KEY]['shape'][0]}")
        return ds

    T.make_dataset = patched
    return T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tactile", action="store_true")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--save-freq", type=int, default=10_000)
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--policy", choices=["act","diffusion","pi05","tvla"], default="act",
                    help="act -> ~ViTacFormer row; diffusion -> RDP row; pi05 -> the pi-0.5 rows; "
                         "tvla -> Tactile-VLA row (pi05 + MLP tactile prefix token, finetune/tactile_vla.py)")
    ap.add_argument("--tactile-history", type=int, default=None, help="tvla: tactile frames per token")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1000,
                    help="a single run is not evidence; report across seeds")
    a = ap.parse_args()

    episodes = json.load(open(a.episodes))
    T = install_dataset_hook(episodes, a.tactile)

    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.default import DatasetConfig, WandBConfig
    if a.policy == "act":
        from lerobot.policies.act.configuration_act import ACTConfig
        policy = ACTConfig(chunk_size=a.chunk_size, n_action_steps=a.chunk_size)
    elif a.policy in ("pi05", "tvla"):
        # The paper: "We reproduce pi-0.5 using the official OpenPI codebase and initialize all
        # policies from the released pi-0.5 pretrained checkpoint", with "dual-arm 2x7 joint
        # control and 22-DoF dexterous hand joint control" -- i.e. exactly our 58-D space.
        # pi05 DOES cap dims at 32, unlike ACT/diffusion, so raise them. The state/action
        # projections are reinitialized as a result; the backbone transfers.
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        _sd = _info["features"][STATE_KEY]["shape"][0]
        _ad = _info["features"]["action"]["shape"][0]
        _td = _info["features"][TACTILE_KEY]["shape"][0]
        width = (_sd + _td) if a.tactile else _sd
        Cfg = PI05Config
        if a.policy == "tvla":            # tactile goes through the MLP token, never into state
            assert not a.tactile, "tvla takes tactile through its encoder; do not pass --tactile"
            from tactile_vla import TactileVLAConfig as Cfg
        policy = Cfg(chunk_size=a.chunk_size, n_action_steps=a.chunk_size,
                     max_state_dim=width, max_action_dim=_ad)
        if a.policy == "tvla" and a.tactile_history:
            policy.tactile_history = a.tactile_history
        policy.pretrained_path = os.environ["XB_PI05_BASE"]
        # 3.5 B params: fp32 weights + grads + AdamW moments is ~56 GB before activations.
        # Checkpointing the three towers is what keeps a useful batch size on one H100.
        policy.gradient_checkpointing = True
        install_pi05_load_hook()
    else:
        # Neither ACT nor diffusion caps state/action dim, so 58-D and 118-D both work as-is.
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        # n_obs_steps=1: the default 2 asks for a frame BEFORE the current one, which indexes
        # out of bounds at episode starts on this dataset.
        policy = DiffusionConfig(horizon=a.chunk_size, n_action_steps=a.chunk_size // 2,
                                 n_obs_steps=1)
    policy.push_to_hub = False          # local-only run; avoids the hub repo_id requirement
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="local/xhand", root=RELEASE,
                              episodes=episodes, video_backend="pyav"),
        policy=policy,
        output_dir=Path(a.out),
        job_name=Path(a.out).name,
        steps=a.steps, batch_size=a.batch_size, save_freq=a.save_freq,
        num_workers=a.num_workers, log_freq=100, seed=a.seed,
        wandb=WandBConfig(enable=False),
    )
    cfg.validate()
    print(f"[train_act] episodes={len(episodes)} tactile={a.tactile} steps={a.steps} "
          f"batch={a.batch_size} out={a.out}")
    T.train.__wrapped__(cfg)


if __name__ == "__main__":
    main()
