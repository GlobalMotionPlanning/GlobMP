"""XHand bimanual teleop (LeRobot v3) for the Asymmetric Tokenizer and the Latent Diffusion Policy.

Reads the LeRobot root in place: parquet for state / action / tactile, H.264 videos decoded on the fly.
Samples follow RealImageTactileDataset exactly: for every frame t of a training episode, a window of
`horizon` steps starting at t-(n_obs_steps-1) (edge-replicated at episode ends, like SequenceSampler);
obs = the first n_obs_steps of the window, subsampled by obs_temporal_downsample_ratio; extended_obs =
the whole window (the fast tactile stream the AT decoder reads at every action step).
relative_action_dims (the arm joints) are taken relative to the state at t, in the action and in the obs
state (RDP's relative_action + relative_tcp_obs_for_relative_action); the other dims stay absolute.
Normalization (limits -> [-1, 1], as RDP) is fitted on the selected (training) episodes only.

shape_meta keys: rgb obs <cam> -> observation.images.<cam>; low_dim 'state' -> observation.state (38),
'tactile' -> observation.tactile_force (30).
"""
import json, os
from typing import Dict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits
from reactive_diffusion_policy.dataset.base_dataset import BaseImageDataset
from reactive_diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from reactive_diffusion_policy.common.normalize_util import get_image_range_normalizer

LOWDIM_SRC = {'state': 'observation.state', 'tactile': 'observation.tactile_force'}


class XHandLeRobotDataset(BaseImageDataset):
    def __init__(self,
                 shape_meta: dict,
                 dataset_path: str,
                 episodes,                       # list of episode_index or a .json file holding one
                 horizon=32,
                 n_obs_steps=4,                  # dataset obs steps (= policy n_obs_steps * downsample ratio)
                 obs_temporal_downsample_ratio=2,
                 n_latency_steps=0,
                 relative_action_dims=None,      # action dims made relative to the current state (RDP: TCP pose)
                 pad_before=None, pad_after=None,  # implied by the window (kept for config compatibility)
                 seed=42, val_ratio=0.0,
                 zero_tactile=False,             # ablation: tactile inputs replaced by zeros (architecture unchanged)
                 **unused):
        assert n_latency_steps == 0 and val_ratio == 0.0
        if isinstance(episodes, str):
            episodes = json.load(open(episodes))
        self.root, self.episodes = dataset_path, [int(e) for e in episodes]
        self.shape_meta, self.horizon = shape_meta, horizon
        self.n_obs_steps, self.obs_downsample_ratio = n_obs_steps, obs_temporal_downsample_ratio
        self.rel = np.array(list(relative_action_dims or []), dtype=int)
        obs_meta, ext_meta = shape_meta['obs'], shape_meta.get('extended_obs', {})
        self.rgb_keys = [k for k, v in obs_meta.items() if v.get('type', 'low_dim') == 'rgb']
        self.lowdim_keys = [k for k, v in obs_meta.items() if v.get('type', 'low_dim') == 'low_dim']
        self.extended_lowdim_keys = list(ext_meta.keys())

        info = json.load(open(os.path.join(dataset_path, 'meta', 'info.json')))
        self.fps = info['fps']
        meta = pd.concat([pd.read_parquet(os.path.join(dataset_path, 'meta', 'episodes', c, f))
                          for c in sorted(os.listdir(os.path.join(dataset_path, 'meta', 'episodes')))
                          for f in sorted(os.listdir(os.path.join(dataset_path, 'meta', 'episodes', c)))])
        meta = meta.set_index('episode_index')
        tasks = pd.read_parquet(os.path.join(dataset_path, 'meta', 'tasks.parquet'))
        self.task_names = {int(i): str(t) for t, i in zip(tasks.index, tasks['task_index'])}

        cols = ['action', 'task_index'] + sorted(set(LOWDIM_SRC[k] for k in self.lowdim_keys + self.extended_lowdim_keys)
                                                  | {'observation.state'})
        self.data, self.video, self.task = {}, {}, {}
        for e in self.episodes:
            m = meta.loc[e]
            df = pd.read_parquet(os.path.join(dataset_path, info['data_path'].format(
                chunk_index=int(m['data/chunk_index']), file_index=int(m['data/file_index']))),
                columns=cols + ['episode_index', 'frame_index'])
            df = df[df['episode_index'] == e].sort_values('frame_index')
            assert len(df) == int(m['length']), f'episode {e}: {len(df)} rows != length {m["length"]}'
            self.data[e] = {c: np.stack(df[c].to_numpy()).astype(np.float32) for c in cols if c != 'task_index'}
            self.task[e] = self.task_names[int(df['task_index'].iloc[0])]
            self.video[e] = {k: (os.path.join(dataset_path, info['video_path'].format(
                video_key=f'observation.images.{k}',
                chunk_index=int(m[f'videos/observation.images.{k}/chunk_index']),
                file_index=int(m[f'videos/observation.images.{k}/file_index']))),
                int(round(m[f'videos/observation.images.{k}/from_timestamp'] * self.fps))) for k in self.rgb_keys}
        self.index = [(e, t) for e in self.episodes for t in range(len(self.data[e]['action']))]
        self.src = dict(LOWDIM_SRC)
        if zero_tactile and 'observation.tactile_force' in self.data[self.episodes[0]]:
            for e in self.episodes:
                self.data[e]['observation.tactile_force_zero'] = np.zeros_like(self.data[e]['observation.tactile_force'])
            self.src['tactile'] = 'observation.tactile_force_zero'

    # ---- RDP dataset API ----
    def __len__(self):
        return len(self.index)

    def get_validation_dataset(self):
        val = XHandLeRobotDataset.__new__(XHandLeRobotDataset)
        val.__dict__.update(self.__dict__)
        val.index = []
        return val

    def _all(self, src):
        return np.concatenate([self.data[e][src] for e in self.episodes])

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self._all('action'))

    def all_windows(self, key):
        """(N*T, D) over every training window: action / obs as the sampler returns them (relative if set)"""
        i = {'action': 1, 'state': 0}[key]
        return np.concatenate([self.lowdim(e, t)[i] for e, t in self.index])

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        rel = len(self.rel) > 0   # as RDP: fit relative action / obs on the relative values
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(self.all_windows('action') if rel else self._all('action'))
        for k in set(self.lowdim_keys + self.extended_lowdim_keys):
            normalizer[k] = SingleFieldLinearNormalizer.create_fit(
                self.all_windows('state') if rel and k == 'state' else self._all(self.src[k]))
        for k in self.rgb_keys:
            normalizer[k] = get_image_range_normalizer()
        return normalizer

    # ---- sampling ----
    def window(self, e, t, length=None):
        """frame indices t-(n_obs_steps-1) .. t-(n_obs_steps-1)+length-1, clamped to the episode"""
        n = len(self.data[e]['action'])
        start = t - (self.n_obs_steps - 1)
        return np.clip(np.arange(start, start + (length or self.horizon)), 0, n - 1)

    def obs_steps(self, e, t):
        """the obs frames the policy sees: RDP's [T_slice][::-ratio][::-1] of the window"""
        return self.window(e, t)[:self.n_obs_steps][::-self.obs_downsample_ratio][::-1]

    def load_rgb(self, e, key, frames):
        from torchcodec.decoders import VideoDecoder
        path, off = self.video[e][key]
        uniq, inv = np.unique(frames, return_inverse=True)
        img = VideoDecoder(path, seek_mode='approximate', num_ffmpeg_threads=1).get_frames_at(
            [int(i) + off for i in uniq]).data[torch.from_numpy(inv)]
        h, w = self.shape_meta['obs'][key]['shape'][1:]
        img = img.float() / 255.
        if img.shape[-2:] != (h, w):
            img = F.interpolate(img, size=(h, w), mode='bilinear', antialias=True, align_corners=False).clamp_(0, 1)
        return img                                                    # T,C,H,W float in [0,1]

    def lowdim(self, e, t):
        """obs state (obs steps), action window; relative dims as RDP's relative_action with
        relative_tcp_obs_for_relative_action: action - state[t], obs state - state[t]"""
        d, obs_t = self.data[e], self.obs_steps(e, t)
        state, action = d['observation.state'][obs_t].copy(), d['action'][self.window(e, t)].copy()
        if len(self.rel):
            base = d['observation.state'][t, self.rel]
            action[:, self.rel] -= base
            state[:, self.rel] -= base
        return state, action

    def sample(self, e, t) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        d, win, obs_t = self.data[e], self.window(e, t), self.obs_steps(e, t)
        state, action = self.lowdim(e, t)
        obs = {k: self.load_rgb(e, k, obs_t) for k in self.rgb_keys}
        obs.update({k: torch.from_numpy(state if k == 'state' else d[self.src[k]][obs_t]) for k in self.lowdim_keys})
        ext = {k: torch.from_numpy(d[self.src[k]][win]) for k in self.extended_lowdim_keys}
        return {'obs': obs, 'action': torch.from_numpy(action), 'extended_obs': ext}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.sample(*self.index[idx])


class XHandLeRobotLatentDataset(XHandLeRobotDataset):
    """Adds the latent-action normalizer (RealImageTactileLatentDiffusionDataset), computed from the action
    windows alone (no image decoding), batched through the frozen AT encoder."""
    def __init__(self, at, use_latent_action_before_vq: bool, **kwargs):
        super().__init__(**kwargs)
        self.at, self.use_latent_action_before_vq = at, use_latent_action_before_vq
        self.at.eval()

    @torch.no_grad()
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = super().get_normalizer(**kwargs)
        at, out = self.at, []
        for i in range(0, len(self.index), 1024):
            act = np.stack([self.lowdim(e, t)[1] for e, t in self.index[i:i + 1024]])
            act = normalizer['action'].normalize(torch.from_numpy(act)).to(at.device)
            lat = at.encoder(at.preprocess(act / at.act_scale))
            if at.use_vq:
                if not self.use_latent_action_before_vq:
                    lat, _, _ = at.quant_state_with_vq(lat)
            else:
                lat, _ = at.quant_state_without_vq(lat)
            T = at.downsampled_input_h if at.use_conv_encoder else 1
            out.append(lat.reshape(lat.shape[0] * T, -1).cpu().numpy())   # (N*T, A), as the original
        normalizer['latent_action'] = SingleFieldLinearNormalizer.create_fit(np.concatenate(out))
        return normalizer
