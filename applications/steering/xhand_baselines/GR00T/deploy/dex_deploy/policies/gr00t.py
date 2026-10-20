"""GR00T policy plugin for the dex_deploy robot harness.

Wraps the GR00T-N1.7 `Gr00tPolicy` into the dex_deploy `Policy` contract. The obs format,
`get_action` call, and 22-DOF action assembly mirror this repo's proven `inference/
rollout_to_npz.py` exactly; the plugin adds the per-control-step action-chunk queue. The
`gr00t` package must be importable and `GROOT_BACKBONE_PATH` set (see run_robot.sh).

Checkpoint: a GR00T checkpoint DIRECTORY containing `experiment_cfg/`.
"""
from __future__ import annotations

import collections

import numpy as np

from dex_deploy.policies import register_policy

# 22-DOF joint layout (identical to the modality config + rollout_to_npz.py)
STATE_SLICES = [("left_arm", 0, 7), ("right_arm", 7, 14), ("torso", 14, 17),
                ("head", 17, 20), ("left_hand", 20, 21), ("right_hand", 21, 22)]
GROUPS = [s[0] for s in STATE_SLICES]
CAMS = ["head_left_rgb", "head_right_rgb", "left_wrist_rgb", "right_wrist_rgb"]
LANG_KEY = "annotation.human.action.task_description"


@register_policy("gr00t")
class GR00TPolicy:
    def __init__(self, policy, action_horizon, device, action_chunk_threshold, task):
        self._policy = policy
        self.action_horizon = int(action_horizon)
        self._action_dim = 22
        self.device = device
        self.threshold = float(action_chunk_threshold)
        self.task = task or ""
        self._queue = collections.deque(maxlen=self.action_horizon)

    @classmethod
    def load(cls, checkpoint, device="cuda:0", action_chunk_threshold: float = 1.0,
             task: str = "", **kwargs):
        # heavy gr00t/torch imports are lazy here to keep package import light
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        policy = Gr00tPolicy(embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
                             model_path=str(checkpoint), device=device)
        # action horizon = len(action.delta_indices) from the checkpoint's modality config
        try:
            mcfg = policy.get_modality_config()
            action_horizon = len(mcfg["action"].delta_indices)
        except Exception:
            action_horizon = 40   # GR00T-N1.7 default
        print(f"[GR00T] loaded from {checkpoint}  (horizon={action_horizon}, dev={device})")
        return cls(policy, action_horizon, device, action_chunk_threshold, task)

    def reset(self, task: str) -> None:
        if task:
            self.task = task
        self._queue.clear()

    @property
    def video_keys(self) -> list[str]:
        return [f"{c}_img" for c in CAMS]

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def _state22(self, state: dict) -> np.ndarray:
        vec = np.concatenate([np.asarray(state[g], np.float32).reshape(-1) for g in GROUPS])
        if vec.shape[0] != 22:
            raise ValueError(f"assembled state is {vec.shape[0]}-dim, expected 22")
        return vec

    def _infer_chunk(self, state: dict, images: dict) -> np.ndarray:
        st22 = self._state22(state)
        obs = {
            "video": {cam: images[cam][None, None] for cam in CAMS},              # (1,1,H,W,3)
            "state": {g: st22[a:b][None, None] for g, a, b in STATE_SLICES},       # (1,1,D)
            "language": {LANG_KEY: [[self.task]]},
        }
        action, _ = self._policy.get_action(obs)                                  # (action_dict, info)
        H = int(np.asarray(action[GROUPS[0]]).shape[1])                           # (1, H, D)
        # concat the 6 groups in canonical order for each step of the chunk
        chunk = np.stack([
            np.concatenate([np.asarray(action[g])[0, k].reshape(-1) for g in GROUPS])
            for k in range(H)
        ]).astype(np.float32)                                                     # (H, 22)
        return chunk

    def predict_action(self, state, images, task) -> np.ndarray:
        if task and task != self.task:
            self.task = task
        imgs = {}
        for vk, img in images.items():
            cam = vk[:-4] if vk.endswith("_img") else vk
            if cam in CAMS:
                imgs[cam] = img
        missing = [c for c in CAMS if c not in imgs]
        if missing:
            raise KeyError(f"missing camera(s) {missing}; got video keys {sorted(images)}")

        chunk_status = len(self._queue) / self.action_horizon
        if len(self._queue) == 0 or chunk_status <= 1.0 - self.threshold:
            self._queue.clear()
            self._queue.extend(self._infer_chunk(state, imgs))
        return np.asarray(self._queue.popleft(), dtype=np.float32).reshape(-1)
