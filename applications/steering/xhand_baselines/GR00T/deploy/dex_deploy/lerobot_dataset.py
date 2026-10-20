"""Read one episode of a LeRobot v2.1 dataset for offline eval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

_DEFAULT_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
_DEFAULT_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
_DEFAULT_CHUNKS_SIZE = 1000


def _to_1d_float32(cell: Any) -> np.ndarray:
    return np.atleast_1d(np.asarray(cell, dtype=np.float32)).reshape(-1)


class LeRobotEpisode:
    """Lazy reader over one episode of a LeRobot v2.1 dataset."""

    def __init__(
        self,
        dataset_path: str,
        episode_index: int,
        image_resolution: Optional[tuple] = None,
    ):
        import pandas as pd

        self.dataset_path = Path(dataset_path)
        self.episode_index = int(episode_index)
        self.image_resolution = image_resolution

        meta_dir = self.dataset_path / "meta"
        info_path = meta_dir / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"info.json not found: {info_path}")
        with open(info_path) as f:
            info = json.load(f)

        self.chunks_size = int(info.get("chunks_size", _DEFAULT_CHUNKS_SIZE))
        self._data_path_pattern = info.get("data_path", _DEFAULT_DATA_PATH)
        self._video_path_pattern = info.get("video_path", _DEFAULT_VIDEO_PATH)
        self.episode_chunk = self.episode_index // self.chunks_size

        modality_path = meta_dir / "modality.json"
        self._video_original_key: dict[str, str] = {}
        if modality_path.is_file():
            with open(modality_path) as f:
                modality = json.load(f)
            for policy_key, spec in modality.get("video", {}).items():
                self._video_original_key[policy_key] = spec.get("original_key", "")

        parquet_path = self.dataset_path / self._data_path_pattern.format(
            episode_chunk=self.episode_chunk, episode_index=self.episode_index
        )
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")
        self.parquet_path = parquet_path
        self.df = pd.read_parquet(parquet_path)
        self.num_frames = len(self.df)

        self._decoders: dict[str, Any] = {}

    def _video_path_for(self, policy_video_key: str) -> Path:
        original_key = self._video_original_key.get(policy_video_key, policy_video_key)
        return self.dataset_path / self._video_path_pattern.format(
            episode_chunk=self.episode_chunk,
            episode_index=self.episode_index,
            video_key=original_key,
        )

    def _get_decoder(self, policy_video_key: str):
        decoder = self._decoders.get(policy_video_key)
        if decoder is None:
            from torchcodec.decoders import VideoDecoder

            mp4 = self._video_path_for(policy_video_key)
            if not mp4.is_file():
                raise FileNotFoundError(
                    f"Video for {policy_video_key!r} not found: {mp4}"
                )
            decoder = VideoDecoder(str(mp4))
            self._decoders[policy_video_key] = decoder
        return decoder

    def get_state(self, frame_idx: int, state_keys: list) -> dict:
        """Nested state {"joint_pos": {<component>: 1-D float32}} for frame_idx."""
        row = self.df.iloc[frame_idx]
        joint_pos: dict[str, np.ndarray] = {}
        for key in state_keys:
            col = f"observation.state.{key}"
            component = key.split(".")[0]
            joint_pos[component] = _to_1d_float32(row[col])
        return {"joint_pos": joint_pos}

    def get_images(self, frame_idx: int, video_keys: list) -> dict:
        """Decode one frame per camera as HxWx3 uint8 RGB; images stay RGB (policy trained on RGB)."""
        images: dict[str, np.ndarray] = {}
        for video_key in video_keys:
            decoder = self._get_decoder(video_key)
            frame = decoder[frame_idx]
            img = frame.permute(1, 2, 0).cpu().numpy()
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            if self.image_resolution is not None:
                import cv2

                img = cv2.resize(img, tuple(self.image_resolution))
            images[video_key] = img
        return images

    def get_gt_action(self, frame_idx: int, action_keys: list) -> np.ndarray:
        """Concatenate action.<key> over action_keys in order -> 1-D float32 vector."""
        row = self.df.iloc[frame_idx]
        parts = [_to_1d_float32(row[f"action.{key}"]) for key in action_keys]
        return np.concatenate(parts).astype(np.float32)
