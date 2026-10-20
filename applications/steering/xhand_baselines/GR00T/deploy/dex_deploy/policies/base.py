"""Policy contract for the policy-agnostic deploy framework."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Policy(Protocol):
    """Structural interface every deployable policy must satisfy."""

    @classmethod
    def load(cls, checkpoint: str, device: str, **kwargs) -> "Policy":
        """Load a policy from a checkpoint and return a ready-to-run instance."""
        ...

    def reset(self, task: str) -> None:
        ...

    @property
    def video_keys(self) -> list[str]:
        ...

    @property
    def action_dim(self) -> int:
        ...

    def predict_action(
        self,
        state: dict[str, np.ndarray],
        images: dict[str, np.ndarray],
        task: str,
    ) -> np.ndarray:
        """Map one observation to ONE action step, shape (D,) float32. Images are RGB."""
        ...
