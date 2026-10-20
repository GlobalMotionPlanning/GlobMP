"""Import-light Policy plugin registry; concrete plugins import lazily."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dex_deploy.policies.base import Policy

POLICY_REGISTRY: dict[str, type] = {}


def register_policy(name: str):
    """Class decorator registering a concrete Policy under name."""

    def _decorator(cls: type) -> type:
        POLICY_REGISTRY[name] = cls
        return cls

    return _decorator


def get_policy(name: str) -> type:
    """Return the registered Policy class for name, lazily importing its plugin."""
    if name not in POLICY_REGISTRY:
        try:
            importlib.import_module(f"dex_deploy.policies.{name}")
        except ImportError as exc:
            raise ValueError(
                f"Unknown policy {name!r}: could not import "
                f"'dex_deploy.policies.{name}' ({exc}). Known policies: "
                f"{sorted(POLICY_REGISTRY)}."
            ) from exc
    if name not in POLICY_REGISTRY:
        raise ValueError(
            f"Policy {name!r} imported but did not register itself. Did the "
            f"plugin module use @register_policy({name!r})? Known policies: "
            f"{sorted(POLICY_REGISTRY)}."
        )
    return POLICY_REGISTRY[name]


def auto_detect_policy(checkpoint: str) -> str:
    """Infer the policy name from a checkpoint layout. A GR00T checkpoint is a DIRECTORY
    containing `experiment_cfg/`."""
    p = Path(checkpoint)
    if (p / "experiment_cfg").is_dir():
        return "gr00t"
    raise FileNotFoundError(
        f"Cannot auto-detect a policy at {checkpoint!r}: expected a checkpoint "
        f"directory containing 'experiment_cfg/'. Known policies: "
        f"{sorted(POLICY_REGISTRY) or '[none imported yet]'}."
    )
