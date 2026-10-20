"""Modality config for UMI handheld-tracker data — GR00T PUSH-BOX training.

UMI embodiment (NOT the 22-DOF teleop robot): a single 6-DOF end-effector recorded
at 30 Hz with ONE egocentric GoPro camera. State/action is a 9-D EE-pose vector:
  ee_pose [0:9] = position [0:3] + 6-D continuous rotation (R[:,0], R[:,1]) [3:9]

Actions are ABSOLUTE, treated as a single NON_EEF group (the 9-D pose vector is
consumed like a generic continuous action — no EEF-specific relative chunking), so
the GR00T engine code is used UNCHANGED; only this config + the dataset differ from
the teleop pick config (configs/vega_pick_modality.py).

Registers under EmbodimentTag.NEW_EMBODIMENT.
Use:  --modality-config-path configs/vega_umi_modality.py
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

ACTION_HORIZON = 40


def _abs():
    return ActionConfig(
        rep=ActionRepresentation.ABSOLUTE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
    )


vega_umi_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["gopro"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ee_pose"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, ACTION_HORIZON)),
        modality_keys=["ee_pose"],
        action_configs=[_abs()],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(vega_umi_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
