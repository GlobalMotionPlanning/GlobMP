"""Stock GR00T-N1.7 modality config for the DexMate Vega 1P (gripper) — PICK baseline.

22-DOF bimanual humanoid recorded at 30 Hz with 4 RGB cameras (head_left, head_right,
left_wrist, right_wrist). State/action layout (absolute joint targets):
  left_arm   [0:7]   right_arm [7:14]   torso [14:17]
  head       [17:20] left_hand [20:21]  right_hand [21:22]

This is a pure GR00T baseline: nothing is tuned. The action horizon is the base
GR00T-N1.7-3B native chunk length (40 steps) — we do NOT shorten it. Actions are ABSOLUTE
because the DexMate teleop is recorded as absolute joint targets.

Registers under EmbodimentTag.NEW_EMBODIMENT (the base model has no dexmate embodiment).
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

# Native GR00T-N1.7-3B action horizon (base config action_horizon = 40). Left stock.
ACTION_HORIZON = 40


def _abs():
    return ActionConfig(
        rep=ActionRepresentation.ABSOLUTE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
    )


vega_pick_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["head_left_rgb", "head_right_rgb", "left_wrist_rgb", "right_wrist_rgb"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_arm", "right_arm", "torso", "head", "left_hand", "right_hand"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, ACTION_HORIZON)),
        modality_keys=["left_arm", "right_arm", "torso", "head", "left_hand", "right_hand"],
        action_configs=[_abs(), _abs(), _abs(), _abs(), _abs(), _abs()],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(vega_pick_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
