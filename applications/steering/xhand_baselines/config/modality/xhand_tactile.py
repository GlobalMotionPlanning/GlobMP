"""GR00T modality config for Vega-1 + XHand1 teleop (any task) — WITH tactile.

XHand is 12-DoF per hand against T-Rex's 22, so state/action are 38-D here, not 58. The state
vector carries tactile at [38:68]; this config simply does not name it, so the no-tactile run
never sees it and the two variants share one conversion.

Absolute joint targets (the recording stores commanded positions), 4 cameras, native 40-step
horizon, registered under NEW_EMBODIMENT.
"""
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (ActionConfig, ActionFormat, ActionRepresentation, ActionType,
                              ModalityConfig)

ACTION_HORIZON = 40
KEYS = ["left_arm", "left_hand", "right_arm", "right_hand"]
CAMS = ["head_left_rgb", "head_right_rgb", "left_wrist", "right_wrist"]
TACTILE = True


def _abs():
    return ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF,
                        format=ActionFormat.DEFAULT)


cfg = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=CAMS),
    "state": ModalityConfig(delta_indices=[0],
                            modality_keys=KEYS + (["tactile"] if TACTILE else [])),
    "action": ModalityConfig(delta_indices=list(range(ACTION_HORIZON)), modality_keys=KEYS,
                             action_configs=[_abs() for _ in KEYS]),
    "language": ModalityConfig(delta_indices=[0],
                               modality_keys=["annotation.human.action.task_description"]),
}
register_modality_config(cfg, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
