# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


MODALITY_CONFIGS = {
    ##### Pre-registered pretrain configurations #####
    "oxe_droid_relative_eef_relative_joint": {
        "video": ModalityConfig(
            delta_indices=[-15, 0],
            modality_keys=["exterior_image_1_left", "wrist_image_left"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["eef_9d", "gripper_position", "joint_position"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(40)),
            modality_keys=["eef_9d", "gripper_position", "joint_position"],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                    state_key="eef_9d",
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="gripper_position",
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="joint_position",
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.language.language_instruction"],
        ),
    },
    ##### Pre-registered posttrain configurations #####
    "unitree_g1_sonic": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_leg",
                "right_leg",
                "waist",
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
                "projected_gravity",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(40)),
            modality_keys=[
                "motion_token",
                "left_hand_joints",
                "right_hand_joints",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    "unitree_g1_smpl": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_leg",
                "right_leg",
                "waist",
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
                "projected_gravity",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(50)),
            modality_keys=[
                "smpl_pose",
                "body_quat_w",
                "left_hand_joints",
                "right_hand_joints",
            ],
            first_frame_relative_quaternion_keys=["body_quat_w"],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    "unitree_g1_full_body_with_waist_height_nav_cmd": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_leg",
                "right_leg",
                "waist",
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(50)),
            modality_keys=[
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
                "waist",
                "base_height_command",
                "navigate_command",
            ],
            action_configs=[
                # left_arm
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # right_arm
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # left_hand
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,  # G1 hand is controlled by binary signals like a gripper
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # right_hand
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,  # G1 hand is controlled by binary signals like a gripper
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # waist
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # base_height_command
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # navigate_command
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    "libero_sim": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["image", "wrist_image"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "simpler_env_widowx": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["image_0"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(8)),
            modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "simpler_env_google": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["image"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["x", "y", "z", "rx", "ry", "rz", "rw", "gripper"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(8)),
            modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
}

MODALITY_CONFIGS["unitree_g1_sonic_human"] = deepcopy(MODALITY_CONFIGS["unitree_g1_sonic"])


_SONIC_TOKEN_ACTION_CONFIGS = [
    ActionConfig(
        rep=ActionRepresentation.ABSOLUTE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
        model_offset=0,
    ),
    ActionConfig(
        rep=ActionRepresentation.ABSOLUTE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
        model_offset=64,
    ),
    ActionConfig(
        rep=ActionRepresentation.ABSOLUTE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
        model_offset=71,
    ),
]


def _eef_action_config(
    key: str,
    model_offset: int,
    delta_indices: list[int],
) -> ActionConfig:
    return ActionConfig(
        rep=ActionRepresentation.RELATIVE,
        type=ActionType.EEF,
        format=ActionFormat.XYZ_ROT6D,
        source_format=ActionFormat.XYZ_QUAT,
        state_key=key,
        delta_indices=delta_indices,
        model_offset=model_offset,
        decode_to_absolute=False,
        is_auxiliary=True,
    )


def _register_sonic_endpose_config(
    tag: str,
    endpose_keys: list[str],
    include_token: bool,
    endpose_delta_indices: list[int],
) -> None:
    config = deepcopy(MODALITY_CONFIGS["unitree_g1_sonic"])
    config["state"].reference_only_keys = list(endpose_keys)

    action_keys = []
    action_configs = []
    if include_token:
        action_keys.extend(["motion_token", "left_hand_joints", "right_hand_joints"])
        action_configs.extend(deepcopy(_SONIC_TOKEN_ACTION_CONFIGS))
    action_keys.extend(endpose_keys)
    action_configs.extend(
        _eef_action_config(
            key,
            model_offset=78 + 9 * index,
            delta_indices=endpose_delta_indices,
        )
        for index, key in enumerate(endpose_keys)
    )
    config["action"] = ModalityConfig(
        delta_indices=list(range(40)),
        modality_keys=action_keys,
        action_configs=action_configs,
    )
    MODALITY_CONFIGS[tag] = config


def enable_sonic_endpose_xyz_only(config: dict[str, ModalityConfig]) -> None:
    """Keep EEF frame conversion but supervise only its relative xyz translation.

    EEF model offsets retain their 9D-lane spacing so xyz-only and full-pose
    checkpoints agree on the semantic location of every limb. Token and EEF
    losses are normalized independently by the model.
    """
    action_config = config.get("action")
    if action_config is None or action_config.action_configs is None:
        raise ValueError("endpose_xyz_only requires explicit action configs")

    eef_configs = [
        item for item in action_config.action_configs if item.type == ActionType.EEF
    ]
    if not eef_configs:
        raise ValueError("endpose_xyz_only requires at least one EEF action")
    for item in eef_configs:
        if item.source_format is None:
            raise ValueError("endpose_xyz_only requires each EEF action to define source_format")
        item.format = ActionFormat.XYZ
        item.decode_to_absolute = False



_register_sonic_endpose_config(
    "unitree_g1_sonic_endpose",
    [
        "left_wrist_endpose",
        "right_wrist_endpose",
        "left_foot_endpose",
        "right_foot_endpose",
    ],
    include_token=False,
    endpose_delta_indices=list(range(1, 41)),
)
_register_sonic_endpose_config(
    "unitree_g1_sonic_token_endpose",
    [
        "left_wrist_endpose",
        "right_wrist_endpose",
        "left_foot_endpose",
        "right_foot_endpose",
    ],
    include_token=True,
    endpose_delta_indices=list(range(1, 41)),
)
_register_sonic_endpose_config(
    "unitree_g1_sonic_human_endpose",
    [
        "human_left_hand_endpose",
        "human_right_hand_endpose",
        "human_left_foot_endpose",
        "human_right_foot_endpose",
    ],
    include_token=False,
    endpose_delta_indices=list(range(1, 41)),
)
_register_sonic_endpose_config(
    "unitree_g1_sonic_human_token_endpose",
    [
        "human_left_hand_endpose",
        "human_right_hand_endpose",
        "human_left_foot_endpose",
        "human_right_foot_endpose",
    ],
    include_token=True,
    endpose_delta_indices=list(range(1, 41)),
)


def register_modality_config(
    config: dict, embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
):
    assert embodiment_tag.value not in MODALITY_CONFIGS, (
        f"Embodiment tag {embodiment_tag} already registered"
    )
    MODALITY_CONFIGS[embodiment_tag.value] = config
