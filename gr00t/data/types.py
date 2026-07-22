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

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any

import numpy as np

from gr00t.data.embodiment_tags import EmbodimentTag


class MessageType(Enum):
    START_OF_EPISODE = "start_of_episode"
    END_OF_EPISODE = "end_of_episode"
    EPISODE_STEP = "episode_step"
    IMAGE = "image"
    TEXT = "text"


class ActionRepresentation(Enum):
    RELATIVE = "relative"
    DELTA = "delta"
    ABSOLUTE = "absolute"


class ActionType(Enum):
    EEF = "eef"
    NON_EEF = "non_eef"


class ActionFormat(Enum):
    DEFAULT = "default"
    XYZ = "xyz"
    XYZ_ROT6D = "xyz+rot6d"
    XYZ_ROTVEC = "xyz+rotvec"
    XYZ_QUAT = "xyz+quat"


@dataclass
class VLAStepData:
    """
    Represents a single step of VLA (Vision-Language-Action) data.

    This is the core data structure returned by datasets, containing raw observation
    and action data that will be processed by the SequenceVLAProcessor.
    """

    # Core data
    images: dict[str, list[np.ndarray]]  # view_name -> list[np.ndarray] (for temporal stacking)
    states: dict[
        str, np.ndarray
    ]  # state_name -> np.ndarray (dim,) for single step or (horizon, dim) for trajectory
    actions: dict[str, np.ndarray]  # action_name -> np.ndarray (horizon, dim) for action chunk
    masks: dict[str, list[np.ndarray]] | None = None  # view_name -> list[np.ndarray] (H, W)
    text: str | None = None  # Optional task description or instruction
    embodiment: EmbodimentTag = (
        EmbodimentTag.NEW_EMBODIMENT
    )  # Optional embodiment tag for cross-embodiment training
    is_demonstration: bool = False  # Whether the step is a demonstration. If True, no loss should be computed for this step.

    # Flexible metadata that can be extended by users
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionConfig:
    rep: ActionRepresentation
    type: ActionType
    format: ActionFormat
    state_key: str | None = None
    source_format: ActionFormat | None = None
    delta_indices: list[int] | None = None
    model_offset: int | None = None
    loss_weight: float = 1.0
    decode_to_absolute: bool = True
    is_auxiliary: bool = False


@dataclass
class ModalityConfig:
    """Configuration for a modality defining how data should be sampled and loaded.

    This class specifies which indices to sample relative to a base index and which
    keys to load for a particular modality (e.g., video, state, action).
    """

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""
    sin_cos_embedding_keys: list[str] | None = None
    """Optional list of keys to apply sin/cos encoding. If None or empty, use min/max normalization for all keys."""
    mean_std_embedding_keys: list[str] | None = None
    """Optional list of keys to apply mean/std normalization. If None or empty, use min/max normalization for all keys."""
    first_frame_relative_quaternion_keys: list[str] | None = None
    """Optional list of quaternion keys to express as rotation relative to the first frame of each episode."""
    reference_only_keys: list[str] | None = None
    """State keys loaded only as action reference frames, never encoded as model state."""
    action_configs: list[ActionConfig] | None = None

    def __post_init__(self):
        """Validate fields and set default values."""
        if self.delta_indices is None or not isinstance(self.delta_indices, list):
            raise ValueError(f"delta_indices must be a non-None list, got {self.delta_indices!r}")
        if (
            self.modality_keys is None
            or not isinstance(self.modality_keys, list)
            or len(self.modality_keys) == 0
        ):
            raise ValueError(f"modality_keys must be a non-empty list, got {self.modality_keys!r}")
        if self.reference_only_keys is not None:
            if not isinstance(self.reference_only_keys, list):
                raise ValueError(
                    f"reference_only_keys must be a list, got {self.reference_only_keys!r}"
                )
            if len(self.reference_only_keys) != len(set(self.reference_only_keys)):
                raise ValueError("reference_only_keys must not contain duplicates")
            overlap = set(self.modality_keys) & set(self.reference_only_keys)
            if overlap:
                raise ValueError(
                    "reference_only_keys must not overlap model modality_keys, got "
                    f"{sorted(overlap)}"
                )
        if self.action_configs is not None:
            assert len(self.action_configs) == len(self.modality_keys), (
                f"Number of action configs ({len(self.action_configs)}) must match number of modality keys ({len(self.modality_keys)})"
            )
            parsed_action_configs = []
            for action_config in self.action_configs:
                if isinstance(action_config, dict):
                    action_config = ActionConfig(
                        rep=ActionRepresentation[action_config["rep"]],
                        type=ActionType[action_config["type"]],
                        format=ActionFormat[action_config["format"]],
                        state_key=action_config.get("state_key", None),
                        source_format=(
                            ActionFormat[action_config["source_format"]]
                            if action_config.get("source_format") is not None
                            else None
                        ),
                        delta_indices=action_config.get("delta_indices"),
                        model_offset=action_config.get("model_offset"),
                        loss_weight=action_config.get("loss_weight", 1.0),
                        decode_to_absolute=action_config.get("decode_to_absolute", True),
                        is_auxiliary=action_config.get("is_auxiliary", False),
                    )
                if action_config.delta_indices is not None and len(
                    action_config.delta_indices
                ) != len(self.delta_indices):
                    raise ValueError(
                        "Per-key action delta_indices must have the same length as the "
                        f"modality horizon ({len(self.delta_indices)}), got "
                        f"{len(action_config.delta_indices)}"
                    )
                if action_config.model_offset is not None and action_config.model_offset < 0:
                    raise ValueError(
                        f"Action model_offset must be non-negative, got {action_config.model_offset}"
                    )
                if action_config.format == ActionFormat.XYZ:
                    if action_config.type != ActionType.EEF:
                        raise ValueError("ActionFormat.XYZ is only supported for EEF actions")
                    if action_config.source_format not in {
                        ActionFormat.XYZ_ROT6D,
                        ActionFormat.XYZ_ROTVEC,
                        ActionFormat.XYZ_QUAT,
                    }:
                        raise ValueError(
                            "XYZ-only EEF actions require a source_format containing rotation"
                        )
                    if action_config.decode_to_absolute:
                        raise ValueError(
                            "XYZ-only EEF actions cannot decode an absolute rotation; "
                            "set decode_to_absolute=False"
                        )
                if not math.isfinite(action_config.loss_weight) or action_config.loss_weight <= 0:
                    raise ValueError(
                        f"Action loss_weight must be finite and positive, got {action_config.loss_weight}"
                    )
                if action_config.is_auxiliary and action_config.type != ActionType.EEF:
                    raise ValueError("Only EEF actions can use the auxiliary action head")
                parsed_action_configs.append(action_config)
            self.action_configs = parsed_action_configs
