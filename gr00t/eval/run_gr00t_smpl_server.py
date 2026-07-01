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

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.policy import BasePolicy, PolicyWrapper
from gr00t.policy.server_client import PolicyServer
import numpy as np
import tyro


DEFAULT_MODEL_SERVER_PORT = 5555

SMPL_ACTION_DIMS = {
    "smpl_pose": 63,
    "body_quat_w": 4,
    "left_hand_joints": 7,
    "right_hand_joints": 7,
}


class SmplActionPolicyWrapper(PolicyWrapper):
    """Adapt GR00T SMPL actions to the flat single-sample format expected by the SMPL client."""

    def __init__(
        self,
        policy: BasePolicy,
        *,
        expected_horizon: int = 50,
        strict: bool = True,
    ):
        super().__init__(policy, strict=strict)
        self.expected_horizon = expected_horizon

    def check_observation(self, observation: dict[str, Any]) -> None:
        self.policy.check_observation(observation)

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        action, info = self.policy.get_action(observation, options)
        smpl_action = {}

        for action_key in SMPL_ACTION_DIMS:
            if action_key not in action:
                raise KeyError(
                    f"Expected SMPL action key '{action_key}' in policy output. "
                    f"Available keys: {sorted(action.keys())}"
                )

            action_arr = np.asarray(action[action_key], dtype=np.float32)
            if action_arr.ndim == 3:
                if action_arr.shape[0] != 1:
                    raise ValueError(
                        f"SMPL server only supports single-sample batches. "
                        f"Action '{action_key}' has batch size {action_arr.shape[0]}."
                    )
                action_arr = action_arr[0]

            smpl_action[f"action.{action_key}"] = action_arr

        return smpl_action, info

    def check_action(self, action: dict[str, Any]) -> None:
        for action_key, expected_dim in SMPL_ACTION_DIMS.items():
            flat_key = f"action.{action_key}"
            assert flat_key in action, f"Action key '{flat_key}' must be in action"

            action_arr = action[flat_key]
            assert isinstance(action_arr, np.ndarray), (
                f"Action key '{flat_key}' must be a numpy array. Got {type(action_arr)}"
            )
            assert action_arr.dtype == np.float32, (
                f"Action key '{flat_key}' must be np.float32. Got {action_arr.dtype}"
            )
            assert action_arr.ndim == 2, (
                f"Action key '{flat_key}' must have shape (T, D). Got {action_arr.shape}"
            )
            if self.expected_horizon > 0:
                assert action_arr.shape[0] == self.expected_horizon, (
                    f"Action key '{flat_key}' horizon must be {self.expected_horizon}. "
                    f"Got {action_arr.shape[0]}"
                )
            assert action_arr.shape[1] == expected_dim, (
                f"Action key '{flat_key}' dim must be {expected_dim}. Got {action_arr.shape[1]}"
            )

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.policy.get_modality_config()


@dataclass
class SmplServerConfig:
    """Configuration for running the GR00T SMPL inference server."""

    model_path: str
    """Path to the model checkpoint directory."""

    embodiment_tag: str = "UNITREE_G1_SMPL"
    """Embodiment tag (name or value, case-insensitive)."""

    device: str = "cuda"
    """Device to run the model on."""

    host: str = "0.0.0.0"
    """Host address for the server."""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server."""

    strict: bool = True
    """Whether to enforce strict input and SMPL output validation."""

    expected_horizon: int = 50
    """Expected SMPL action horizon. Set <= 0 to disable horizon validation."""


def main(config: SmplServerConfig):
    embodiment_tag = EmbodimentTag.resolve(config.embodiment_tag)
    print("Starting GR00T SMPL inference server...")
    print(f"  Embodiment tag: {embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")
    print(f"  Expected SMPL horizon: {config.expected_horizon}")

    model_path = Path(config.model_path)
    if model_path.is_absolute() and not model_path.exists():
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=config.model_path,
        device=config.device,
        strict=config.strict,
    )
    policy = SmplActionPolicyWrapper(
        policy,
        expected_horizon=config.expected_horizon,
        strict=config.strict,
    )

    server = PolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
    )

    print(f"\nServer ready - listening on {config.host}:{config.port}\n")
    print("SMPL action response keys:")
    for action_key, action_dim in SMPL_ACTION_DIMS.items():
        print(f"  action.{action_key}: ({config.expected_horizon}, {action_dim})")
    print()

    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    config = tyro.cli(SmplServerConfig)
    main(config)
