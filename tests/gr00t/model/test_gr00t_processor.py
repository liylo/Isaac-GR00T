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

"""
Test Gr00tN1d7Processor: state/action processing, VLM input generation, decode_action.

Uses the fixture configs in tests/fixtures/processor_config/ with a mocked VLM
processor (no model download needed).
"""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData
import numpy as np
from PIL import Image
import pytest
import torch


FIXTURE_DIR = Path(__file__).parent.parent.parent / "fixtures" / "processor_config"
EMBODIMENT = "libero_sim"


@pytest.fixture
def processor():
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    mock_vlm = MagicMock()
    mock_vlm.apply_chat_template.return_value = "mock text"
    mock_vlm.tokenizer.padding_side = "left"

    with patch(
        "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.build_processor",
        return_value=mock_vlm,
    ):
        proc = Gr00tN1d7Processor.from_pretrained(FIXTURE_DIR)
    proc.eval()
    return proc


@pytest.fixture
def proc_config():
    with open(FIXTURE_DIR / "processor_config.json") as f:
        return json.load(f)["processor_kwargs"]


def _make_step_data(proc_config) -> VLAStepData:
    """Create synthetic VLAStepData matching the fixture config."""
    import json as _json

    mc = proc_config["modality_configs"][EMBODIMENT]
    state_keys = mc["state"]["modality_keys"]
    action_keys = mc["action"]["modality_keys"]
    video_keys = mc["video"]["modality_keys"]
    action_horizon = len(mc["action"]["delta_indices"])

    # Load statistics to get correct dimensions per key
    with open(FIXTURE_DIR / "statistics.json") as f:
        statistics = _json.load(f)

    images = {k: [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)] for k in video_keys}
    states = {
        k: np.random.randn(1, len(statistics[EMBODIMENT]["state"][k]["min"])).astype(np.float32)
        for k in state_keys
    }
    actions = {
        k: np.random.randn(action_horizon, len(statistics[EMBODIMENT]["action"][k]["min"])).astype(
            np.float32
        )
        for k in action_keys
    }

    return VLAStepData(
        images=images,
        states=states,
        actions=actions,
        text="pick up the apple",
        embodiment=EmbodimentTag(EMBODIMENT),
    )


class TestProcessorCall:
    """Test __call__ with VLAStepData messages."""

    def test_call_returns_expected_keys(self, processor, proc_config):
        step_data = _make_step_data(proc_config)
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": step_data}]
        result = processor(messages)
        assert "state" in result
        assert "action" in result
        assert "embodiment_id" in result
        assert "action_mask" in result
        assert "primary_action_mask" in result

    def test_state_padded_to_max_dim(self, processor, proc_config):
        step_data = _make_step_data(proc_config)
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": step_data}]
        result = processor(messages)
        state = result["state"]
        max_dim = proc_config["max_state_dim"]
        assert state.shape[-1] == max_dim, f"Expected state dim {max_dim}, got {state.shape[-1]}"

    def test_action_padded_to_max_dim(self, processor, proc_config):
        step_data = _make_step_data(proc_config)
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": step_data}]
        result = processor(messages)
        action = result["action"]
        max_dim = proc_config["max_action_dim"]
        assert action.shape[-1] == max_dim, f"Expected action dim {max_dim}, got {action.shape[-1]}"

    def test_action_mask_matches_action(self, processor, proc_config):
        step_data = _make_step_data(proc_config)
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": step_data}]
        result = processor(messages)
        assert result["action_mask"].shape == result["action"].shape
        assert result["primary_action_mask"].shape == result["action"].shape

    def test_embodiment_id_is_integer(self, processor, proc_config):
        step_data = _make_step_data(proc_config)
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": step_data}]
        result = processor(messages)
        assert isinstance(result["embodiment_id"], (int, np.integer))

    def test_inference_without_actions_still_emits_primary_sampling_mask(
        self, processor, proc_config
    ):
        step_data = _make_step_data(proc_config)
        step_data.actions = None
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": step_data}]
        result = processor(messages)
        assert "action" not in result
        assert "action_mask" not in result
        assert result["primary_action_mask"].shape == (
            processor.max_action_horizon,
            processor.max_action_dim,
        )


class TestProcessorVLMInputs:
    """Test VLM input generation."""

    def test_get_vlm_inputs(self, processor):
        image_keys = processor.modality_configs[EMBODIMENT]["video"].modality_keys
        mock_images = {
            k: [Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))]
            for k in image_keys
        }
        vlm_inputs = processor._get_vlm_inputs(
            image_keys=image_keys,
            images=mock_images,
            image_transform=processor.eval_image_transform,
            language="pick up the apple",
            masks=None,
        )
        assert "vlm_content" in vlm_inputs


class TestProcessorDecodeAction:
    """Test action denormalization."""

    def test_decode_action_returns_all_keys(self, processor, proc_config):
        mc = proc_config["modality_configs"][EMBODIMENT]
        action_keys = mc["action"]["modality_keys"]
        # Load statistics so decode_action can denormalize
        with open(FIXTURE_DIR / "statistics.json") as f:
            stats = json.load(f)
        processor.set_statistics(stats)

        action_dim = processor.state_action_processor.get_action_dim(EMBODIMENT)
        action_horizon = len(mc["action"]["delta_indices"])
        dummy_action = np.random.randn(action_horizon, action_dim).astype(np.float32)
        result = processor.decode_action(dummy_action, EmbodimentTag(EMBODIMENT))
        assert set(result.keys()) == set(action_keys)


class TestProcessorStatistics:
    """Test statistics management."""

    def test_set_statistics(self, processor):
        with open(FIXTURE_DIR / "statistics.json") as f:
            stats = json.load(f)
        processor.set_statistics(stats)
        assert EMBODIMENT in processor.statistics

    def test_train_eval_mode(self, processor):
        processor.train()
        assert processor.training is True
        assert processor.state_action_processor.training is True
        processor.eval()
        assert processor.training is False
        assert processor.state_action_processor.training is False


class TestFixedActionLayout:
    def test_pack_actions_masks_only_configured_slots(self):
        from gr00t.data.state_action.state_action_processor import StateActionProcessor
        from gr00t.data.types import (
            ActionConfig,
            ActionFormat,
            ActionRepresentation,
            ActionType,
            ModalityConfig,
        )
        from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

        tag = "layout_test"
        configs = {
            tag: {
                "state": ModalityConfig(delta_indices=[0], modality_keys=["joints"]),
                "action": ModalityConfig(
                    delta_indices=[0, 1],
                    modality_keys=["eef"],
                    action_configs=[
                        ActionConfig(
                            ActionRepresentation.RELATIVE,
                            ActionType.EEF,
                            ActionFormat.XYZ_ROT6D,
                            model_offset=78,
                            loss_weight=2.0,
                        )
                    ],
                ),
            }
        }
        stat = {
            name: [value] * 9
            for name, value in {
                "mean": 0.0,
                "std": 1.0,
                "min": -1.0,
                "max": 1.0,
                "q01": -1.0,
                "q99": 1.0,
            }.items()
        }
        state_action_processor = StateActionProcessor(
            modality_configs=configs,
            statistics={tag: {"state": {}, "action": {"eef": stat}}},
        )
        processor = object.__new__(Gr00tN1d7Processor)
        processor.modality_configs = configs
        processor.state_action_processor = state_action_processor
        processor.max_action_dim = 132

        values = np.ones((2, 9), dtype=np.float32)
        packed, mask, auxiliary_mask = processor._pack_actions({"eef": values}, tag)
        assert packed.shape == (2, 132)
        assert torch.count_nonzero(packed[:, :78]) == 0
        assert torch.all(packed[:, 78:87] == 1)
        assert torch.count_nonzero(packed[:, 87:]) == 0
        assert torch.count_nonzero(mask[:, :78]) == 0
        assert torch.all(mask[:, 78:87] == 2)
        assert torch.count_nonzero(mask[:, 87:]) == 0
        assert torch.count_nonzero(auxiliary_mask) == 0

    @pytest.mark.parametrize(
        ("tag", "include_token"),
        [
            ("unitree_g1_sonic_endpose", False),
            ("unitree_g1_sonic_token_endpose", True),
            ("unitree_g1_sonic_human_endpose", False),
            ("unitree_g1_sonic_human_token_endpose", True),
        ],
    )
    def test_sonic_endpose_tags_share_timing_slots_and_projector(self, tag, include_token):
        from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
        from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import EMBODIMENT_TAG_TO_PROJECTOR_INDEX

        config = MODALITY_CONFIGS[tag]
        action_config_by_key = dict(
            zip(config["action"].modality_keys, config["action"].action_configs)
        )
        endpose_keys = config["state"].reference_only_keys

        assert EMBODIMENT_TAG_TO_PROJECTOR_INDEX[tag] == 11
        assert len(endpose_keys) == 4
        for index, key in enumerate(endpose_keys):
            action_config = action_config_by_key[key]
            assert action_config.delta_indices == list(range(1, 41))
            assert action_config.model_offset == 78 + 9 * index
            assert action_config.loss_weight == 1.0
            assert action_config.decode_to_absolute is False
            assert action_config.is_auxiliary is True

        token_keys = ["motion_token", "left_hand_joints", "right_hand_joints"]
        if include_token:
            assert config["action"].modality_keys[:3] == token_keys
            assert [action_config_by_key[key].model_offset for key in token_keys] == [0, 64, 71]
            assert all(action_config_by_key[key].loss_weight == 1.0 for key in token_keys)
        else:
            assert not set(token_keys) & set(config["action"].modality_keys)

    @pytest.mark.parametrize(
        ("tag", "include_token"),
        [
            ("unitree_g1_sonic_endpose", False),
            ("unitree_g1_sonic_token_endpose", True),
            ("unitree_g1_sonic_human_endpose", False),
            ("unitree_g1_sonic_human_token_endpose", True),
        ],
    )
    def test_sonic_xyz_only_keeps_limb_lanes(self, tag, include_token):
        from gr00t.configs.data.embodiment_configs import (
            MODALITY_CONFIGS,
            enable_sonic_endpose_xyz_only,
        )
        from gr00t.data.types import ActionFormat, ActionType

        config = deepcopy(MODALITY_CONFIGS[tag])
        enable_sonic_endpose_xyz_only(config)
        configs = config["action"].action_configs
        eef_configs = [item for item in configs if item.type is ActionType.EEF]

        assert len(eef_configs) == 4
        assert [item.model_offset for item in eef_configs] == [78, 87, 96, 105]
        assert all(item.format is ActionFormat.XYZ for item in eef_configs)
        assert all(item.decode_to_absolute is False for item in eef_configs)

        token_configs = [item for item in configs if item.type is ActionType.NON_EEF]
        if include_token:
            assert len(token_configs) == 3
            assert all(item.loss_weight == 1.0 for item in token_configs)
        else:
            assert token_configs == []

    @pytest.mark.parametrize(
        ("tag", "expected_primary_dims"),
        [
            ("unitree_g1_sonic_endpose", 0),
            ("unitree_g1_sonic_token_endpose", 78),
            ("unitree_g1_sonic_human_endpose", 0),
            ("unitree_g1_sonic_human_token_endpose", 78),
        ],
    )
    def test_primary_sampling_mask_excludes_auxiliary_slots(self, tag, expected_primary_dims):
        from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
        from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

        config = MODALITY_CONFIGS[tag]
        processor = object.__new__(Gr00tN1d7Processor)
        processor.modality_configs = {tag: config}
        processor.max_action_horizon = 40
        processor.max_action_dim = 132
        processor.state_action_processor = MagicMock()
        processor.state_action_processor.get_action_slices.return_value = {
            key: (
                action_config.model_offset,
                action_config.model_offset
                + (64 if key == "motion_token" else 7 if "hand_joints" in key else 9),
            )
            for key, action_config in zip(
                config["action"].modality_keys, config["action"].action_configs
            )
        }

        mask = processor._build_primary_action_mask(tag)
        assert mask.shape == (40, 132)
        assert torch.count_nonzero(mask[0]) == expected_primary_dims
        assert torch.count_nonzero(mask[:, 78:]) == 0


class TestFixtureCompleteness:
    """Guard against fixture drift: the fixture must contain every field that save_pretrained writes."""

    def test_fixture_keys_match_save_pretrained_roundtrip(self, processor):
        """Load the fixture, save to a temp dir, and verify the keys are identical.

        If this test fails, a field was added to save_pretrained() (or __init__)
        without updating the test fixture JSON.  Fix by running save_pretrained()
        on a default-constructed processor and copying the output back into
        tests/fixtures/processor_config/processor_config.json.
        """
        with open(FIXTURE_DIR / "processor_config.json") as f:
            fixture_kwargs = json.load(f)["processor_kwargs"]

        with tempfile.TemporaryDirectory() as tmp:
            processor.save_pretrained(tmp)
            with open(Path(tmp) / "processor_config.json") as f:
                saved_kwargs = json.load(f)["processor_kwargs"]

        fixture_keys = set(fixture_kwargs.keys())
        saved_keys = set(saved_kwargs.keys())

        missing = saved_keys - fixture_keys
        extra = fixture_keys - saved_keys

        assert not missing, (
            f"Fixture is missing fields that save_pretrained() writes: {missing}. "
            f"Update tests/fixtures/processor_config/processor_config.json."
        )
        assert not extra, (
            f"Fixture has fields that save_pretrained() no longer writes: {extra}. "
            f"Remove them from tests/fixtures/processor_config/processor_config.json."
        )
