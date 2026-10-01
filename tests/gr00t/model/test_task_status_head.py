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

"""Focused tests for the optional completion/progress task-status head."""

from types import SimpleNamespace
from unittest.mock import Mock

from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7, Gr00tTaskStatusHead
import pytest
import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature


def _status_config(**overrides) -> SimpleNamespace:
    values = {
        "task_status_head_only": True,
        "task_status_completion_window_frames": 25,
        "task_status_completion_pos_weight": 3.0,
        "task_status_completion_loss_weight": 2.0,
        "task_status_progress_loss_weight": 0.5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _backbone_outputs(features: torch.Tensor, mask: torch.Tensor | None = None) -> BatchFeature:
    if mask is None:
        mask = torch.ones(features.shape[:2], dtype=torch.long)
    return BatchFeature(
        data={
            "backbone_features": features,
            "backbone_attention_mask": mask,
        }
    )


def _status_outputs(
    head: Gr00tTaskStatusHead,
    features: torch.Tensor,
    frame_index: torch.Tensor,
    episode_length: torch.Tensor,
    config: SimpleNamespace | None = None,
) -> BatchFeature:
    harness = SimpleNamespace(
        task_status_head=head,
        config=config or _status_config(),
    )
    return Gr00tN1d7._get_task_status_outputs(
        harness,
        _backbone_outputs(features),
        BatchFeature(
            data={
                "frame_index": frame_index,
                "episode_length": episode_length,
            }
        ),
    )


def test_attention_mask_ignores_padded_tokens_and_blocks_their_gradients():
    torch.manual_seed(0)
    head = Gr00tTaskStatusHead(embedding_dim=16, hidden_dim=8, num_heads=4)
    head.eval()

    features = torch.randn(2, 5, 16)
    features[1, :3] = features[0, :3]
    features[1, 3:] = torch.nan
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]])

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = head(features, mask)
    torch.testing.assert_close(logits[0], logits[1])
    assert logits.dtype == torch.float32

    differentiable_features = features[:1].detach().clone().requires_grad_(True)
    head(differentiable_features, mask[:1]).sum().backward()
    assert torch.count_nonzero(differentiable_features.grad[:, :3]) > 0
    assert torch.count_nonzero(differentiable_features.grad[:, 3:]) == 0

    with pytest.raises(ValueError, match="at least one valid token"):
        head(features[:1], torch.zeros(1, 5, dtype=torch.long))


def test_reset_parameters_recovers_non_finite_status_head() -> None:
    head = Gr00tTaskStatusHead(embedding_dim=16, hidden_dim=8, num_heads=4)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.fill_(torch.nan)

    head.reset_parameters()

    assert all(torch.isfinite(parameter).all() for parameter in head.parameters())


def test_last_25_frames_are_completion_and_progress_plateaus():
    torch.manual_seed(1)
    head = Gr00tTaskStatusHead(embedding_dim=16, hidden_dim=8, num_heads=4)
    frame_index = torch.tensor([0, 74, 75, 76, 99])
    episode_length = torch.full_like(frame_index, 100)

    outputs = _status_outputs(
        head,
        torch.randn(frame_index.numel(), 4, 16),
        frame_index,
        episode_length,
    )

    torch.testing.assert_close(
        outputs.completion_target,
        torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]),
    )
    torch.testing.assert_close(
        outputs.progress_target,
        torch.tensor([0.0, 74.0 / 75.0, 1.0, 1.0, 1.0]),
    )


def test_training_rejects_missing_episode_metadata():
    head = Gr00tTaskStatusHead(embedding_dim=16, hidden_dim=8, num_heads=4)
    harness = SimpleNamespace(task_status_head=head, config=_status_config(), training=True)

    with pytest.raises(ValueError, match="requires frame_index"):
        Gr00tN1d7._get_task_status_outputs(
            harness,
            _backbone_outputs(torch.randn(2, 4, 16)),
            BatchFeature(),
        )


def test_status_loss_matches_weighted_objectives_and_backpropagates():
    torch.manual_seed(2)
    config = _status_config()
    head = Gr00tTaskStatusHead(embedding_dim=16, hidden_dim=8, num_heads=4)
    frame_index = torch.tensor([10, 74, 75, 99])
    episode_length = torch.full_like(frame_index, 100)

    outputs = _status_outputs(
        head,
        torch.randn(frame_index.numel(), 6, 16),
        frame_index,
        episode_length,
        config,
    )

    expected_completion_loss = F.binary_cross_entropy_with_logits(
        outputs.completion_logit,
        outputs.completion_target,
        pos_weight=torch.tensor(config.task_status_completion_pos_weight),
    )
    expected_progress_loss = F.smooth_l1_loss(outputs.progress, outputs.progress_target)
    expected_status_loss = (
        config.task_status_completion_loss_weight * expected_completion_loss
        + config.task_status_progress_loss_weight * expected_progress_loss
    )
    torch.testing.assert_close(outputs.completion_loss, expected_completion_loss)
    torch.testing.assert_close(outputs.progress_loss, expected_progress_loss)
    torch.testing.assert_close(outputs.task_status_loss, expected_status_loss)
    assert outputs.task_status_loss.ndim == 0
    assert torch.isfinite(outputs.task_status_loss)

    outputs.task_status_loss.backward()
    for parameter in (head.query, head.attention.in_proj_weight, head.mlp[-1].weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


class _StatusOnlyForwardHarness:
    forward = Gr00tN1d7.forward
    get_task_status = Gr00tN1d7.get_task_status
    _get_task_status_outputs = Gr00tN1d7._get_task_status_outputs

    def __init__(self) -> None:
        self.config = _status_config()
        self.training = True
        self.task_status_head = Gr00tTaskStatusHead(
            embedding_dim=16,
            hidden_dim=8,
            num_heads=4,
        )
        self.action_head = Mock(side_effect=AssertionError("action head must be skipped"))
        self._backbone_inputs = BatchFeature(data={"input_ids": torch.ones(2, 1)})
        self._action_inputs = BatchFeature(
            data={
                "frame_index": torch.tensor([74, 75]),
                "episode_length": torch.tensor([100, 100]),
            }
        )
        self.backbone = Mock(return_value=_backbone_outputs(torch.randn(2, 4, 16)))

    def prepare_input(self, inputs: dict) -> tuple[BatchFeature, BatchFeature]:
        return self._backbone_inputs, self._action_inputs


def test_status_head_only_forward_skips_action_head():
    model = _StatusOnlyForwardHarness()

    outputs = model.forward({})

    model.backbone.assert_called_once_with(model._backbone_inputs)
    model.action_head.assert_not_called()
    assert isinstance(outputs, dict)
    torch.testing.assert_close(outputs["loss"], outputs["task_status_loss"])
    assert "action_policy_loss" not in outputs


def test_get_task_status_skips_action_diffusion_head():
    model = _StatusOnlyForwardHarness()
    model.training = False
    model._action_inputs = BatchFeature()

    outputs = model.get_task_status({})

    model.action_head.assert_not_called()
    assert set(outputs) == {
        "completion_logit",
        "progress_logit",
        "completion",
        "progress",
    }
    assert not outputs.completion.requires_grad
