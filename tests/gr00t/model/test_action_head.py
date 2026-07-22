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
Test Gr00tN1d7ActionHead: flow matching forward, get_action, feature encoding.

These tests instantiate the action head directly (no backbone required)
and feed it synthetic backbone output tensors.
"""

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


def _small_config(**overrides) -> Gr00tN1d7Config:
    defaults = dict(
        backbone_embedding_dim=64,
        hidden_size=64,
        input_embedding_dim=64,
        max_state_dim=7,
        max_action_dim=7,
        action_horizon=4,
        state_history_length=1,
        num_inference_timesteps=2,
        max_num_embodiments=4,
        add_pos_embed=True,
        use_vlln=True,
        max_seq_len=32,
        use_alternate_vl_dit=False,
        attend_text_every_n_blocks=2,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        state_dropout_prob=0.0,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
        attn_dropout=0.0,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 32,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 64,
            "interleave_self_attention": True,
        },
    )
    defaults.update(overrides)
    return Gr00tN1d7Config(**defaults)


@pytest.fixture
def action_head():
    config = _small_config()
    head = Gr00tN1d7ActionHead(config)
    head.eval()
    return head, config


def _make_backbone_output(config, batch_size=2, seq_len=8):
    return BatchFeature(
        data={
            "backbone_features": torch.randn(batch_size, seq_len, config.backbone_embedding_dim),
            "backbone_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "image_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        }
    )


def _make_action_input(config, batch_size=2):
    return BatchFeature(
        data={
            "state": torch.randn(batch_size, config.state_history_length, config.max_state_dim),
            "action": torch.randn(batch_size, config.action_horizon, config.max_action_dim),
            "embodiment_id": torch.zeros(batch_size, dtype=torch.long),
            "action_mask": torch.ones(batch_size, config.action_horizon, config.max_action_dim),
        }
    )


def _has_nonzero_grad(module) -> bool:
    return any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in module.parameters()
    )


def _has_any_grad(module) -> bool:
    return any(parameter.grad is not None for parameter in module.parameters())


class TestActionHeadForward:
    """Test training forward pass."""

    def test_forward_returns_loss(self, action_head):
        head, config = action_head
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert "loss" in out
        assert out["loss"].dim() == 0
        assert torch.isfinite(out["loss"])

    def test_forward_loss_shape(self, action_head):
        head, config = action_head
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert out["action_loss"].shape == (2, config.action_horizon, config.max_action_dim)

    def test_forward_with_state_dropout(self):
        config = _small_config(state_dropout_prob=0.5)
        head = Gr00tN1d7ActionHead(config)
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert torch.isfinite(out["loss"])


class TestActionHeadGetAction:
    """Test inference (denoising loop)."""

    def test_get_action_output_shape(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config)
        del action_input["action"]  # get_action doesn't need ground-truth action
        out = head.get_action(_make_backbone_output(config), action_input)
        assert "action_pred" in out
        assert out["action_pred"].shape == (2, config.action_horizon, config.max_action_dim)

    def test_get_action_no_grad(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config)
        del action_input["action"]
        out = head.get_action(_make_backbone_output(config), action_input)
        assert not out["action_pred"].requires_grad

    def test_get_action_single_sample(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config, batch_size=1)
        del action_input["action"]
        out = head.get_action(
            _make_backbone_output(config, batch_size=1),
            action_input,
        )
        assert out["action_pred"].shape[0] == 1

    def test_combined_inference_keeps_auxiliary_slots_zero(self):
        config = _small_config(action_branch_mode="primary_eef")
        head = Gr00tN1d7ActionHead(config)
        head.eval()
        action_input = _make_action_input(config)
        del action_input["action"]
        primary_mask = torch.zeros_like(action_input.action_mask)
        primary_mask[..., :4] = 1.0
        action_input["primary_action_mask"] = primary_mask
        encoded_trajectories = []
        hook = head.action_encoder.register_forward_pre_hook(
            lambda _module, args: encoded_trajectories.append(args[0].detach().clone())
        )
        try:
            output = head.get_action(_make_backbone_output(config), action_input)
        finally:
            hook.remove()

        assert encoded_trajectories
        assert all(torch.count_nonzero(value[..., 4:]) == 0 for value in encoded_trajectories)
        assert torch.count_nonzero(output["action_pred"][..., 4:]) == 0

    def test_combined_inference_requires_primary_mask(self):
        config = _small_config(action_branch_mode="primary_eef")
        head = Gr00tN1d7ActionHead(config)
        action_input = _make_action_input(config)
        del action_input["action"]
        with pytest.raises(RuntimeError, match="requires primary_action_mask"):
            head.get_action(_make_backbone_output(config), action_input)

    def test_combined_rtc_keeps_auxiliary_slots_zero(self):
        config = _small_config(action_branch_mode="primary_eef")
        head = Gr00tN1d7ActionHead(config)
        action_input = _make_action_input(config)
        primary_mask = torch.zeros_like(action_input.action_mask)
        primary_mask[..., :4] = 1.0
        action_input["primary_action_mask"] = primary_mask
        output = head.get_action(
            _make_backbone_output(config),
            action_input,
            options={
                "action_horizon": config.action_horizon,
                "rtc_overlap_steps": 2,
                "rtc_frozen_steps": 1,
                "rtc_ramp_rate": 1.0,
            },
        )
        assert torch.count_nonzero(output["action_pred"][..., 4:]) == 0

    def test_eef_only_checkpoint_has_no_policy_action(self):
        config = _small_config(action_branch_mode="eef")
        head = Gr00tN1d7ActionHead(config)
        action_input = _make_action_input(config)
        del action_input["action"]
        with pytest.raises(RuntimeError, match="training supervision"):
            head.get_action(_make_backbone_output(config), action_input)


class TestActionHeadEncodeFeatures:
    """Test feature encoding helper."""

    def test_encode_features_shapes(self, action_head):
        head, config = action_head
        result = head._encode_features(
            _make_backbone_output(config),
            _make_action_input(config),
        )
        assert result["backbone_features"].shape == (2, 8, config.backbone_embedding_dim)
        assert result["state_features"].shape == (2, 1, config.input_embedding_dim)


class TestActionHeadTrainableParams:
    """Test parameter freezing."""

    def test_all_trainable_by_default(self, action_head):
        head, _ = action_head
        head.set_trainable_parameters(True, True, True)
        assert all(p.requires_grad for p in head.parameters())

    def test_freeze_projector(self):
        config = _small_config()
        head = Gr00tN1d7ActionHead(config)
        head.set_trainable_parameters(False, True, True)
        for p in head.state_encoder.parameters():
            assert not p.requires_grad
        for p in head.action_encoder.parameters():
            assert not p.requires_grad

    def test_freeze_diffusion(self):
        config = _small_config()
        head = Gr00tN1d7ActionHead(config)
        head.set_trainable_parameters(True, False, True)
        for p in head.model.parameters():
            assert not p.requires_grad


class TestAuxiliaryEefBranch:
    """EEF has task-specific I/O adapters but shares the DiT trunk."""

    @staticmethod
    def _set_branch_masks(action_input, primary_dims: int, include_primary: bool, include_eef: bool):
        action_mask = torch.zeros_like(action_input.action_mask)
        auxiliary_mask = torch.zeros_like(action_input.action_mask)
        if include_primary:
            action_mask[..., :primary_dims] = 1.0
        if include_eef:
            action_mask[..., primary_dims:] = 1.0
            auxiliary_mask[..., primary_dims:] = 1.0
        action_input["action_mask"] = action_mask
        action_input["auxiliary_action_mask"] = auxiliary_mask
        return action_input

    def test_eef_only_freezes_and_skips_primary_adapters(self):
        config = _small_config(action_branch_mode="eef", auxiliary_action_init_category=0)
        head = Gr00tN1d7ActionHead(config)
        head.train()
        action_input = self._set_branch_masks(
            _make_action_input(config), primary_dims=4, include_primary=False, include_eef=True
        )
        primary_calls = []
        encoder_hook = head.action_encoder.register_forward_hook(
            lambda *_args: primary_calls.append("encoder")
        )
        decoder_hook = head.action_decoder.register_forward_hook(
            lambda *_args: primary_calls.append("decoder")
        )
        try:
            output = head(_make_backbone_output(config), action_input)
            output["loss"].backward()
        finally:
            encoder_hook.remove()
            decoder_hook.remove()

        assert primary_calls == []
        assert not any(parameter.requires_grad for parameter in head.action_encoder.parameters())
        assert not any(parameter.requires_grad for parameter in head.action_decoder.parameters())
        assert not _has_any_grad(head.action_encoder)
        assert not _has_any_grad(head.action_decoder)
        assert _has_nonzero_grad(head.eef_action_encoder)
        assert _has_nonzero_grad(head.eef_action_decoder)
        assert _has_nonzero_grad(head.model)

    def test_primary_only_skips_eef_adapters(self):
        config = _small_config(action_branch_mode="primary_eef")
        head = Gr00tN1d7ActionHead(config)
        head.train()
        action_input = self._set_branch_masks(
            _make_action_input(config), primary_dims=4, include_primary=True, include_eef=False
        )
        eef_calls = []
        primary_inputs = []
        primary_hook = head.action_encoder.register_forward_pre_hook(
            lambda _module, args: primary_inputs.append(args[0].detach().clone())
        )
        encoder_hook = head.eef_action_encoder.register_forward_hook(
            lambda *_args: eef_calls.append("encoder")
        )
        decoder_hook = head.eef_action_decoder.register_forward_hook(
            lambda *_args: eef_calls.append("decoder")
        )
        try:
            output = head(_make_backbone_output(config), action_input)
            output["loss"].backward()
        finally:
            primary_hook.remove()
            encoder_hook.remove()
            decoder_hook.remove()

        assert eef_calls == []
        assert len(primary_inputs) == 1
        assert torch.count_nonzero(primary_inputs[0][..., 4:]) == 0
        assert _has_nonzero_grad(head.action_encoder)
        assert _has_nonzero_grad(head.action_decoder)
        assert not _has_any_grad(head.eef_action_encoder)
        assert not _has_any_grad(head.eef_action_decoder)
        assert _has_nonzero_grad(head.model)

    def test_eef_only_batch_in_combined_mode_skips_primary_adapters(self):
        config = _small_config(action_branch_mode="primary_eef")
        head = Gr00tN1d7ActionHead(config)
        head.train()
        action_input = self._set_branch_masks(
            _make_action_input(config), primary_dims=4, include_primary=False, include_eef=True
        )
        output = head(_make_backbone_output(config), action_input)
        output["loss"].backward()

        # The modules stay trainable for a later primary batch, but this batch
        # must leave grad=None so AdamW cannot decay them.
        assert all(parameter.requires_grad for parameter in head.action_encoder.parameters())
        assert all(parameter.requires_grad for parameter in head.action_decoder.parameters())
        assert not _has_any_grad(head.action_encoder)
        assert not _has_any_grad(head.action_decoder)
        assert _has_nonzero_grad(head.eef_action_encoder)
        assert _has_nonzero_grad(head.eef_action_decoder)
        assert _has_nonzero_grad(head.model)

    def test_combined_updates_both_adapters_and_shared_dit(self):
        config = _small_config(action_branch_mode="primary_eef")
        head = Gr00tN1d7ActionHead(config)
        head.train()
        action_input = self._set_branch_masks(
            _make_action_input(config), primary_dims=4, include_primary=True, include_eef=True
        )
        output = head(_make_backbone_output(config), action_input)
        output["loss"].backward()

        assert _has_nonzero_grad(head.action_encoder)
        assert _has_nonzero_grad(head.action_decoder)
        assert _has_nonzero_grad(head.eef_action_encoder)
        assert _has_nonzero_grad(head.eef_action_decoder)
        assert _has_nonzero_grad(head.model)

    def test_combined_losses_are_normalized_separately(self):
        config = _small_config(
            action_branch_mode="primary_eef", auxiliary_action_loss_weight=0.25
        )
        head = Gr00tN1d7ActionHead(config)
        head.train()
        action_input = self._set_branch_masks(
            _make_action_input(config), primary_dims=4, include_primary=True, include_eef=True
        )
        # Make the masks deliberately unequal and non-unit weighted.
        action_input.action_mask[..., :4] = 0.5
        action_input.action_mask[:, :2, 4:] = 0.0
        output = head(_make_backbone_output(config), action_input)

        auxiliary_selector = action_input.auxiliary_action_mask
        auxiliary_mask = action_input.action_mask * auxiliary_selector
        primary_mask = action_input.action_mask * (1 - auxiliary_selector)
        expected_primary = (
            (output["action_loss"] * (primary_mask > 0)).sum()
            / (primary_mask.sum() + 1e-6)
        )
        expected_auxiliary = (
            (output["action_loss"] * (auxiliary_mask > 0)).sum()
            / (auxiliary_mask.sum() + 1e-6)
        )
        torch.testing.assert_close(output["primary_loss"], expected_primary)
        torch.testing.assert_close(output["auxiliary_loss"], expected_auxiliary)
        torch.testing.assert_close(
            output["loss"], expected_primary + 0.25 * expected_auxiliary
        )

    @pytest.mark.parametrize(
        ("branch_mode", "include_primary", "include_eef", "message"),
        [
            ("eef", True, False, "EEF-only branch mode"),
            ("primary", False, True, "Primary-only branch mode"),
        ],
    )
    def test_branch_mode_rejects_mismatched_supervision(
        self, branch_mode, include_primary, include_eef, message
    ):
        config = _small_config(action_branch_mode=branch_mode)
        head = Gr00tN1d7ActionHead(config)
        action_input = self._set_branch_masks(
            _make_action_input(config),
            primary_dims=4,
            include_primary=include_primary,
            include_eef=include_eef,
        )
        with pytest.raises(ValueError, match=message):
            head(_make_backbone_output(config), action_input)

    def test_eef_branch_initializes_from_one_primary_category(self):
        config = _small_config(
            action_branch_mode="eef",
            auxiliary_action_init_category=2,
        )
        head = Gr00tN1d7ActionHead(config)
        with torch.no_grad():
            for parameter in head.action_encoder.parameters():
                parameter.copy_(torch.randn_like(parameter))
            for parameter in head.action_decoder.parameters():
                parameter.copy_(torch.randn_like(parameter))
        head.initialize_eef_branch_from_primary()

        for target, source in (
            (head.eef_action_encoder.W1, head.action_encoder.W1),
            (head.eef_action_encoder.W2, head.action_encoder.W2),
            (head.eef_action_encoder.W3, head.action_encoder.W3),
            (head.eef_action_decoder.layer1, head.action_decoder.layer1),
            (head.eef_action_decoder.layer2, head.action_decoder.layer2),
        ):
            torch.testing.assert_close(target.W[0], source.W[2])
            torch.testing.assert_close(target.b[0], source.b[2])
