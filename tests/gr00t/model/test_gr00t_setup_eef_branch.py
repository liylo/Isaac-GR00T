# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Targeted checkpoint-loading tests for the auxiliary EEF action branch."""

from types import SimpleNamespace

from gr00t.model.gr00t_n1d7 import setup as setup_impl
from gr00t.model.gr00t_n1d7.setup import EEF_ACTION_BRANCH_KEYS, Gr00tN1d7Pipeline
import pytest
import torch


class _FakeActionHead:
    mask_token = None

    def __init__(self):
        self.eef_initializations = 0

    def initialize_eef_branch_from_primary(self):
        self.eef_initializations += 1


class _FakeModel(torch.nn.Module):
    def __init__(self, branch_mode):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(
            action_branch_mode=branch_mode,
            auxiliary_action_init_category=11,
        )
        self.action_head = _FakeActionHead()


def _make_pipeline(tmp_path, branch_mode="eef", auxiliary_weight=0.25):
    pipeline = object.__new__(Gr00tN1d7Pipeline)
    pipeline.save_cfg_dir = tmp_path
    pipeline.transformers_loading_kwargs = {}
    pipeline.config = SimpleNamespace(
        training=SimpleNamespace(start_from_checkpoint="old-checkpoint", skip_weight_loading=False),
        model=SimpleNamespace(
            tune_llm=False,
            tune_visual=False,
            tune_projector=True,
            tune_diffusion_model=True,
            tune_vlln=True,
            model_name="local-backbone",
            state_dropout_prob=0.0,
            backbone_trainable_params_fp32=True,
            load_bf16=False,
            action_branch_mode=branch_mode,
            auxiliary_action_loss_weight=auxiliary_weight,
            auxiliary_action_init_category=11,
        ),
    )
    return pipeline


def _install_fake_loader(monkeypatch, loading_info, branch_mode, captured):
    model = _FakeModel(branch_mode)

    def fake_from_pretrained(*args, **kwargs):
        captured.update(kwargs)
        return model, loading_info

    monkeypatch.setattr(setup_impl.AutoModel, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(setup_impl, "get_rank", lambda: 1)
    return model


def test_old_checkpoint_initializes_complete_missing_eef_branch(monkeypatch, tmp_path):
    captured = {}
    model = _install_fake_loader(
        monkeypatch,
        {"missing_keys": sorted(EEF_ACTION_BRANCH_KEYS)},
        branch_mode="eef",
        captured=captured,
    )
    pipeline = _make_pipeline(tmp_path, branch_mode="eef", auxiliary_weight=0.25)

    assert pipeline._create_model() is model
    assert model.action_head.eef_initializations == 1
    assert captured["action_branch_mode"] == "eef"
    assert captured["auxiliary_action_loss_weight"] == 0.25
    assert captured["auxiliary_action_init_category"] == 11


def test_partial_missing_eef_branch_is_rejected(monkeypatch, tmp_path):
    captured = {}
    model = _install_fake_loader(
        monkeypatch,
        {"missing_keys": [next(iter(EEF_ACTION_BRANCH_KEYS))]},
        branch_mode="eef",
        captured=captured,
    )
    pipeline = _make_pipeline(tmp_path, branch_mode="eef")

    with pytest.raises(RuntimeError, match="Missing keys"):
        pipeline._create_model()
    assert model.action_head.eef_initializations == 0


def test_primary_transfer_discards_only_a_complete_eef_branch(monkeypatch, tmp_path):
    captured = {}
    model = _install_fake_loader(
        monkeypatch,
        {"unexpected_keys": sorted(EEF_ACTION_BRANCH_KEYS)},
        branch_mode="primary",
        captured=captured,
    )
    pipeline = _make_pipeline(tmp_path, branch_mode="primary")

    assert pipeline._create_model() is model
    assert captured["action_branch_mode"] == "primary"


@pytest.mark.parametrize(
    "unexpected_keys",
    [
        [next(iter(EEF_ACTION_BRANCH_KEYS))],
        sorted(EEF_ACTION_BRANCH_KEYS) + ["action_head.unrelated.weight"],
    ],
)
def test_primary_transfer_rejects_partial_or_extra_unexpected_keys(
    monkeypatch, tmp_path, unexpected_keys
):
    captured = {}
    _install_fake_loader(
        monkeypatch,
        {"unexpected_keys": unexpected_keys},
        branch_mode="primary",
        captured=captured,
    )
    pipeline = _make_pipeline(tmp_path, branch_mode="primary")

    with pytest.raises(RuntimeError, match="Unexpected keys"):
        pipeline._create_model()
