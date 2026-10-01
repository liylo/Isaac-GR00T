# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint compatibility tests for adding a fresh task-status head."""

from types import SimpleNamespace

from gr00t.model.gr00t_n1d7 import setup as setup_impl
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tTaskStatusHead
from gr00t.model.gr00t_n1d7.setup import TASK_STATUS_HEAD_KEYS, Gr00tN1d7Pipeline
import pytest
import torch


class _ActionHead:
    mask_token = None


def test_status_key_allowlist_matches_model() -> None:
    head = Gr00tTaskStatusHead(embedding_dim=16, hidden_dim=8, num_heads=4)
    actual = {f"task_status_head.{key}" for key in head.state_dict()}
    assert actual == TASK_STATUS_HEAD_KEYS


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.task_status_head = torch.nn.Linear(2, 2)
        self.action_head = _ActionHead()
        self.config = SimpleNamespace(
            action_branch_mode="primary",
            auxiliary_action_init_category=11,
        )


def _pipeline(tmp_path) -> Gr00tN1d7Pipeline:
    pipeline = object.__new__(Gr00tN1d7Pipeline)
    pipeline.save_cfg_dir = tmp_path
    pipeline.transformers_loading_kwargs = {}
    pipeline.config = SimpleNamespace(
        training=SimpleNamespace(
            start_from_checkpoint="checkpoint-without-status",
            skip_weight_loading=False,
        ),
        model=SimpleNamespace(
            tune_llm=False,
            tune_visual=False,
            tune_projector=False,
            tune_diffusion_model=False,
            tune_vlln=False,
            tune_top_llm_layers=0,
            model_name="local-backbone",
            state_dropout_prob=0.0,
            backbone_trainable_params_fp32=True,
            load_bf16=False,
            action_branch_mode="primary",
            auxiliary_action_loss_weight=0.5,
            auxiliary_action_init_category=11,
            task_status_head_enabled=True,
            task_status_head_only=True,
            task_status_hidden_dim=512,
            task_status_num_heads=8,
            task_status_dropout=0.0,
            task_status_completion_window_frames=25,
            task_status_completion_pos_weight=42.0,
            task_status_loss_weight=1.0,
            task_status_completion_loss_weight=1.0,
            task_status_progress_loss_weight=1.0,
            task_status_completion_threshold=0.5,
            task_status_progress_threshold=0.9,
            task_status_consecutive_steps=1,
        ),
    )
    return pipeline


def _install_loader(monkeypatch, missing_keys):
    model = _Model()
    with torch.no_grad():
        model.task_status_head.weight.fill_(torch.nan)
        model.task_status_head.bias.fill_(torch.nan)
    monkeypatch.setattr(
        setup_impl.AutoModel,
        "from_pretrained",
        lambda *_args, **_kwargs: (model, {"missing_keys": missing_keys}),
    )
    monkeypatch.setattr(setup_impl, "get_rank", lambda: 1)
    return model


def test_complete_missing_status_head_is_initialized(monkeypatch, tmp_path) -> None:
    model = _install_loader(
        monkeypatch,
        ["task_status_head.weight", "task_status_head.bias"],
    )

    assert _pipeline(tmp_path)._create_model() is model
    assert all(torch.isfinite(parameter).all() for parameter in model.task_status_head.parameters())


def test_partial_missing_status_head_is_rejected(monkeypatch, tmp_path) -> None:
    _install_loader(monkeypatch, ["task_status_head.weight"])

    with pytest.raises(RuntimeError, match="Missing keys"):
        _pipeline(tmp_path)._create_model()


def test_non_finite_status_checkpoint_is_rejected(monkeypatch, tmp_path) -> None:
    _install_loader(monkeypatch, [])

    with pytest.raises(RuntimeError, match="non-finite parameters"):
        _pipeline(tmp_path)._create_model()


def test_complete_unexpected_status_head_can_be_discarded(monkeypatch, tmp_path) -> None:
    model = _Model()
    model.task_status_head = None
    monkeypatch.setattr(
        setup_impl.AutoModel,
        "from_pretrained",
        lambda *_args, **_kwargs: (
            model,
            {"unexpected_keys": sorted(TASK_STATUS_HEAD_KEYS)},
        ),
    )
    monkeypatch.setattr(setup_impl, "get_rank", lambda: 1)
    pipeline = _pipeline(tmp_path)
    pipeline.config.model.task_status_head_enabled = False
    pipeline.config.model.task_status_head_only = False

    assert pipeline._create_model() is model
