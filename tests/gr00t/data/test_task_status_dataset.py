# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for terminal-frame metadata used by the task-status objective."""

from types import SimpleNamespace

from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
import numpy as np
import pandas as pd


def _bare_dataset(*, allow_padding: bool) -> ShardedSingleStepDataset:
    dataset = object.__new__(ShardedSingleStepDataset)
    dataset.allow_padding = allow_padding
    dataset.max_action_delta_index = 39
    dataset.episode_loader = SimpleNamespace(get_episode_length=lambda _index: 100)
    return dataset


def test_padding_keeps_terminal_observations() -> None:
    assert _bare_dataset(allow_padding=False).get_effective_episode_length(0) == 61
    assert _bare_dataset(allow_padding=True).get_effective_episode_length(0) == 100


def test_get_datapoint_adds_episode_position(monkeypatch) -> None:
    dataset = _bare_dataset(allow_padding=True)
    dataset.modality_configs = {}
    dataset.embodiment_tag = SimpleNamespace(value="unitree_g1_sonic")
    dataset.processor = lambda _messages: {"state": np.zeros((1, 1), dtype=np.float32)}
    episode = pd.DataFrame(index=range(100))

    monkeypatch.setattr(
        "gr00t.data.dataset.sharded_single_step_dataset.extract_step_data",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    datapoint = dataset.get_datapoint(episode, step_index=99)

    assert datapoint["frame_index"].dtype == np.int64
    assert datapoint["episode_length"].dtype == np.int64
    assert datapoint["frame_index"].item() == 99
    assert datapoint["episode_length"].item() == 100
