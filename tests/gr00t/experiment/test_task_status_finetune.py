# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for task-status finetuning helpers."""

import json

from gr00t.experiment.launch_finetune import compute_completion_pos_weight
import pytest


def test_compute_completion_pos_weight(tmp_path) -> None:
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    episodes_path = meta_dir / "episodes.jsonl"
    episodes_path.write_text(
        "".join(
            json.dumps({"episode_index": index, "length": length}) + "\n"
            for index, length in enumerate((100, 200))
        )
    )

    # 25 positives per episode, 250 negatives in total.
    assert compute_completion_pos_weight(str(tmp_path), window_frames=25) == 5.0


def test_compute_completion_pos_weight_rejects_invalid_window(tmp_path) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        compute_completion_pos_weight(str(tmp_path), window_frames=0)
