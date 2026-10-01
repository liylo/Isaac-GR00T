# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for converting task-status predictions into a stable done signal."""

from gr00t.policy.task_status import TaskStatusDoneDetector
import numpy as np
import pytest


def test_requires_consecutive_votes_and_failed_vote_resets_streak():
    detector = TaskStatusDoneDetector(
        completion_threshold=0.5,
        progress_threshold=0.9,
        consecutive_steps=3,
    )

    assert detector.update(completion=0.5, progress=0.9) is False
    assert detector.update(completion=0.8, progress=0.95) is False
    assert detector.update(completion=0.49, progress=0.99) is False

    assert detector.update(completion=0.7, progress=0.95) is False
    assert detector.update(completion=0.7, progress=0.95) is False
    assert detector.update(completion=0.7, progress=0.95) is True


def test_reset_clears_streak():
    detector = TaskStatusDoneDetector(consecutive_steps=2)

    assert detector.update(completion=1.0, progress=1.0) is False
    detector.reset()

    assert detector.update(completion=1.0, progress=1.0) is False
    assert detector.update(completion=1.0, progress=1.0) is True


def test_batch_elements_keep_independent_streaks():
    detector = TaskStatusDoneDetector(consecutive_steps=2)

    first = detector.update(
        completion=np.array([0.8, 0.8, 0.1]),
        progress=np.array([0.95, 0.95, 0.95]),
    )
    second = detector.update(
        completion=np.array([0.8, 0.1, 0.8]),
        progress=np.array([0.95, 0.95, 0.95]),
    )
    third = detector.update(
        completion=np.array([0.1, 0.8, 0.8]),
        progress=np.array([0.95, 0.95, 0.95]),
    )

    np.testing.assert_array_equal(first, np.array([False, False, False]))
    np.testing.assert_array_equal(second, np.array([True, False, False]))
    np.testing.assert_array_equal(third, np.array([False, False, True]))
    assert first.dtype == np.bool_


@pytest.mark.parametrize(
    ("completion", "progress"),
    [
        (np.nan, 1.0),
        (1.0, np.nan),
    ],
)
def test_nan_vote_resets_streak(completion, progress):
    detector = TaskStatusDoneDetector(consecutive_steps=2)

    assert detector.update(completion=1.0, progress=1.0) is False
    assert detector.update(completion=completion, progress=progress) is False
    assert detector.update(completion=1.0, progress=1.0) is False
    assert detector.update(completion=1.0, progress=1.0) is True


def test_mismatched_shapes_raise_value_error():
    detector = TaskStatusDoneDetector()

    with pytest.raises(ValueError, match="must have the same shape"):
        detector.update(
            completion=np.array([0.8, 0.9]),
            progress=np.array([0.95]),
        )


def test_changing_batch_shape_clears_existing_streaks():
    detector = TaskStatusDoneDetector(consecutive_steps=2)

    np.testing.assert_array_equal(
        detector.update(
            completion=np.array([1.0, 1.0]),
            progress=np.array([1.0, 1.0]),
        ),
        np.array([False, False]),
    )
    np.testing.assert_array_equal(
        detector.update(
            completion=np.array([1.0]),
            progress=np.array([1.0]),
        ),
        np.array([False]),
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"completion_threshold": -0.1}, "completion_threshold"),
        ({"completion_threshold": 1.1}, "completion_threshold"),
        ({"progress_threshold": -0.1}, "progress_threshold"),
        ({"progress_threshold": 1.1}, "progress_threshold"),
        ({"consecutive_steps": 0}, "consecutive_steps"),
    ],
)
def test_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TaskStatusDoneDetector(**kwargs)
