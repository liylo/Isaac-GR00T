# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for turning task-status predictions into a stable done signal."""

from __future__ import annotations

import numpy as np


class TaskStatusDoneDetector:
    """Require consecutive completion/progress votes before declaring done.

    Votes are counted per policy invocation, not per camera frame. A failed vote
    resets the corresponding batch element's streak.
    """

    def __init__(
        self,
        completion_threshold: float = 0.5,
        progress_threshold: float = 0.9,
        consecutive_steps: int = 3,
    ) -> None:
        for name, value in (
            ("completion_threshold", completion_threshold),
            ("progress_threshold", progress_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if consecutive_steps <= 0:
            raise ValueError(f"consecutive_steps must be positive, got {consecutive_steps}")

        self.completion_threshold = completion_threshold
        self.progress_threshold = progress_threshold
        self.consecutive_steps = consecutive_steps
        self._streak: np.ndarray | None = None

    def reset(self) -> None:
        self._streak = None

    def update(
        self,
        completion: float | np.ndarray,
        progress: float | np.ndarray,
    ) -> bool | np.ndarray:
        completion_array = np.asarray(completion, dtype=np.float32)
        progress_array = np.asarray(progress, dtype=np.float32)
        if completion_array.shape != progress_array.shape:
            raise ValueError(
                "completion and progress must have the same shape, got "
                f"{completion_array.shape} and {progress_array.shape}"
            )

        scalar_input = completion_array.ndim == 0
        completion_array = np.atleast_1d(completion_array)
        progress_array = np.atleast_1d(progress_array)
        if self._streak is None or self._streak.shape != completion_array.shape:
            self._streak = np.zeros(completion_array.shape, dtype=np.int64)

        positive = (
            np.isfinite(completion_array)
            & np.isfinite(progress_array)
            & (completion_array >= self.completion_threshold)
            & (progress_array >= self.progress_threshold)
        )
        self._streak = np.where(positive, self._streak + 1, 0)
        done = self._streak >= self.consecutive_steps
        return bool(done[0]) if scalar_input else done
