from typing import Any

from gr00t.eval.run_gr00t_smpl_server import SMPL_ACTION_DIMS, SmplActionPolicyWrapper
from gr00t.policy.policy import BasePolicy
import numpy as np
import pytest


class MockSmplPolicy(BasePolicy):
    def __init__(self, action: dict[str, np.ndarray]):
        super().__init__(strict=False)
        self.action = action
        self.checked_observations = 0

    def check_observation(self, observation: dict[str, Any]) -> None:
        self.checked_observations += 1

    def check_action(self, action: dict[str, Any]) -> None:
        pass

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.action, {"mock": True}

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return {}

    def get_modality_config(self):
        return {}


def _make_smpl_action(batch_size: int = 1, horizon: int = 50) -> dict[str, np.ndarray]:
    return {
        key: np.zeros((batch_size, horizon, dim), dtype=np.float32)
        for key, dim in SMPL_ACTION_DIMS.items()
    }


def test_smpl_wrapper_returns_flat_action_keys_without_batch_dim():
    policy = MockSmplPolicy(_make_smpl_action())
    wrapper = SmplActionPolicyWrapper(policy, expected_horizon=50, strict=True)

    action, info = wrapper.get_action({"observation": "unused"})

    assert info == {"mock": True}
    assert policy.checked_observations == 1
    assert set(action) == {f"action.{key}" for key in SMPL_ACTION_DIMS}
    for key, dim in SMPL_ACTION_DIMS.items():
        action_arr = action[f"action.{key}"]
        assert action_arr.shape == (50, dim)
        assert action_arr.dtype == np.float32


def test_smpl_wrapper_rejects_missing_smpl_key():
    raw_action = _make_smpl_action()
    raw_action.pop("body_quat_w")
    wrapper = SmplActionPolicyWrapper(MockSmplPolicy(raw_action), strict=False)

    with pytest.raises(KeyError, match="body_quat_w"):
        wrapper.get_action({"observation": "unused"})


def test_smpl_wrapper_rejects_batched_output():
    wrapper = SmplActionPolicyWrapper(MockSmplPolicy(_make_smpl_action(batch_size=2)), strict=False)

    with pytest.raises(ValueError, match="single-sample batches"):
        wrapper.get_action({"observation": "unused"})


def test_smpl_wrapper_validates_expected_horizon():
    wrapper = SmplActionPolicyWrapper(
        MockSmplPolicy(_make_smpl_action(horizon=40)),
        expected_horizon=50,
        strict=True,
    )

    with pytest.raises(AssertionError, match="horizon"):
        wrapper.get_action({"observation": "unused"})
