# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
from scripts.data import add_g1_endposes, add_smpl_endposes


def test_smpl_endpose_positions_preserve_root_motion():
    frame_count = 3
    joints = np.zeros((frame_count, 24, 3), dtype=np.float32)
    pelvis_positions = np.array(
        [[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0], [7.0, -8.0, 9.0]], dtype=np.float32
    )
    joints[:] = pelvis_positions[:, None, :]
    expected_positions = []
    for endpose_index, (_name, joint_index) in enumerate(add_smpl_endposes.ENDPOSE_SPECS):
        offset = np.array([endpose_index + 0.1, -endpose_index - 0.2, 0.3], dtype=np.float32)
        joints[:, joint_index] += offset
        expected_positions.append(pelvis_positions + offset)

    endposes = add_smpl_endposes.compute_endposes(
        joints, np.zeros((frame_count, 63), dtype=np.float32)
    )
    for endpose_index, expected in enumerate(expected_positions):
        start = endpose_index * add_smpl_endposes.POSE_DIM
        np.testing.assert_allclose(
            endposes[:, start : start + 3], expected, atol=1e-6
        )

        expected_base_quaternion = np.tile(
            add_smpl_endposes.SMPL_BASE_ROT_OFFSET_WXYZ, (frame_count, 1)
        )
        np.testing.assert_allclose(
            endposes[:, start + 3 : start + 7], expected_base_quaternion, atol=1e-6
        )


def test_smpl_metadata_exposes_full_pose_in_state_and_action(tmp_path):
    modality_path = tmp_path / "modality.json"
    modality_path.write_text(json.dumps({"state": {}, "action": {}}))

    add_smpl_endposes._update_modality(modality_path, dry_run=False, overwrite=False)
    modality = json.loads(modality_path.read_text())

    for group_name in ("state", "action"):
        group = modality[group_name]
        for endpose_index, (name, _joint_index) in enumerate(add_smpl_endposes.ENDPOSE_SPECS):
            entry = group[f"human_{name}_endpose"]
            assert entry["end"] - entry["start"] == 7
            assert entry["start"] == endpose_index * 7
            assert entry["original_key"] == add_smpl_endposes.ENDPOSE_FEATURE


def test_g1_metadata_uses_observation_and_action_sources(tmp_path):
    modality_path = tmp_path / "modality.json"
    modality_path.write_text(json.dumps({"state": {}, "action": {}}))

    add_g1_endposes._update_modality(modality_path, dry_run=False, overwrite=False)
    modality = json.loads(modality_path.read_text())

    for index, key in enumerate(add_g1_endposes.ENDPOSE_MODALITY_KEYS):
        state_entry = modality["state"][key]
        action_entry = modality["action"][key]
        assert state_entry["start"] == action_entry["start"] == index * 7
        assert state_entry["end"] - state_entry["start"] == 7
        assert action_entry["end"] - action_entry["start"] == 7
        assert state_entry["original_key"] == add_g1_endposes.OBSERVATION_ENDPOSE_FEATURE
        assert action_entry["original_key"] == add_g1_endposes.ACTION_ENDPOSE_FEATURE
