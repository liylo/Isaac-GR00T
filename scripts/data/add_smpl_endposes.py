#!/usr/bin/env python3

"""Add SMPL hand/foot end poses without pelvis subtraction or retargeting.

The added (or validated) feature is ``teleop.human_endpose_abs`` with shape ``[28]``:

    left_hand xyz+quat, right_hand xyz+quat, left_foot xyz+quat, right_foot xyz+quat

Quaternions are stored as ``qw, qx, qy, qz``. Positions are copied directly from
the stored SMPL wrist/foot joints, and orientations are derived from the SMPL
kinematic chain in the same coordinate convention. No pelvis centering, body-size
scaling, robot retargeting, or SMPL-to-G1 local-axis calibration is applied. This
preserves whole-body motion in the human end-pose trajectory. The same column is
exposed as observation and action aliases; the human endpose training config
samples the action aliases at t+1..t+40 because this dataset's retargeted action
stream is lead-1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


ENDPOSE_FEATURE = "teleop.human_endpose_abs"
SMPL_JOINTS_FEATURE = "teleop.smpl_joints"
SMPL_POSE_FEATURE = "teleop.smpl_pose"

POSE_DIM = 7
ENDPOSE_SPECS = (
    ("left_hand", 20),  # SMPL left wrist; robot target is left_wrist_yaw_link.
    ("right_hand", 21),  # SMPL right wrist; robot target is right_wrist_yaw_link.
    ("left_foot", 10),  # SMPL left foot; robot target is left_ankle_roll_link.
    ("right_foot", 11),  # SMPL right foot; robot target is right_ankle_roll_link.
)

# First 22 SMPL/SMPL-H parents from gear_sonic/data/human/human_joints_info.pkl.
SMPL_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    dtype=np.int64,
)

# Matches gear_sonic.utils.inference.smpl_stream_utils._SMPL_BASE_ROT_OFFSET.
SMPL_BASE_ROT_OFFSET_WXYZ = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

def _as_quat_wxyz(rotation: Rotation) -> np.ndarray:
    quat_xyzw = rotation.as_quat()
    quat_wxyz = quat_xyzw[..., [3, 0, 1, 2]].astype(np.float32)
    norms = np.linalg.norm(quat_wxyz, axis=-1, keepdims=True)
    quat_wxyz = quat_wxyz / np.maximum(norms, 1e-8)
    quat_wxyz[quat_wxyz[..., 0] < 0.0] *= -1.0
    return quat_wxyz


def _endpose_names() -> list[str]:
    names = []
    for name, _joint_index in ENDPOSE_SPECS:
        names.extend(
            [
                f"{name}_x",
                f"{name}_y",
                f"{name}_z",
                f"{name}_qw",
                f"{name}_qx",
                f"{name}_qy",
                f"{name}_qz",
            ]
        )
    return names


def _read_vector_column(table: pa.Table, key: str, width: int) -> np.ndarray:
    if key not in table.column_names:
        raise KeyError(f"Missing required column {key!r}")
    values = np.asarray(table[key].to_pylist(), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"{key!r} must have shape [N, {width}], got {values.shape}")
    return values


def _compute_smpl_root_local_rotations(smpl_pose: np.ndarray) -> dict[int, Rotation]:
    """Return root-local rotations for the first 22 SMPL joints.

    ``smpl_pose`` contains local axis-angle rotations for joints 1..21. The
    stored ``teleop.smpl_joints`` frame additionally includes the SMPL base
    rotation offset, so we apply the same offset to orientations.
    """

    num_frames = smpl_pose.shape[0]
    pose_aa = smpl_pose.reshape(num_frames, 21, 3)

    identity_quat_xyzw = np.broadcast_to(
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), (num_frames, 4)
    )
    global_rots: list[Rotation | None] = [None] * len(SMPL_PARENTS)
    global_rots[0] = Rotation.from_quat(identity_quat_xyzw)

    for joint_index in range(1, len(SMPL_PARENTS)):
        parent_index = int(SMPL_PARENTS[joint_index])
        parent_rot = global_rots[parent_index]
        if parent_rot is None:
            raise RuntimeError(f"Parent rotation for SMPL joint {joint_index} was not computed")
        local_rot = Rotation.from_rotvec(pose_aa[:, joint_index - 1, :].astype(np.float64))
        global_rots[joint_index] = parent_rot * local_rot

    base_quat_xyzw = np.broadcast_to(
        SMPL_BASE_ROT_OFFSET_WXYZ[[1, 2, 3, 0]].astype(np.float64), (num_frames, 4)
    )
    base_rot = Rotation.from_quat(base_quat_xyzw)
    root_local_rotations = {}
    for _name, joint_index in ENDPOSE_SPECS:
        root_local_rotations[joint_index] = base_rot * global_rots[joint_index]
    return root_local_rotations


def compute_endposes(smpl_joints: np.ndarray, smpl_pose: np.ndarray) -> np.ndarray:
    if smpl_joints.ndim != 3 or smpl_joints.shape[1] < 22 or smpl_joints.shape[2] != 3:
        raise ValueError(f"smpl_joints must have shape [N, >=22, 3], got {smpl_joints.shape}")
    if smpl_pose.ndim != 2 or smpl_pose.shape[1] != 63:
        raise ValueError(f"smpl_pose must have shape [N, 63], got {smpl_pose.shape}")
    if smpl_joints.shape[0] != smpl_pose.shape[0]:
        raise ValueError(
            f"smpl_joints and smpl_pose frame counts differ: {smpl_joints.shape[0]} vs {smpl_pose.shape[0]}"
        )

    joint_rots = _compute_smpl_root_local_rotations(smpl_pose)
    endposes = np.zeros((smpl_pose.shape[0], len(ENDPOSE_SPECS) * POSE_DIM), dtype=np.float32)
    for endpose_index, (_name, joint_index) in enumerate(ENDPOSE_SPECS):
        start = endpose_index * POSE_DIM
        endposes[:, start : start + 3] = smpl_joints[:, joint_index, :]
        endposes[:, start + 3 : start + 7] = _as_quat_wxyz(joint_rots[joint_index])
    return endposes


def _endpose_error(existing: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    position_indices = []
    rotation_errors = []
    for endpose_index in range(len(ENDPOSE_SPECS)):
        start = endpose_index * POSE_DIM
        position_indices.extend(range(start, start + 3))
        # Promote before normalization/dot: float32 roundoff close to |dot|=1
        # is strongly amplified by arccos and can look like a milliradian error.
        existing_quat = existing[:, start + 3 : start + 7].astype(np.float64)
        expected_quat = expected[:, start + 3 : start + 7].astype(np.float64)
        existing_quat = existing_quat / np.maximum(
            np.linalg.norm(existing_quat, axis=1, keepdims=True), 1e-8
        )
        expected_quat = expected_quat / np.maximum(
            np.linalg.norm(expected_quat, axis=1, keepdims=True), 1e-8
        )
        dot = np.abs(np.sum(existing_quat * expected_quat, axis=1))
        rotation_errors.append(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))
    position_error = float(
        np.max(np.abs(existing[:, position_indices] - expected[:, position_indices]))
    )
    rotation_error = float(np.max(np.concatenate(rotation_errors)))
    return position_error, rotation_error


def _compute_statistics(values: np.ndarray) -> dict[str, list[float]]:
    values = values.astype(np.float32, copy=False)
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _json_load(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _json_dump(path: Path, data: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        return
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")
    tmp_path.replace(path)


def _update_info(info_path: Path, dry_run: bool, overwrite: bool) -> None:
    info = _json_load(info_path)
    features = info.setdefault("features", {})
    if ENDPOSE_FEATURE in features and not overwrite:
        existing = features[ENDPOSE_FEATURE]
        expected_shape = [len(ENDPOSE_SPECS) * POSE_DIM]
        if existing.get("shape") != expected_shape:
            raise ValueError(
                f"{ENDPOSE_FEATURE!r} already exists in {info_path} with shape "
                f"{existing.get('shape')}, expected {expected_shape}; pass --overwrite"
            )
    else:
        features[ENDPOSE_FEATURE] = {
            "dtype": "float32",
            "shape": [len(ENDPOSE_SPECS) * POSE_DIM],
            "names": _endpose_names(),
            "coordinate_frame": "stored_smpl_joint_frame",
            "quaternion_order": "wxyz",
            "pelvis_centered": False,
            "robot_axis_calibration": False,
        }
    _json_dump(info_path, info, dry_run)


def _update_stats(stats_path: Path, stats: dict[str, list[float]], dry_run: bool) -> None:
    all_stats = _json_load(stats_path) if stats_path.exists() else {}
    all_stats[ENDPOSE_FEATURE] = stats
    _json_dump(stats_path, all_stats, dry_run)


def _update_modality(modality_path: Path, dry_run: bool, overwrite: bool) -> None:
    modality = _json_load(modality_path)
    state = modality.setdefault("state", {})
    action = modality.setdefault("action", {})

    entries: dict[str, dict[str, Any]] = {}
    pose_entries: dict[str, dict[str, Any]] = {}
    for endpose_index, (name, _joint_index) in enumerate(ENDPOSE_SPECS):
        start = endpose_index * POSE_DIM
        pose_entries[f"human_{name}_endpose"] = {
            "start": start,
            "end": start + POSE_DIM,
            "original_key": ENDPOSE_FEATURE,
            "rotation_type": "quaternion",
        }
        entries[f"human_{name}_pos"] = {
            "start": start,
            "end": start + 3,
            "original_key": ENDPOSE_FEATURE,
        }
        entries[f"human_{name}_abs_quat"] = {
            "start": start + 3,
            "end": start + 7,
            "original_key": ENDPOSE_FEATURE,
            "rotation_type": "quaternion",
        }

    for key, value in entries.items():
        if key in state and not overwrite:
            continue
        state[key] = value
    for group in (state, action):
        for key, value in pose_entries.items():
            if key not in group or overwrite:
                group[key] = value

    _json_dump(modality_path, modality, dry_run)


def _process_parquet(path: Path, dry_run: bool, overwrite: bool) -> np.ndarray:
    table = pq.read_table(path)
    smpl_joints_flat = _read_vector_column(table, SMPL_JOINTS_FEATURE, 72)
    smpl_pose = _read_vector_column(table, SMPL_POSE_FEATURE, 63)
    endposes = compute_endposes(smpl_joints_flat.reshape(-1, 24, 3), smpl_pose)

    if ENDPOSE_FEATURE in table.column_names and not overwrite:
        existing = _read_vector_column(table, ENDPOSE_FEATURE, len(ENDPOSE_SPECS) * POSE_DIM)
        position_error, rotation_error = _endpose_error(existing, endposes)
        if position_error > 1e-6 or rotation_error > 1e-5:
            raise ValueError(
                f"{path}: existing {ENDPOSE_FEATURE} was generated by a different formula "
                f"(position max error={position_error:.3e} m, rotation max error="
                f"{rotation_error:.3e} rad); rerun with --overwrite"
            )
        return existing

    if dry_run:
        return endposes

    if ENDPOSE_FEATURE in table.column_names:
        table = table.drop([ENDPOSE_FEATURE])

    endpose_column = pa.array(endposes.tolist(), type=pa.list_(pa.float32()))
    updated = table.append_column(ENDPOSE_FEATURE, endpose_column)

    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        pq.write_table(updated, tmp_path)
        tmp_path.chmod(path.stat().st_mode)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return endposes


def _parquet_paths(dataset_path: Path) -> list[Path]:
    return sorted((dataset_path / "data").glob("chunk-*/*.parquet"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path", type=Path, help="LeRobot dataset root")
    parser.add_argument(
        "--dry-run", action="store_true", help="Compute and validate without writes"
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing endpose field")
    parser.add_argument(
        "--max-files", type=int, default=0, help="Only process the first N parquet files"
    )
    parser.add_argument(
        "--skip-modality",
        action="store_true",
        help="Do not add state/action entries to meta/modality.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path
    meta_path = dataset_path / "meta"
    info_path = meta_path / "info.json"
    stats_path = meta_path / "stats.json"
    modality_path = meta_path / "modality.json"

    if not dataset_path.is_dir():
        raise FileNotFoundError(dataset_path)
    for path in [info_path, modality_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    parquet_paths = _parquet_paths(dataset_path)
    if args.max_files:
        parquet_paths = parquet_paths[: args.max_files]
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {dataset_path / 'data'}")

    all_endposes = []
    for index, parquet_path in enumerate(parquet_paths, start=1):
        endposes = _process_parquet(parquet_path, dry_run=args.dry_run, overwrite=args.overwrite)
        all_endposes.append(endposes)
        if index == 1 or index % 100 == 0 or index == len(parquet_paths):
            print(f"[{index}/{len(parquet_paths)}] {parquet_path.name}: {endposes.shape}")

    values = np.concatenate(all_endposes, axis=0)
    stats = _compute_statistics(values)

    if args.max_files:
        print("--max-files was used; metadata was not updated because stats are partial.")
    else:
        _update_info(info_path, dry_run=args.dry_run, overwrite=args.overwrite)
        _update_stats(stats_path, stats, dry_run=args.dry_run)
        if not args.skip_modality:
            _update_modality(modality_path, dry_run=args.dry_run, overwrite=args.overwrite)

    mode = "dry-run" if args.dry_run else "updated"
    print(
        f"{mode}: {len(parquet_paths)} parquet files, {values.shape[0]} frames, "
        f"{ENDPOSE_FEATURE} shape [{values.shape[1]}]"
    )
    print("first mean values:", np.asarray(stats["mean"][:7], dtype=np.float32).tolist())


if __name__ == "__main__":
    main()
