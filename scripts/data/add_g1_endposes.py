#!/usr/bin/env python3
"""Add pelvis-local G1 hand/foot end poses to a LeRobot dataset.

Two float64[28] features are added:

* ``observation.robot_endpose_abs`` from ``observation.state``
* ``action.robot_endpose_abs`` from ``action.wbc``

Each feature is ordered as left hand, right hand, left foot, right foot. Every
entry is ``x, y, z, qw, qx, qy, qz``. Despite the historic ``_abs`` suffix,
the robot poses are expressed in the fixed G1 pelvis frame. Human supervision
instead preserves the stored SMPL end-pose trajectory without pelvis subtraction;
each stream is converted to a change relative to its own current observation pose.

The selected G1 frames deliberately mirror the SMPL end-pose mapping used by
``add_smpl_endposes.py``: wrist joints map to wrist-yaw links and foot joints
map to ankle-roll links. In particular, palm links are not used because they
are offset from the human wrist reference point.

The default limit handling matches the SONIC data exporter that produced
``observation.eef_state``. Run this with the WholeBodyControl inference
environment, which provides Pinocchio, for example:

    PYTHONPATH=/data/liuyilong/GR00T-WholeBodyControl \
      /data/liuyilong/GR00T-WholeBodyControl/.venv_inference/bin/python \
      scripts/data/add_g1_endposes.py /data/liuyilong/dataset/walk_and_place
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


OBSERVATION_JOINT_FEATURE = "observation.state"
ACTION_JOINT_FEATURE = "action.wbc"
OBSERVATION_ENDPOSE_FEATURE = "observation.robot_endpose_abs"
ACTION_ENDPOSE_FEATURE = "action.robot_endpose_abs"
OBSERVATION_EEF_FEATURE = "observation.eef_state"

POSE_DIM = 7
ENDPOSE_SPECS = (
    ("left_hand", "left_wrist_yaw_link"),
    ("right_hand", "right_wrist_yaw_link"),
    ("left_foot", "left_ankle_roll_link"),
    ("right_foot", "right_ankle_roll_link"),
)
ENDPOSE_MODALITY_KEYS = (
    "left_wrist_endpose",
    "right_wrist_endpose",
    "left_foot_endpose",
    "right_foot_endpose",
)
ENDPOSE_WIDTH = len(ENDPOSE_SPECS) * POSE_DIM

DEFAULT_WBC_ROOT = Path("/data/liuyilong/GR00T-WholeBodyControl")


def _feature_names() -> list[str]:
    names: list[str] = []
    for endpose_name, _frame_name in ENDPOSE_SPECS:
        names.extend(
            [
                f"{endpose_name}_x",
                f"{endpose_name}_y",
                f"{endpose_name}_z",
                f"{endpose_name}_qw",
                f"{endpose_name}_qx",
                f"{endpose_name}_qy",
                f"{endpose_name}_qz",
            ]
        )
    return names


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        return
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        f.write("\n")
    tmp_path.replace(path)


def _read_vector_column(table: pa.Table, key: str, width: int) -> np.ndarray:
    if key not in table.column_names:
        raise KeyError(f"Missing required column {key!r}")
    values = np.asarray(table[key].to_pylist(), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"{key!r} must have shape [N, {width}], got {values.shape}")
    return values


def _canonicalize_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        raise ValueError("Forward kinematics produced a zero-norm quaternion")
    quaternion = quaternion / norm
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion


def _load_g1_robot_model(wbc_root: Path):
    """Load the configured, real G1 URDF model used by SONIC data collection."""
    if not wbc_root.is_dir():
        raise FileNotFoundError(f"WholeBodyControl root does not exist: {wbc_root}")

    root_string = str(wbc_root.resolve())
    if root_string not in sys.path:
        sys.path.insert(0, root_string)

    try:
        from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
    except ImportError as exc:
        raise ImportError(
            "Could not import the G1 Pinocchio model. Run this script with the "
            "GR00T-WholeBodyControl .venv_inference Python and set PYTHONPATH to its root."
        ) from exc

    return instantiate_g1_robot_model()


def _validate_joint_names(
    feature_name: str,
    source_names: Any,
    model_joint_names: list[str],
) -> list[int]:
    if not isinstance(source_names, list) or not all(
        isinstance(name, str) for name in source_names
    ):
        raise ValueError(f"{feature_name!r} must declare a list of joint names in meta/info.json")
    if len(set(source_names)) != len(source_names):
        raise ValueError(f"{feature_name!r} contains duplicate joint names")

    source_indices = {name: index for index, name in enumerate(source_names)}
    missing = [name for name in model_joint_names if name not in source_indices]
    extra = [name for name in source_names if name not in set(model_joint_names)]
    if missing or extra:
        raise ValueError(
            f"{feature_name!r} joint names do not match the G1 URDF. "
            f"missing={missing}, extra={extra}"
        )
    return [source_indices[name] for name in model_joint_names]


def _configuration_rows(
    joint_values: np.ndarray,
    source_to_model_indices: list[int],
    robot_model: Any,
) -> np.ndarray:
    if joint_values.shape[1] != len(source_to_model_indices):
        raise ValueError(
            f"Expected {len(source_to_model_indices)} joint values, got {joint_values.shape[1]}"
        )
    if robot_model.num_dofs != len(source_to_model_indices):
        raise ValueError(
            f"Expected a fixed-base G1 model with {len(source_to_model_indices)} DOFs, "
            f"got {robot_model.num_dofs}"
        )
    return joint_values[:, source_to_model_indices]


def compute_endposes(
    joint_values: np.ndarray,
    source_to_model_indices: list[int],
    robot_model: Any,
    clip_to_limits: bool,
) -> np.ndarray:
    """Compute four G1 link poses in the fixed pelvis coordinate frame."""
    q_rows = _configuration_rows(joint_values, source_to_model_indices, robot_model)
    endposes = np.empty((q_rows.shape[0], ENDPOSE_WIDTH), dtype=np.float64)

    for row_index, q in enumerate(q_rows):
        robot_model.cache_forward_kinematics(q, auto_clip=clip_to_limits)
        for endpose_index, (_endpose_name, frame_name) in enumerate(ENDPOSE_SPECS):
            placement = robot_model.frame_placement(frame_name)
            start = endpose_index * POSE_DIM
            endposes[row_index, start : start + 3] = placement.translation
            endposes[row_index, start + 3 : start + 7] = _canonicalize_quaternion_wxyz(
                Rotation.from_matrix(placement.rotation).as_quat(scalar_first=True)
            )

    return endposes


def _rotation_distance_radians(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = first / np.linalg.norm(first, axis=1, keepdims=True)
    second = second / np.linalg.norm(second, axis=1, keepdims=True)
    dot = np.abs(np.sum(first * second, axis=1))
    return 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))


def _verify_observation_wrist_poses(table: pa.Table, endposes: np.ndarray, path: Path) -> None:
    if OBSERVATION_EEF_FEATURE not in table.column_names:
        return

    existing = _read_vector_column(table, OBSERVATION_EEF_FEATURE, 14)
    generated = endposes[:, :14]
    position_error = float(
        np.max(np.abs(existing[:, [0, 1, 2, 7, 8, 9]] - generated[:, [0, 1, 2, 7, 8, 9]]))
    )
    orientation_error = float(
        np.max(
            np.concatenate(
                [
                    _rotation_distance_radians(existing[:, 3:7], generated[:, 3:7]),
                    _rotation_distance_radians(existing[:, 10:14], generated[:, 10:14]),
                ]
            )
        )
    )
    if position_error > 1e-8 or orientation_error > 1e-7:
        raise ValueError(
            f"{path}: G1 wrist FK does not reproduce {OBSERVATION_EEF_FEATURE}: "
            f"position max error={position_error:.3e} m, "
            f"orientation max error={orientation_error:.3e} rad. "
            "Use the matching WholeBodyControl model and default limit handling."
        )


def _replace_or_append_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    column = pa.array(values.tolist(), type=pa.list_(pa.float64()))
    if name in table.column_names:
        return table.set_column(table.schema.get_field_index(name), name, column)
    return table.append_column(name, column)


def _existing_endposes(table: pa.Table, feature: str) -> np.ndarray:
    return _read_vector_column(table, feature, ENDPOSE_WIDTH)


def _process_parquet(
    path: Path,
    observation_indices: list[int],
    action_indices: list[int],
    robot_model: Any,
    clip_to_limits: bool,
    dry_run: bool,
    overwrite: bool,
    verify_observation_wrists: bool,
) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path)
    has_observation = OBSERVATION_ENDPOSE_FEATURE in table.column_names
    has_action = ACTION_ENDPOSE_FEATURE in table.column_names

    if (has_observation or has_action) and not overwrite:
        if not (has_observation and has_action):
            raise ValueError(
                f"{path}: only one robot end-pose column exists; pass --overwrite to repair it"
            )
        observation_endposes = _existing_endposes(table, OBSERVATION_ENDPOSE_FEATURE)
        action_endposes = _existing_endposes(table, ACTION_ENDPOSE_FEATURE)
    else:
        observation_joints = _read_vector_column(
            table, OBSERVATION_JOINT_FEATURE, len(observation_indices)
        )
        action_joints = _read_vector_column(table, ACTION_JOINT_FEATURE, len(action_indices))
        observation_endposes = compute_endposes(
            observation_joints,
            observation_indices,
            robot_model,
            clip_to_limits=clip_to_limits,
        )
        action_endposes = compute_endposes(
            action_joints,
            action_indices,
            robot_model,
            clip_to_limits=clip_to_limits,
        )

    if verify_observation_wrists:
        _verify_observation_wrist_poses(table, observation_endposes, path)

    # Existing columns were already validated above. Metadata-only reruns should
    # not rewrite every parquet file merely to add new modality aliases.
    if dry_run or ((has_observation and has_action) and not overwrite):
        return observation_endposes, action_endposes

    updated = _replace_or_append_column(table, OBSERVATION_ENDPOSE_FEATURE, observation_endposes)
    updated = _replace_or_append_column(updated, ACTION_ENDPOSE_FEATURE, action_endposes)

    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        pq.write_table(updated, tmp_path)
        tmp_path.chmod(path.stat().st_mode)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return observation_endposes, action_endposes


def _compute_statistics(values: np.ndarray) -> dict[str, list[float]]:
    values = values.astype(np.float64, copy=False)
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _update_info(info_path: Path, dry_run: bool, overwrite: bool) -> None:
    info = _read_json(info_path)
    features = info.setdefault("features", {})
    expected = {"dtype": "float64", "shape": [ENDPOSE_WIDTH], "names": _feature_names()}

    for feature in (OBSERVATION_ENDPOSE_FEATURE, ACTION_ENDPOSE_FEATURE):
        if feature in features and not overwrite:
            existing = features[feature]
            if (
                existing.get("dtype") != expected["dtype"]
                or existing.get("shape") != expected["shape"]
            ):
                raise ValueError(
                    f"{feature!r} already exists in {info_path} with incompatible metadata; "
                    "pass --overwrite"
                )
        else:
            features[feature] = expected

    _write_json(info_path, info, dry_run)


def _update_stats(
    stats_path: Path,
    observation_stats: dict[str, list[float]],
    action_stats: dict[str, list[float]],
    dry_run: bool,
) -> None:
    stats = _read_json(stats_path) if stats_path.exists() else {}
    stats[OBSERVATION_ENDPOSE_FEATURE] = observation_stats
    stats[ACTION_ENDPOSE_FEATURE] = action_stats
    _write_json(stats_path, stats, dry_run)


def _modality_entries(feature: str) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for endpose_index, (endpose_name, _frame_name) in enumerate(ENDPOSE_SPECS):
        start = endpose_index * POSE_DIM
        entries[ENDPOSE_MODALITY_KEYS[endpose_index]] = {
            "start": start,
            "end": start + POSE_DIM,
            "original_key": feature,
            "rotation_type": "quaternion",
        }
        entries[f"robot_{endpose_name}_pos"] = {
            "start": start,
            "end": start + 3,
            "original_key": feature,
        }
        entries[f"robot_{endpose_name}_abs_quat"] = {
            "start": start + 3,
            "end": start + 7,
            "original_key": feature,
            "rotation_type": "quaternion",
        }
    return entries


def _update_modality(modality_path: Path, dry_run: bool, overwrite: bool) -> None:
    modality = _read_json(modality_path)
    for group_name, feature in (
        ("state", OBSERVATION_ENDPOSE_FEATURE),
        ("action", ACTION_ENDPOSE_FEATURE),
    ):
        group = modality.setdefault(group_name, {})
        for name, entry in _modality_entries(feature).items():
            if name not in group or overwrite:
                group[name] = entry
    _write_json(modality_path, modality, dry_run)


def _parquet_paths(dataset_path: Path) -> list[Path]:
    return sorted((dataset_path / "data").glob("chunk-*/episode_*.parquet"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path", type=Path, help="LeRobot dataset root")
    parser.add_argument(
        "--wbc-root",
        type=Path,
        default=DEFAULT_WBC_ROOT,
        help="GR00T-WholeBodyControl checkout containing the real G1 URDF",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Compute and validate without writes"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing end-pose columns"
    )
    parser.add_argument(
        "--max-files", type=int, default=0, help="Only process the first N parquet files"
    )
    parser.add_argument(
        "--no-clip-to-limits",
        action="store_false",
        dest="clip_to_limits",
        help="Use raw joint values instead of exporter-compatible joint-limit clipping",
    )
    parser.set_defaults(clip_to_limits=True)
    parser.add_argument(
        "--no-verify-observation-wrists",
        action="store_false",
        dest="verify_observation_wrists",
        help="Skip validation against the existing observation.eef_state wrist poses",
    )
    parser.set_defaults(verify_observation_wrists=True)
    parser.add_argument(
        "--skip-modality",
        action="store_true",
        help="Do not add state/action entries to meta/modality.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path.resolve()
    info_path = dataset_path / "meta" / "info.json"
    stats_path = dataset_path / "meta" / "stats.json"
    modality_path = dataset_path / "meta" / "modality.json"

    if not dataset_path.is_dir():
        raise FileNotFoundError(dataset_path)
    for path in (info_path, modality_path):
        if not path.exists():
            raise FileNotFoundError(path)

    info = _read_json(info_path)
    features = info.get("features", {})
    robot_model = _load_g1_robot_model(args.wbc_root)
    observation_indices = _validate_joint_names(
        OBSERVATION_JOINT_FEATURE,
        features.get(OBSERVATION_JOINT_FEATURE, {}).get("names"),
        robot_model.joint_names,
    )
    action_indices = _validate_joint_names(
        ACTION_JOINT_FEATURE,
        features.get(ACTION_JOINT_FEATURE, {}).get("names"),
        robot_model.joint_names,
    )

    parquet_paths = _parquet_paths(dataset_path)
    if args.max_files:
        parquet_paths = parquet_paths[: args.max_files]
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {dataset_path / 'data'}")

    observation_values: list[np.ndarray] = []
    action_values: list[np.ndarray] = []
    for index, parquet_path in enumerate(parquet_paths, start=1):
        observation_endposes, action_endposes = _process_parquet(
            parquet_path,
            observation_indices,
            action_indices,
            robot_model,
            clip_to_limits=args.clip_to_limits,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            verify_observation_wrists=args.verify_observation_wrists,
        )
        observation_values.append(observation_endposes)
        action_values.append(action_endposes)
        if index == 1 or index % 25 == 0 or index == len(parquet_paths):
            print(
                f"[{index}/{len(parquet_paths)}] {parquet_path.name}: "
                f"obs={observation_endposes.shape}, action={action_endposes.shape}",
                flush=True,
            )

    observation_all = np.concatenate(observation_values, axis=0)
    action_all = np.concatenate(action_values, axis=0)
    observation_stats = _compute_statistics(observation_all)
    action_stats = _compute_statistics(action_all)

    if args.max_files:
        print("--max-files was used; metadata was not updated because stats are partial.")
    else:
        _update_info(info_path, dry_run=args.dry_run, overwrite=args.overwrite)
        _update_stats(stats_path, observation_stats, action_stats, dry_run=args.dry_run)
        if not args.skip_modality:
            _update_modality(modality_path, dry_run=args.dry_run, overwrite=args.overwrite)

    mode = "dry-run" if args.dry_run else "updated"
    print(
        f"{mode}: {len(parquet_paths)} parquet files, {observation_all.shape[0]} frames, "
        f"{OBSERVATION_ENDPOSE_FEATURE} and {ACTION_ENDPOSE_FEATURE} shape [{ENDPOSE_WIDTH}]"
    )
    print(
        "first observation pose:",
        np.asarray(observation_stats["mean"][:POSE_DIM], dtype=np.float64).tolist(),
    )


if __name__ == "__main__":
    main()
