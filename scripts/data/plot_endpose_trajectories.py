#!/usr/bin/env python3

"""Plot one human and one robot absolute end-pose trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pyarrow.parquet as pq


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HUMAN_FEATURE = "teleop.human_endpose_abs"
ROBOT_FEATURES = {
    "action": "action.robot_endpose_abs",
    "observation": "observation.robot_endpose_abs",
}
COMPONENTS = ("x [m]", "y [m]", "z [m]", "qw", "qx", "qy", "qz")
COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#8c564b", "#e377c2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--human-dataset",
        type=Path,
        default=Path(
            "/data/liuyilong/dataset/"
            "token_only_gr00t_human_ankle_mid_arms_only_delay8_action_lead1"
        ),
    )
    parser.add_argument(
        "--robot-dataset",
        type=Path,
        default=Path("/data/liuyilong/dataset/walk_and_place"),
    )
    parser.add_argument("--human-episode", type=int, default=0)
    parser.add_argument("--robot-episode", type=int, default=0)
    parser.add_argument(
        "--robot-source",
        choices=sorted(ROBOT_FEATURES),
        default="action",
        help="Plot robot action targets or observation reference poses.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/liuyilong/endpose_trajectory_plots"),
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def find_episode(dataset: Path, episode_index: int) -> Path:
    matches = sorted((dataset / "data").glob(f"chunk-*/episode_{episode_index:06d}.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one parquet for episode {episode_index} in {dataset}, got {matches}"
        )
    return matches[0]


def load_poses(dataset: Path, episode_index: int, feature: str) -> tuple[np.ndarray, Path]:
    episode_path = find_episode(dataset, episode_index)
    table = pq.read_table(episode_path, columns=[feature])
    poses = np.asarray(table[feature].combine_chunks().to_pylist(), dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 28:
        raise ValueError(f"{feature} in {episode_path} has shape {poses.shape}, expected (T, 28)")
    if not np.isfinite(poses).all():
        bad_count = poses.size - np.count_nonzero(np.isfinite(poses))
        raise ValueError(f"{feature} in {episode_path} contains {bad_count} non-finite values")
    return poses.reshape(-1, 4, 7), episode_path


def plot_trajectory(
    poses: np.ndarray,
    limb_names: tuple[str, str, str, str],
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    timesteps = np.arange(poses.shape[0])
    fig, axes = plt.subplots(
        nrows=7,
        ncols=4,
        figsize=(22, 17),
        sharex=True,
        constrained_layout=True,
    )
    for limb_index, limb_name in enumerate(limb_names):
        for component_index, component_name in enumerate(COMPONENTS):
            axis = axes[component_index, limb_index]
            axis.plot(
                timesteps,
                poses[:, limb_index, component_index],
                color=COLORS[component_index],
                linewidth=1.0,
            )
            axis.grid(alpha=0.25, linewidth=0.6)
            if component_index == 0:
                axis.set_title(limb_name, fontsize=12, fontweight="bold")
            if limb_index == 0:
                axis.set_ylabel(component_name)
            if component_index == len(COMPONENTS) - 1:
                axis.set_xlabel("t [frame]")

    fig.suptitle(f"{title}\nT={poses.shape[0]} frames", fontsize=16, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    human_poses, human_episode_path = load_poses(
        args.human_dataset, args.human_episode, HUMAN_FEATURE
    )
    robot_feature = ROBOT_FEATURES[args.robot_source]
    robot_poses, robot_episode_path = load_poses(
        args.robot_dataset, args.robot_episode, robot_feature
    )

    human_output = args.output_dir / f"human_endpose_episode_{args.human_episode:06d}.png"
    robot_output = (
        args.output_dir
        / f"robot_{args.robot_source}_endpose_episode_{args.robot_episode:06d}.png"
    )
    plot_trajectory(
        human_poses,
        ("left hand", "right hand", "left foot", "right foot"),
        f"Human absolute end-pose — {HUMAN_FEATURE}",
        human_output,
        args.dpi,
    )
    plot_trajectory(
        robot_poses,
        ("left wrist", "right wrist", "left foot", "right foot"),
        f"Robot absolute end-pose — {robot_feature}",
        robot_output,
        args.dpi,
    )

    print(f"human source: {human_episode_path}")
    print(f"human plot:   {human_output.resolve()}")
    print(f"robot source: {robot_episode_path}")
    print(f"robot plot:   {robot_output.resolve()}")


if __name__ == "__main__":
    main()
