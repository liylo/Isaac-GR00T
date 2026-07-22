#!/usr/bin/env python3

"""Validate a SONIC endpose dataset without decoding video or loading a model."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS, enable_sonic_endpose_xyz_only
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.state_action.state_action_processor import StateActionProcessor
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("embodiment_tag")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument(
        "--endpose-xyz-only",
        action="store_true",
        help="Validate translation-only EEF supervision using the existing full-pose stats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    config = deepcopy(MODALITY_CONFIGS[embodiment_tag.value])
    if args.endpose_xyz_only:
        enable_sonic_endpose_xyz_only(config)
    lowdim_config = {key: value for key, value in config.items() if key != "video"}

    loader = LeRobotEpisodeLoader(args.dataset_path, lowdim_config)
    max_action_offset = max(
        max(action_config.delta_indices or config["action"].delta_indices)
        for action_config in config["action"].action_configs
    )
    episode_length = loader.get_episode_length(args.episode_index)
    if args.step_index + max_action_offset >= episode_length:
        raise IndexError(
            f"step {args.step_index} with max action offset {max_action_offset} "
            f"exceeds episode length {episode_length}"
        )

    statistics = loader.get_dataset_statistics()
    relative_stats = statistics.get("relative_action", {})
    reference_keys = set(config["state"].reference_only_keys or [])
    action_config_by_key = dict(
        zip(config["action"].modality_keys, config["action"].action_configs)
    )
    for key in reference_keys:
        mean = np.asarray(relative_stats[key]["mean"])
        expected_dim = 3 if action_config_by_key[key].format.name == "XYZ" else 9
        expected_horizon = len(config["action"].delta_indices)
        allowed_stat_dims = {3, 9} if expected_dim == 3 else {9}
        if (
            mean.ndim != 2
            or mean.shape[0] != expected_horizon
            or mean.shape[1] not in allowed_stat_dims
        ):
            raise ValueError(
                f"relative stats for {key} have shape {mean.shape}; "
                f"cannot provide ({expected_horizon}, {expected_dim}) EEF targets"
            )

    episode = loader.load_episode_data(args.episode_index)
    step = extract_step_data(
        episode,
        args.step_index,
        lowdim_config,
        embodiment_tag,
    )
    processor = StateActionProcessor(
        modality_configs={embodiment_tag.value: config},
        statistics={embodiment_tag.value: statistics},
        use_percentiles=True,
        use_relative_action=True,
    )
    processed_state, processed_action = processor.apply(
        step.states, step.actions, embodiment_tag.value
    )
    decoded_action = processor.unapply_action(processed_action, embodiment_tag.value, state=None)
    leaked_references = reference_keys & set(processed_state)
    if leaked_references:
        raise ValueError(
            f"reference-only keys leaked into model state: {sorted(leaked_references)}"
        )

    model_processor = object.__new__(Gr00tN1d7Processor)
    model_processor.modality_configs = {embodiment_tag.value: config}
    model_processor.state_action_processor = processor
    model_processor.max_action_dim = 132
    model_processor.max_action_horizon = 40
    packed_action, loss_mask, auxiliary_mask = model_processor._pack_actions(
        processed_action, embodiment_tag.value
    )
    primary_sampling_mask = model_processor._build_primary_action_mask(embodiment_tag.value)

    summary = {
        "dataset": str(args.dataset_path),
        "embodiment_tag": embodiment_tag.value,
        "episode_index": args.episode_index,
        "step_index": args.step_index,
        "episode_length": episode_length,
        "max_action_offset": max_action_offset,
        "action_offsets": {
            key: action_config.delta_indices or config["action"].delta_indices
            for key, action_config in zip(
                config["action"].modality_keys, config["action"].action_configs
            )
        },
        "model_state_keys": list(processed_state),
        "action_shapes": {key: list(value.shape) for key, value in processed_action.items()},
        "decoded_action_shapes": {key: list(value.shape) for key, value in decoded_action.items()},
        "action_slices": {
            key: list(value)
            for key, value in processor.get_action_slices(embodiment_tag.value).items()
        },
        "packed_action_shape": list(packed_action.shape),
        "supervised_dims_per_step": int(torch.count_nonzero(loss_mask[0]).item()),
        "loss_weight_sum_per_step": float(loss_mask[0].sum().item()),
        "auxiliary_dims_per_step": int(torch.count_nonzero(auxiliary_mask[0]).item()),
        "primary_sampling_dims_per_step": int(
            torch.count_nonzero(primary_sampling_mask[0]).item()
        ),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
