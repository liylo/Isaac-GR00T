# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Launch finetuning for N1.7 on "single node".
# This script tries to provide a similar user experience as current OSS.

from copy import deepcopy
import json
import math
import os
from pathlib import Path

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run


# Make sure the user provided modality config is registered.
def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


def compute_completion_pos_weight(dataset_path: str, window_frames: int) -> float:
    """Compute natural-distribution negative/positive ratio from LeRobot metadata."""
    if window_frames <= 0:
        raise ValueError("task_status_completion_window_frames must be positive")
    episodes_path = Path(dataset_path) / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(
            f"Automatic task-status pos_weight requires LeRobot episode metadata at {episodes_path}"
        )

    lengths = []
    with open(episodes_path) as f:
        for line in f:
            if line.strip():
                lengths.append(int(json.loads(line)["length"]))
    if not lengths:
        raise ValueError(f"No episodes found in {episodes_path}")

    positive_frames = sum(min(window_frames, max(length - 1, 0)) for length in lengths)
    total_frames = sum(lengths)
    negative_frames = total_frames - positive_frames
    if positive_frames <= 0 or negative_frames <= 0:
        raise ValueError("Completion window must leave both positive and negative status examples")
    return negative_frames / positive_frames


if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    # Use tyro for clean CLI
    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    from gr00t.data.embodiment_tags import EmbodimentTag

    ft_config.embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag)
    embodiment_tag = ft_config.embodiment_tag.value

    # all rank workers should register for the modality config
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [ft_config.dataset_path],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    if ft_config.endpose_xyz_only:
        from gr00t.configs.data.embodiment_configs import enable_sonic_endpose_xyz_only

        selected_modality_config = deepcopy(config.data.modality_configs[embodiment_tag])
        enable_sonic_endpose_xyz_only(selected_modality_config)
        config.data.modality_configs[embodiment_tag] = selected_modality_config
        print(f"Enabled xyz-only endpose supervision for {embodiment_tag}")

    selected_action_configs = config.data.modality_configs[embodiment_tag]["action"].action_configs
    # Legacy modality configs do not define per-key ActionConfig entries; they
    # use the original primary action flow branch.
    has_auxiliary_actions = bool(selected_action_configs) and any(
        item.is_auxiliary for item in selected_action_configs
    )
    has_primary_actions = not selected_action_configs or any(
        not item.is_auxiliary for item in selected_action_configs
    )
    if has_auxiliary_actions and has_primary_actions:
        config.model.action_branch_mode = "primary_eef"
    elif has_auxiliary_actions:
        config.model.action_branch_mode = "eef"
    else:
        config.model.action_branch_mode = "primary"
    # Mixed primary/EEF batches can legitimately leave one task adapter unused
    # on a rank. EEF-only freezes the primary adapters explicitly below.
    config.training.ddp_find_unused_parameters = config.model.action_branch_mode == "primary_eef"
    print(f"Action branch mode: {config.model.action_branch_mode}")

    # overwrite with finetune config supplied by the user
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True
    config.model.auxiliary_action_loss_weight = ft_config.endpose_loss_weight
    config.model.task_status_head_enabled = ft_config.task_status_head
    config.model.task_status_head_only = (
        ft_config.task_status_head and ft_config.task_status_head_only
    )
    config.model.task_status_hidden_dim = ft_config.task_status_hidden_dim
    config.model.task_status_num_heads = ft_config.task_status_num_heads
    config.model.task_status_completion_window_frames = (
        ft_config.task_status_completion_window_frames
    )
    config.model.task_status_completion_threshold = ft_config.task_status_completion_threshold
    config.model.task_status_progress_threshold = ft_config.task_status_progress_threshold
    config.model.task_status_consecutive_steps = ft_config.task_status_consecutive_steps
    if ft_config.task_status_head:
        if not ft_config.task_status_head_only:
            raise ValueError(
                "Joint task-status/action finetuning is not supported by this launcher: "
                "terminal action padding would create invalid action supervision"
            )
        for name, value in (
            ("task_status_completion_threshold", ft_config.task_status_completion_threshold),
            ("task_status_progress_threshold", ft_config.task_status_progress_threshold),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1], got {value}")
        if ft_config.task_status_consecutive_steps <= 0:
            raise ValueError("task_status_consecutive_steps must be positive")
        pos_weight = ft_config.task_status_completion_pos_weight
        if pos_weight is None:
            pos_weight = compute_completion_pos_weight(
                ft_config.dataset_path,
                ft_config.task_status_completion_window_frames,
            )
        if not math.isfinite(pos_weight) or pos_weight <= 0:
            raise ValueError("task_status_completion_pos_weight must be positive")
        config.model.task_status_completion_pos_weight = pos_weight
        # The terminal observations must not be truncated by the action horizon.
        config.data.allow_padding = True
        print(
            "Task-status head enabled: "
            f"window={ft_config.task_status_completion_window_frames} frames, "
            f"completion_pos_weight={pos_weight:.4f}"
        )
    if config.model.task_status_head_only:
        # Keep the pretrained policy intact and optimize only the fresh status head.
        config.model.tune_llm = False
        config.model.tune_visual = False
        config.model.tune_top_llm_layers = 0
        config.model.tune_projector = False
        config.model.tune_diffusion_model = False
        config.model.tune_vlln = False
        config.training.ddp_find_unused_parameters = False
        print("Task-status-only mode: froze backbone and action head")
    if ft_config.embodiment_tag is EmbodimentTag.UNITREE_G1_SMPL:
        config.model.action_horizon = len(
            config.data.modality_configs[embodiment_tag]["action"].delta_indices
        )

    config.training.experiment_name = ft_config.experiment_name
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.use_ddp = ft_config.use_ddp
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch

    config.training.save_only_model = ft_config.save_only_model
    config.training.skip_weight_loading = ft_config.skip_weight_loading

    run(config)
