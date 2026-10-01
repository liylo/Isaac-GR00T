#!/usr/bin/env bash

set -x -euo pipefail

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-12345}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/data/liuyilong/Isaac-GR00T/output_walk/checkpoint-20000}"
DATASET_PATH="${DATASET_PATH:-/data/liuyilong/dataset/walk_and_place_ep10}"
OUTPUT_DIR="${OUTPUT_DIR:-output_walk_dual2_ep10}"
MAX_STEPS="${MAX_STEPS:-50000}"

torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
    gr00t/experiment/launch_finetune.py \
    --base-model-path "${BASE_MODEL_PATH}" \
    --dataset-path "${DATASET_PATH}" \
    --embodiment-tag UNITREE_G1_SONIC \
    --modality-config-path gr00t/configs/data/embodiment_configs.py \
    --num-gpus "${NUM_GPUS}" \
    --output-dir "${OUTPUT_DIR}" \
    --save-total-limit 10 \
    --save-steps 5000 \
    --max-steps "${MAX_STEPS}" \
    --use-wandb \
    --global-batch-size 64 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 8 \
    --task-status-head \
    --task-status-head-only \
    --task-status-completion-window-frames 25 \
    --task-status-consecutive-steps 1 \
    "$@"
