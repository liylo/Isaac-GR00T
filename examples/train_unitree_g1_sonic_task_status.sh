#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: bash $0 <task-policy-checkpoint> <output-dir> [extra launch_finetune.py args...]" >&2
    exit 1
fi

BASE_MODEL_PATH="$1"
OUTPUT_DIR="$2"
shift 2

DATASET_PATH="${DATASET_PATH:-/data/liuyilong/dataset/walk_and_place}"
TASK_STATUS_WINDOW_FRAMES="${TASK_STATUS_WINDOW_FRAMES:-25}"
TASK_STATUS_CONSECUTIVE_STEPS="${TASK_STATUS_CONSECUTIVE_STEPS:-1}"

export MAX_STEPS="${MAX_STEPS:-3000}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export USE_WANDB="${USE_WANDB:-0}"

exec bash examples/finetune.sh \
    --base-model-path "$BASE_MODEL_PATH" \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag UNITREE_G1_SONIC \
    --output-dir "$OUTPUT_DIR" \
    -- \
    --task-status-head \
    --task-status-completion-window-frames "$TASK_STATUS_WINDOW_FRAMES" \
    --task-status-consecutive-steps "$TASK_STATUS_CONSECUTIVE_STEPS" \
    "$@"
