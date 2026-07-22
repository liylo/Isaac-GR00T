#!/usr/bin/env bash

set -x -euo pipefail

# Sequential endpose-transfer ablation:
# 1. Robot-only training from the same base checkpoint.
# 2. Human-only endpose pretraining.
# 3. Robot training initialized from the human checkpoint.
#
# HUMAN_ACTION_MODE=endpose (default) supervises only four relative EEF poses.
# HUMAN_ACTION_MODE=token_endpose also supervises token/hands; both task losses are
# normalized independently and ENDPOSE_LOSS_WEIGHT weights EEF. ROBOT_ACTION_MODE defaults to token so
# the downstream baseline and transfer runs are evaluated through the real control
# output. Set ROBOT_ACTION_MODE=token_endpose for an auxiliary EEF robot objective.
# Human and robot EEF chunks both target t+1..t+40 relative to observation t.
# ENDPOSE_XYZ_ONLY=1 keeps that observation-frame conversion but supervises only
# relative xyz; the default 0 continues to supervise xyz+rot6d.
# ENDPOSE_LOSS_WEIGHT controls the auxiliary EEF branch only when token and EEF
# are active together. EEF-only human pretraining keeps a unit-normalized loss.

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-12345}"
TORCHRUN="${TORCHRUN:-$(command -v torchrun || true)}"
if [ -z "${TORCHRUN}" ] && [ -x ".venv/bin/torchrun" ]; then
    TORCHRUN=".venv/bin/torchrun"
fi
if [ -z "${TORCHRUN}" ]; then
    echo "torchrun not found. Activate the project environment or set TORCHRUN=/path/to/torchrun." >&2
    exit 1
fi

BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.7-3B}"
MODALITY_CONFIG_PATH="${MODALITY_CONFIG_PATH:-gr00t/configs/data/embodiment_configs.py}"

ROBOT_DATASET_PATH="${ROBOT_DATASET_PATH:-/data/liuyilong/dataset/walk_and_place}"
HUMAN_DATASET_PATH="${HUMAN_DATASET_PATH:-/data/liuyilong/dataset/token_only_gr00t_human_ankle_mid_arms_only_delay8_action_lead1}"

HUMAN_ACTION_MODE="${HUMAN_ACTION_MODE:-${ACTION_MODE:-endpose}}"
case "${HUMAN_ACTION_MODE}" in
    endpose)
        HUMAN_EMBODIMENT_TAG=UNITREE_G1_SONIC_HUMAN_ENDPOSE
        ;;
    token_endpose)
        HUMAN_EMBODIMENT_TAG=UNITREE_G1_SONIC_HUMAN_TOKEN_ENDPOSE
        ;;
    token)
        HUMAN_EMBODIMENT_TAG=UNITREE_G1_SONIC_HUMAN
        ;;
    *)
        echo "HUMAN_ACTION_MODE must be endpose, token_endpose, or token; got ${HUMAN_ACTION_MODE}" >&2
        exit 2
        ;;
esac

ROBOT_ACTION_MODE="${ROBOT_ACTION_MODE:-token}"
case "${ROBOT_ACTION_MODE}" in
    endpose)
        ROBOT_EMBODIMENT_TAG=UNITREE_G1_SONIC_ENDPOSE
        ;;
    token_endpose)
        ROBOT_EMBODIMENT_TAG=UNITREE_G1_SONIC_TOKEN_ENDPOSE
        ;;
    token)
        ROBOT_EMBODIMENT_TAG=UNITREE_G1_SONIC
        ;;
    *)
        echo "ROBOT_ACTION_MODE must be endpose, token_endpose, or token; got ${ROBOT_ACTION_MODE}" >&2
        exit 2
        ;;
esac

ENDPOSE_XYZ_ONLY="${ENDPOSE_XYZ_ONLY:-0}"
case "${ENDPOSE_XYZ_ONLY}" in
    0|1)
        ;;
    *)
        echo "ENDPOSE_XYZ_ONLY must be 0 or 1; got ${ENDPOSE_XYZ_ONLY}" >&2
        exit 2
        ;;
esac

HUMAN_ENDPOSE_ARGS=()
HUMAN_ENDPOSE_SUFFIX=""
ROBOT_ENDPOSE_ARGS=()
ROBOT_ENDPOSE_SUFFIX=""
if [ "${ENDPOSE_XYZ_ONLY}" = "1" ]; then
    if [ "${HUMAN_ACTION_MODE}" != "token" ]; then
        HUMAN_ENDPOSE_ARGS+=(--endpose-xyz-only)
        HUMAN_ENDPOSE_SUFFIX="_xyz_only"
    fi
    if [ "${ROBOT_ACTION_MODE}" != "token" ]; then
        ROBOT_ENDPOSE_ARGS+=(--endpose-xyz-only)
        ROBOT_ENDPOSE_SUFFIX="_xyz_only"
    fi
    if [ "${HUMAN_ACTION_MODE}" = "token" ] && [ "${ROBOT_ACTION_MODE}" = "token" ]; then
        echo "ENDPOSE_XYZ_ONLY=1 requires an endpose or token_endpose action mode" >&2
        exit 2
    fi
fi

TRANSFER_NAME="${HUMAN_ACTION_MODE}${HUMAN_ENDPOSE_SUFFIX}_to_${ROBOT_ACTION_MODE}${ROBOT_ENDPOSE_SUFFIX}"
ROBOT_BASELINE_OUTPUT_DIR="${ROBOT_BASELINE_OUTPUT_DIR:-output_walk_${ROBOT_ACTION_MODE}${ROBOT_ENDPOSE_SUFFIX}_baseline}"
HUMAN_PRETRAIN_OUTPUT_DIR="${HUMAN_PRETRAIN_OUTPUT_DIR:-output_human_${HUMAN_ACTION_MODE}${HUMAN_ENDPOSE_SUFFIX}_pretrain}"
HUMAN_TO_ROBOT_OUTPUT_DIR="${HUMAN_TO_ROBOT_OUTPUT_DIR:-output_walk_${TRANSFER_NAME}_from_human}"

SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
MAX_STEPS="${MAX_STEPS:-20000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
SHARD_SIZE="${SHARD_SIZE:-4096}"
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"
ENDPOSE_LOSS_WEIGHT="${ENDPOSE_LOSS_WEIGHT:-0.5}"
HUMAN_PRETRAIN_CHECKPOINT="${HUMAN_PRETRAIN_CHECKPOINT:-${HUMAN_PRETRAIN_OUTPUT_DIR}/checkpoint-${MAX_STEPS}}"

RUN_ROBOT_BASELINE="${RUN_ROBOT_BASELINE:-1}"
RUN_HUMAN_PRETRAIN="${RUN_HUMAN_PRETRAIN:-1}"
RUN_TRANSFER="${RUN_TRANSFER:-1}"

DDP_FLAG=()
if [ "${USE_DDP:-1}" = "1" ]; then
    DDP_FLAG+=(--use-ddp)
fi

WANDB_FLAG=()
if [ "${USE_WANDB:-1}" = "1" ]; then
    WANDB_FLAG+=(--use-wandb)
fi

COMMON_ARGS=(
    --modality-config-path "${MODALITY_CONFIG_PATH}"
    --num-gpus "${NUM_GPUS}"
    --save-total-limit "${SAVE_TOTAL_LIMIT}"
    --save-steps "${SAVE_STEPS}"
    --max-steps "${MAX_STEPS}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
    --dataloader-num-workers "${DATALOADER_NUM_WORKERS}"
    --shard-size "${SHARD_SIZE}"
    --num-shards-per-epoch "${NUM_SHARDS_PER_EPOCH}"
    --endpose-loss-weight "${ENDPOSE_LOSS_WEIGHT}"
)
COMMON_ARGS+=("${DDP_FLAG[@]}")
COMMON_ARGS+=("${WANDB_FLAG[@]}")

BASE_INIT_ARGS=()
# This is a true random action-head ablation: it skips all GR00T checkpoint
# weights, not just the newly supervised EEF slots. Keep it off for transfer.
if [ "${RANDOM_ACTION_HEAD:-0}" = "1" ]; then
    BASE_INIT_ARGS+=(--skip-weight-loading)
fi

export HF_HOME=/data/liuyilong/nvidia_hf_cache
export HF_ENDPOINT=https://hf-mirror.com

run_training() {
    local master_port="$1"
    shift

    "${TORCHRUN}" --nproc_per_node="${NUM_GPUS}" --master_port="${master_port}" \
        gr00t/experiment/launch_finetune.py \
        "$@"
}

if [ "${RUN_ROBOT_BASELINE}" = "1" ]; then
    run_training "${MASTER_PORT}" \
        --base-model-path "${BASE_MODEL_PATH}" \
        --dataset-path "${ROBOT_DATASET_PATH}" \
        --embodiment-tag "${ROBOT_EMBODIMENT_TAG}" \
        --output-dir "${ROBOT_BASELINE_OUTPUT_DIR}" \
        "${ROBOT_ENDPOSE_ARGS[@]}" \
        "${BASE_INIT_ARGS[@]}" \
        "${COMMON_ARGS[@]}"
fi

if [ "${RUN_HUMAN_PRETRAIN}" = "1" ]; then
    run_training "$((MASTER_PORT + 1))" \
        --base-model-path "${BASE_MODEL_PATH}" \
        --dataset-path "${HUMAN_DATASET_PATH}" \
        --embodiment-tag "${HUMAN_EMBODIMENT_TAG}" \
        --output-dir "${HUMAN_PRETRAIN_OUTPUT_DIR}" \
        "${HUMAN_ENDPOSE_ARGS[@]}" \
        "${BASE_INIT_ARGS[@]}" \
        "${COMMON_ARGS[@]}"
fi

if [ "${RUN_TRANSFER}" = "1" ]; then
    if [ ! -d "${HUMAN_PRETRAIN_CHECKPOINT}" ]; then
        echo "Missing human pretrain checkpoint: ${HUMAN_PRETRAIN_CHECKPOINT}" >&2
        exit 1
    fi
    run_training "$((MASTER_PORT + 2))" \
        --base-model-path "${HUMAN_PRETRAIN_CHECKPOINT}" \
        --dataset-path "${ROBOT_DATASET_PATH}" \
        --embodiment-tag "${ROBOT_EMBODIMENT_TAG}" \
        --output-dir "${HUMAN_TO_ROBOT_OUTPUT_DIR}" \
        "${ROBOT_ENDPOSE_ARGS[@]}" \
        "${COMMON_ARGS[@]}"
fi
