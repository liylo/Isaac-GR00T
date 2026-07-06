#!/usr/bin/env bash

set -x -euo pipefail

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-12345}"
CONFIG_PATH="${CONFIG_PATH:-examples/unitree_g1_sonic_dual2_2to1.yaml}"

export HF_HOME=/data/liuyilong/nvidia_hf_cache
export HF_ENDPOINT=https://hf-mirror.com

torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
    gr00t/experiment/launch_train.py \
    --load_config_path "${CONFIG_PATH}"
