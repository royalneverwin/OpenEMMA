#!/usr/bin/env bash

set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
DATAROOT=/mnt/bn/yufei1900/wangxinhao/paper/data/nuscenes

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MODEL_PATH="${MODEL_PATH:-/mnt/bn/yufei1900/wangxinhao/paper/checkpoint/llava-v1.6-mistral-7b}"
DATAROOT="${DATAROOT:-/mnt/bn/yufei1900/wangxinhao/paper/data/nuscenes-mini}"
VERSION="${VERSION:-v1.0-mini}"
METHOD="${METHOD:-openemma}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" main_multi.py \
    --model-path "${MODEL_PATH}" \
    --dataroot "${DATAROOT}" \
    --version "${VERSION}" \
    --method "${METHOD}" \
    --output-dir ./output_multi \
    "$@"
