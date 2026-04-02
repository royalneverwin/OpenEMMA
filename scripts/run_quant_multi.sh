#!/usr/bin/env bash

set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MODEL_PATH="${MODEL_PATH:-/mnt/bn/yufei1900/wangxinhao/paper/checkpoint/llava-v1.6-mistral-7b}"
DATAROOT="${DATAROOT:-/mnt/bn/yufei1900/wangxinhao/paper/data/nuscenes-mini}"
VERSION="${VERSION:-v1.0-mini}"
METHOD="${METHOD:-openemma}"
OUTPUT_DIR="${OUTPUT_DIR:-./output_quant_multi}"
MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 20000))}"

POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --master_port|--master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --master_port=*|--master-port=*)
            MASTER_PORT="${1#*=}"
            shift
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "Using torchrun master port: ${MASTER_PORT}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" main_multi.py \
    --model-path "${MODEL_PATH}" \
    --dataroot "${DATAROOT}" \
    --version "${VERSION}" \
    --method "${METHOD}" \
    --output-dir "${OUTPUT_DIR}" \
    --load-4bit \
    --visual-token-num 32 \
    --add-quant \
    --pruning-method cdpruner \
    --alpha 0.5 \
    --quant-method quant_error_group \
    --run-calibration \
    "${POSITIONAL_ARGS[@]}"
