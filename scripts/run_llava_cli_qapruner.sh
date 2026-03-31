#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.6-mistral-7b}"
IMAGE_FILE="${IMAGE_FILE:-$REPO_ROOT/assets/scene-0103.jpg}"
CONV_MODE="${CONV_MODE:-mistral_instruct}"

LOAD_4BIT="${LOAD_4BIT:-1}"
LOAD_8BIT="${LOAD_8BIT:-0}"
USE_QVLM_CUSTOM_BNB="${USE_QVLM_CUSTOM_BNB:-1}"
CUSTOM_BNB_PATH="${CUSTOM_BNB_PATH:-$REPO_ROOT/custom_bitsandbytes}"

VISUAL_TOKEN_NUM="${VISUAL_TOKEN_NUM:-128}"
ADD_QUANT="${ADD_QUANT:-1}"
ALPHA="${ALPHA:-0.7}"
DYNAMIC_ALPHA="${DYNAMIC_ALPHA:-0}"
QUANT_METHOD="${QUANT_METHOD:-quant_error_group}"
PRUNING_METHOD="${PRUNING_METHOD:-cdpruner}"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CMD=(
  "$PYTHON_BIN" -m llava.serve.cli
  --model-path "$MODEL_PATH"
  --image-file "$IMAGE_FILE"
  --conv-mode "$CONV_MODE"
  --visual-token-num "$VISUAL_TOKEN_NUM"
  --alpha "$ALPHA"
  --quant-method "$QUANT_METHOD"
  --pruning-method "$PRUNING_METHOD"
  --custom-bnb-path "$CUSTOM_BNB_PATH"
)

if [[ "$LOAD_4BIT" == "1" ]]; then
  CMD+=(--load-4bit)
fi

if [[ "$LOAD_8BIT" == "1" ]]; then
  CMD+=(--load-8bit)
fi

if [[ "$USE_QVLM_CUSTOM_BNB" == "1" ]]; then
  CMD+=(--use-qvlm-custom-bnb)
fi

if [[ "$ADD_QUANT" == "1" ]]; then
  CMD+=(--add-quant)
fi

if [[ "$DYNAMIC_ALPHA" == "1" ]]; then
  CMD+=(--dynamic-alpha)
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
exec "${CMD[@]}"
