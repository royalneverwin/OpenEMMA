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

PASSTHROUGH_ARGS=()

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [options] [-- extra-cli-args]

Examples:
  $(basename "$0") \\
    --image-file assets/scene-0103.jpg \\
    --visual-token-num 128 \\
    --quant-method quant_error_group \\
    --pruning-method cdpruner

Options:
  --python-bin PATH
  --model-path VALUE
  --image-file PATH
  --conv-mode VALUE
  --load-4bit / --no-load-4bit
  --load-8bit / --no-load-8bit
  --use-qvlm-custom-bnb / --no-use-qvlm-custom-bnb
  --custom-bnb-path PATH
  --visual-token-num VALUE
  --add-quant / --no-add-quant
  --alpha VALUE
  --dynamic-alpha / --no-dynamic-alpha
  --quant-method VALUE
  --pruning-method VALUE
  -h, --help
EOF
}

require_value() {
  local option_name="$1"
  local option_value="${2:-}"
  if [[ -z "$option_value" ]]; then
    echo "Missing value for $option_name" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-bin)
      require_value "$1" "${2:-}"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --model-path)
      require_value "$1" "${2:-}"
      MODEL_PATH="$2"
      shift 2
      ;;
    --image-file)
      require_value "$1" "${2:-}"
      IMAGE_FILE="$2"
      shift 2
      ;;
    --conv-mode)
      require_value "$1" "${2:-}"
      CONV_MODE="$2"
      shift 2
      ;;
    --load-4bit)
      LOAD_4BIT=1
      shift
      ;;
    --no-load-4bit)
      LOAD_4BIT=0
      shift
      ;;
    --load-8bit)
      LOAD_8BIT=1
      shift
      ;;
    --no-load-8bit)
      LOAD_8BIT=0
      shift
      ;;
    --use-qvlm-custom-bnb)
      USE_QVLM_CUSTOM_BNB=1
      shift
      ;;
    --no-use-qvlm-custom-bnb)
      USE_QVLM_CUSTOM_BNB=0
      shift
      ;;
    --custom-bnb-path)
      require_value "$1" "${2:-}"
      CUSTOM_BNB_PATH="$2"
      shift 2
      ;;
    --visual-token-num)
      require_value "$1" "${2:-}"
      VISUAL_TOKEN_NUM="$2"
      shift 2
      ;;
    --add-quant)
      ADD_QUANT=1
      shift
      ;;
    --no-add-quant)
      ADD_QUANT=0
      shift
      ;;
    --alpha)
      require_value "$1" "${2:-}"
      ALPHA="$2"
      shift 2
      ;;
    --dynamic-alpha)
      DYNAMIC_ALPHA=1
      shift
      ;;
    --no-dynamic-alpha)
      DYNAMIC_ALPHA=0
      shift
      ;;
    --quant-method)
      require_value "$1" "${2:-}"
      QUANT_METHOD="$2"
      shift 2
      ;;
    --pruning-method)
      require_value "$1" "${2:-}"
      PRUNING_METHOD="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      PASSTHROUGH_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Run with --help to see supported options." >&2
      exit 1
      ;;
  esac
done

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

if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
  CMD+=("${PASSTHROUGH_ARGS[@]}")
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
exec "${CMD[@]}"
