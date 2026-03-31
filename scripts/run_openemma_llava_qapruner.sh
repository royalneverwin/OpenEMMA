#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-llava}"
DATAROOT="${DATAROOT:-$REPO_ROOT/datasets/NuScenes}"
VERSION="${VERSION:-v1.0-mini}"
METHOD="${METHOD:-openemma}"
SCENE_NAMES="${SCENE_NAMES:-scene-0103,scene-1077}"
PLOT="${PLOT:-true}"

LOAD_4BIT="${LOAD_4BIT:-1}"
LOAD_8BIT="${LOAD_8BIT:-0}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-0}"
USE_QVLM_CUSTOM_BNB="${USE_QVLM_CUSTOM_BNB:-1}"
CUSTOM_BNB_PATH="${CUSTOM_BNB_PATH:-$REPO_ROOT/custom_bitsandbytes}"

VISUAL_TOKEN_NUM="${VISUAL_TOKEN_NUM:-128}"
ADD_QUANT="${ADD_QUANT:-1}"
ALPHA="${ALPHA:-0.7}"
DYNAMIC_ALPHA="${DYNAMIC_ALPHA:-0}"
QUANT_METHOD="${QUANT_METHOD:-quant_error_group}"
PRUNING_METHOD="${PRUNING_METHOD:-cdpruner}"

RUN_CALIBRATION="${RUN_CALIBRATION:-1}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-8}"
CALIBRATION_SEARCH_SAMPLES="${CALIBRATION_SEARCH_SAMPLES:-2}"
CALIBRATION_MAX_NEW_TOKENS="${CALIBRATION_MAX_NEW_TOKENS:-32}"
CALIBRATION_SCENE_NAMES="${CALIBRATION_SCENE_NAMES:-}"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CMD=(
  "$PYTHON_BIN" "$REPO_ROOT/main.py"
  --model-path "$MODEL_PATH"
  --dataroot "$DATAROOT"
  --version "$VERSION"
  --method "$METHOD"
  --scene-names "$SCENE_NAMES"
  --plot "$PLOT"
  --visual-token-num "$VISUAL_TOKEN_NUM"
  --alpha "$ALPHA"
  --quant-method "$QUANT_METHOD"
  --pruning-method "$PRUNING_METHOD"
  --custom-bnb-path "$CUSTOM_BNB_PATH"
  --calibration-samples "$CALIBRATION_SAMPLES"
  --calibration-search-samples "$CALIBRATION_SEARCH_SAMPLES"
  --calibration-max-new-tokens "$CALIBRATION_MAX_NEW_TOKENS"
)

if [[ "$LOAD_4BIT" == "1" ]]; then
  CMD+=(--load-4bit)
fi

if [[ "$LOAD_8BIT" == "1" ]]; then
  CMD+=(--load-8bit)
fi

if [[ "$USE_FLASH_ATTN" == "1" ]]; then
  CMD+=(--use-flash-attn)
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

if [[ "$RUN_CALIBRATION" == "1" ]]; then
  CMD+=(--run-calibration)
fi

if [[ -n "$CALIBRATION_SCENE_NAMES" ]]; then
  CMD+=(--calibration-scene-names "$CALIBRATION_SCENE_NAMES")
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
exec "${CMD[@]}"
