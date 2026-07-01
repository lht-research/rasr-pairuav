#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

DATA_ROOT=""
OUTPUT_DIR=""
LIMIT=""
MAX_STEPS=""
DEVICE="cpu"
FEATURE_MODE="smoke"
RUNTIME_ROOT=""
MODEL_PATH=""
MANIFEST=""
JSON_DIR=""
IMAGE_DIR=""
SKIP_TRAINING=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --max-steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --feature-mode)
      FEATURE_MODE="$2"
      shift 2
      ;;
    --runtime-root)
      RUNTIME_ROOT="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --json-dir)
      JSON_DIR="$2"
      shift 2
      ;;
    --image-dir)
      IMAGE_DIR="$2"
      shift 2
      ;;
    --skip-training)
      SKIP_TRAINING=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$DATA_ROOT" || -z "$OUTPUT_DIR" ]]; then
  cat >&2 <<EOF
Usage: $0 --data-root PATH --output-dir PATH [options]

Options:
  --limit N                 Process a prefix for smoke validation.
  --max-steps N             Limit distance-head training steps.
  --device DEVICE           Torch device for distance checkpoint inference/training.
  --feature-mode MODE       smoke or production. Default: smoke.
  --runtime-root PATH       Production feature runtime root.
  --model-path PATH         Production feature model path.
  --manifest PATH           Production test manifest.
  --json-dir PATH           Production JSON directory.
  --image-dir PATH          Production image directory.
  --skip-training           Reuse prepared model/prediction outputs in output dir.
EOF
  exit 2
fi

if [[ "$FEATURE_MODE" != "smoke" && "$FEATURE_MODE" != "production" ]]; then
  echo "--feature-mode must be smoke or production" >&2
  exit 2
fi

WORK_DIR="$OUTPUT_DIR/work"
FEATURE_DIR="$OUTPUT_DIR/features"
MODEL_DIR="$OUTPUT_DIR/models"
PRED_DIR="$OUTPUT_DIR/predictions"
SUBMISSION_DIR="$OUTPUT_DIR/submission"
mkdir -p "$WORK_DIR" "$FEATURE_DIR" "$MODEL_DIR" "$PRED_DIR" "$SUBMISSION_DIR"

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

EXTRACT_COMMON=(
  "$ROOT_DIR/data_processing/03_extract_features.py"
  --mode "$FEATURE_MODE"
  --data-root "$DATA_ROOT"
)
if [[ -n "$RUNTIME_ROOT" ]]; then
  EXTRACT_COMMON+=(--runtime-root "$RUNTIME_ROOT")
fi
if [[ -n "$MODEL_PATH" ]]; then
  EXTRACT_COMMON+=(--model-path "$MODEL_PATH")
fi
if [[ -n "$JSON_DIR" ]]; then
  EXTRACT_COMMON+=(--json-dir "$JSON_DIR")
fi
if [[ -n "$IMAGE_DIR" ]]; then
  EXTRACT_COMMON+=(--image-dir "$IMAGE_DIR")
fi

PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" "$ROOT_DIR/data_processing/02_build_pairs.py" \
  --data-root "$DATA_ROOT" \
  --split train \
  --output "$WORK_DIR/train_pairs.csv" \
  "${LIMIT_ARGS[@]}"

PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" "$ROOT_DIR/data_processing/02_build_pairs.py" \
  --data-root "$DATA_ROOT" \
  --split test \
  --output "$WORK_DIR/test_pairs.csv" \
  "${LIMIT_ARGS[@]}"

if [[ "$FEATURE_MODE" == "smoke" ]]; then
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" "${EXTRACT_COMMON[@]}" \
    --pairs "$WORK_DIR/train_pairs.csv" \
    --output "$FEATURE_DIR/train_pair_features.csv" \
    "${LIMIT_ARGS[@]}"
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" "${EXTRACT_COMMON[@]}" \
    --pairs "$WORK_DIR/test_pairs.csv" \
    --output "$FEATURE_DIR/test_pair_features.csv" \
    "${LIMIT_ARGS[@]}"
else
  TRAIN_EXTRACT=("${EXTRACT_COMMON[@]}" --output "$FEATURE_DIR/train_pair_features.csv")
  TEST_EXTRACT=("${EXTRACT_COMMON[@]}" --output "$FEATURE_DIR/test_pair_features.csv")
  if [[ -n "$MANIFEST" ]]; then
    TEST_EXTRACT+=(--manifest "$MANIFEST")
  fi
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" "${TRAIN_EXTRACT[@]}" "${LIMIT_ARGS[@]}"
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" "${TEST_EXTRACT[@]}" "${LIMIT_ARGS[@]}"
fi

PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" "$ROOT_DIR/data_processing/04_make_splits.py" \
  --pairs "$WORK_DIR/train_pairs.csv" \
  --output-dir "$WORK_DIR/splits" \
  "${LIMIT_ARGS[@]}"

if [[ "$SKIP_TRAINING" -eq 0 ]]; then
  for HEAD_NAME in distance_head_a distance_head_b distance_head_c distance_head_d; do
    TRAIN_ARGS=(
      "$ROOT_DIR/models_src/train_head.py"
      --config "$ROOT_DIR/configs/heads.yaml"
      --head-name "$HEAD_NAME"
      --train-features "$FEATURE_DIR/train_pair_features.csv"
      --output "$MODEL_DIR/${HEAD_NAME}.pt"
      --device "$DEVICE"
    )
    if [[ -n "$MAX_STEPS" ]]; then
      TRAIN_ARGS+=(--max-steps "$MAX_STEPS")
    fi
    PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" "${TRAIN_ARGS[@]}"
  done

  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" "$ROOT_DIR/models_src/heading_baseline.py" fit \
    --features "$FEATURE_DIR/train_pair_features.csv" \
    --output "$MODEL_DIR/heading_baseline.json"
fi

PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" "$ROOT_DIR/models_src/released_head_inference.py" \
  --feature-csv "$FEATURE_DIR/test_pair_features.csv" \
  --model-dir "$MODEL_DIR" \
  --output-dir "$PRED_DIR" \
  --device "$DEVICE" \
  "${LIMIT_ARGS[@]}"

PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" "$ROOT_DIR/models_src/heading_baseline.py" predict \
  --features "$FEATURE_DIR/test_pair_features.csv" \
  --model "$MODEL_DIR/heading_baseline.json" \
  --output "$PRED_DIR/heading_predictions.csv" \
  "${LIMIT_ARGS[@]}"

bash "$ROOT_DIR/scripts/reproduce_from_models.sh" \
  --data-root "$DATA_ROOT" \
  --feature-root "$PRED_DIR" \
  --model-dir "$ROOT_DIR/models" \
  --heading-transform "$ROOT_DIR/models/heading_transform_identity.json" \
  --output-dir "$SUBMISSION_DIR" \
  ${LIMIT:+--limit "$LIMIT"} \
  --skip-verify-sha

cat <<EOF
From-scratch public reproduction finished.

Output:
  $SUBMISSION_DIR/submit.zip

This path is an end-to-end public RASR pipeline. It is intended for transparent
method reproduction and independent experimentation, not bit-exact reproduction
of the archived online submission.
For exact reproduction, use the frozen artifact package and
scripts/reproduce_from_models.sh.
EOF
