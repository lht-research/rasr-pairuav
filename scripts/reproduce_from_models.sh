#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

DATA_ROOT=""
FEATURE_ROOT=""
OUTPUT_DIR=""
MODEL_DIR="$ROOT_DIR/models"
LASTMETER_CONFIG=""
HEADING_TRANSFORM=""
PAIR_FEATURE_CSV=""
PAIR_FEATURE_GLOB=""
PAIR_FEATURE_DIR=""
HEADING_CSV=""
SELFPAIR_INDICES=""
BATCH_SIZE=""
DEVICE=""
LEGACY_BATCH_POLICY="--legacy-batch-policy"
LIMIT=""
EXPECTED_ROWS="2773116"
EXPECTED_SELFPAIR_ROWS="51354"
EXPECTED_SHA256="2f742f2eff83e535b96a8dbd46db370fa3ac0538a9f3e53b684d65c253b34b77"
SKIP_VERIFY_SHA=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --feature-root)
      FEATURE_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --lastmeter-config)
      LASTMETER_CONFIG="$2"
      shift 2
      ;;
    --heading-transform)
      HEADING_TRANSFORM="$2"
      shift 2
      ;;
    --pair-feature-csv)
      PAIR_FEATURE_CSV="$2"
      shift 2
      ;;
    --pair-feature-glob)
      PAIR_FEATURE_GLOB="$2"
      shift 2
      ;;
    --pair-feature-dir)
      PAIR_FEATURE_DIR="$2"
      shift 2
      ;;
    --heading-csv)
      HEADING_CSV="$2"
      shift 2
      ;;
    --selfpair-indices)
      SELFPAIR_INDICES="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --no-legacy-batch-policy)
      LEGACY_BATCH_POLICY="--no-legacy-batch-policy"
      shift
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --expected-rows)
      EXPECTED_ROWS="$2"
      shift 2
      ;;
    --expected-selfpair-rows)
      EXPECTED_SELFPAIR_ROWS="$2"
      shift 2
      ;;
    --skip-verify-sha)
      SKIP_VERIFY_SHA=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$DATA_ROOT" || -z "$FEATURE_ROOT" || -z "$OUTPUT_DIR" ]]; then
  echo "Usage: $0 --data-root PATH --feature-root PATH --output-dir PATH [--model-dir PATH] [--limit N]" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

INFER_ARGS=(
  "$ROOT_DIR/inference/run_inference.py"
  --feature-root "$FEATURE_ROOT"
  --model-dir "$MODEL_DIR"
  --output-dir "$OUTPUT_DIR"
  --expected-rows "$EXPECTED_ROWS"
  --expected-selfpair-rows "$EXPECTED_SELFPAIR_ROWS"
)
VERIFY_ARGS=(
  "$ROOT_DIR/scripts/verify_result.py"
  --result-zip "$OUTPUT_DIR/result.zip"
  --expected-rows "$EXPECTED_ROWS"
  --expected-selfpair-rows "$EXPECTED_SELFPAIR_ROWS"
)

if [[ -z "$LASTMETER_CONFIG" ]]; then
  LASTMETER_CONFIG="$MODEL_DIR/lastmeter_config.json"
fi
if [[ -z "$HEADING_TRANSFORM" ]]; then
  HEADING_TRANSFORM="$MODEL_DIR/heading_transform.json"
fi
INFER_ARGS+=(--lastmeter-config "$LASTMETER_CONFIG")
INFER_ARGS+=(--heading-transform "$HEADING_TRANSFORM")

if [[ -n "$LIMIT" ]]; then
  INFER_ARGS+=(--limit "$LIMIT")
  VERIFY_ARGS=( "$ROOT_DIR/scripts/verify_result.py" --result-zip "$OUTPUT_DIR/result.zip" --expected-rows "$LIMIT" --skip-sha256 )
else
  if [[ -n "$SELFPAIR_INDICES" ]]; then
    VERIFY_ARGS+=(--selfpair-indices "$SELFPAIR_INDICES")
  fi
  if [[ "$SKIP_VERIFY_SHA" -eq 1 ]]; then
    VERIFY_ARGS+=(--skip-sha256)
  fi
fi

if [[ -n "$PAIR_FEATURE_CSV" ]]; then
  INFER_ARGS+=(--pair-feature-csv "$PAIR_FEATURE_CSV")
fi
if [[ -n "$PAIR_FEATURE_GLOB" ]]; then
  INFER_ARGS+=(--pair-feature-glob "$PAIR_FEATURE_GLOB")
fi
if [[ -n "$PAIR_FEATURE_DIR" ]]; then
  INFER_ARGS+=(--pair-feature-dir "$PAIR_FEATURE_DIR")
fi
if [[ -n "$HEADING_CSV" ]]; then
  INFER_ARGS+=(--heading-csv "$HEADING_CSV")
fi
if [[ -n "$SELFPAIR_INDICES" ]]; then
  INFER_ARGS+=(--selfpair-indices "$SELFPAIR_INDICES")
fi
if [[ -n "$BATCH_SIZE" ]]; then
  INFER_ARGS+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n "$DEVICE" ]]; then
  INFER_ARGS+=(--device "$DEVICE")
fi
INFER_ARGS+=("$LEGACY_BATCH_POLICY")

PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" "${INFER_ARGS[@]}"
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" "${VERIFY_ARGS[@]}"

cp -f "$OUTPUT_DIR/result.zip" "$OUTPUT_DIR/submit.zip"
if [[ -z "$LIMIT" && "$SKIP_VERIFY_SHA" -eq 0 ]]; then
  if command -v sha256sum >/dev/null 2>&1; then
    actual_sha="$(sha256sum "$OUTPUT_DIR/submit.zip" | awk '{print $1}')"
  else
    actual_sha="$("$PYTHON_BIN" - "$OUTPUT_DIR/submit.zip" <<'PY'
from pathlib import Path
import hashlib
import sys

print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
  fi
  if [[ "$actual_sha" != "$EXPECTED_SHA256" ]]; then
    echo "Unexpected submit.zip SHA256: $actual_sha expected $EXPECTED_SHA256" >&2
    exit 1
  fi
fi
