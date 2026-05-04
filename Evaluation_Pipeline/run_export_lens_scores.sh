#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

discover_manifest() {
  shopt -s nullglob
  local candidates=("$SCRIPT_DIR"/*_manifest.jsonl)
  shopt -u nullglob

  if [ "${#candidates[@]}" -eq 1 ]; then
    printf '%s\n' "${candidates[0]}"
    return 0
  fi

  if [ "${#candidates[@]}" -gt 1 ]; then
    echo "Error: multiple manifest files found in $SCRIPT_DIR" >&2
    printf ' - %s\n' "${candidates[@]}" >&2
    echo "Please pass one explicitly, e.g.:" >&2
    echo "  bash run_export_lens_scores.sh /path/to/v10_manifest.jsonl" >&2
    return 1
  fi

  return 0
}

MANIFEST_PATH="${MANIFEST_PATH:-}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
DATASET_ROOT="${DATASET_ROOT:-}"
AUTO_MANIFEST_OUTPUT="${AUTO_MANIFEST_OUTPUT:-$SCRIPT_DIR/auto_manifest.jsonl}"

if [ "$#" -ge 1 ]; then
  MANIFEST_PATH="$1"
fi

if [ "$#" -ge 2 ]; then
  OUTPUT_PATH="$2"
fi

if [ -z "$MANIFEST_PATH" ]; then
  MANIFEST_PATH="$(discover_manifest)"
fi

if [ -n "$MANIFEST_PATH" ] && [ ! -f "$MANIFEST_PATH" ]; then
  echo "Error: manifest not found: $MANIFEST_PATH"
  exit 1
fi

if [ -z "$OUTPUT_PATH" ]; then
  if [ -n "$MANIFEST_PATH" ]; then
    MANIFEST_BASENAME="$(basename "$MANIFEST_PATH")"
    MANIFEST_STEM="${MANIFEST_BASENAME%.jsonl}"
  else
    MANIFEST_STEM="auto_manifest"
  fi
  OUTPUT_PATH="$SCRIPT_DIR/${MANIFEST_STEM}_lens_scores_all5.jsonl"
fi

if [ -n "$MANIFEST_PATH" ]; then
  MANIFEST_BASENAME="$(basename "$MANIFEST_PATH")"
  MANIFEST_STEM="${MANIFEST_BASENAME%.jsonl}"
else
  MANIFEST_STEM="auto_manifest"
fi
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_PATH="${LOG_PATH:-$SCRIPT_DIR/${MANIFEST_STEM}_lens_scores_all5_${TIMESTAMP}.log}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNS_ROOT="${RUNS_ROOT:-}"
DATASET_BASE_DIR="${DATASET_BASE_DIR:-$REPO_ROOT}"
LOG_EVERY="${LOG_EVERY:-20}"
DEFAULT_VENV_PYTHON="$REPO_ROOT/Model_Training/.venv-a100-unsloth/bin/python"

if [ -z "${PYTHON_BIN_OVERRIDE_APPLIED:-}" ] && [ "$PYTHON_BIN" = "python3" ] && [ -x "$DEFAULT_VENV_PYTHON" ]; then
  PYTHON_BIN="$DEFAULT_VENV_PYTHON"
fi

echo "============================================================"
echo "[run_export_lens_scores] Starting LENS score export"
if [ -n "$MANIFEST_PATH" ]; then
  echo "[run_export_lens_scores] Manifest       : $MANIFEST_PATH"
else
  echo "[run_export_lens_scores] Manifest       : <auto-build>"
fi
echo "[run_export_lens_scores] Dataset root   : ${DATASET_ROOT:-<auto-discover>}"
echo "[run_export_lens_scores] Auto manifest  : $AUTO_MANIFEST_OUTPUT"
echo "[run_export_lens_scores] Output         : $OUTPUT_PATH"
echo "[run_export_lens_scores] Log file       : $LOG_PATH"
echo "[run_export_lens_scores] Runs root      : ${RUNS_ROOT:-<auto-discover>}"
echo "[run_export_lens_scores] Dataset base   : $DATASET_BASE_DIR"
echo "[run_export_lens_scores] Python         : $PYTHON_BIN"
echo "[run_export_lens_scores] Metric models  : all 5 ready models"
echo "[run_export_lens_scores] Log every      : $LOG_EVERY pairs"
echo "============================================================"

{
  "$PYTHON_BIN" - <<'PY'
import importlib
import sys
print("[env] python_executable =", sys.executable)
print("[env] python_version    =", sys.version.replace("\n", " "))
for name in ["unsloth", "unsloth_zoo", "transformers", "peft", "torch"]:
    try:
        mod = importlib.import_module(name)
        print(f"[env] {name:14s} =", getattr(mod, "__version__", "unknown"))
    except Exception as exc:
        print(f"[env] {name:14s} = IMPORT_FAIL ({type(exc).__name__}: {exc})")
PY

  CMD=(
    "$PYTHON_BIN" "$SCRIPT_DIR/export_lens_scores.py"
    --output "$OUTPUT_PATH"
    --runs_root "$RUNS_ROOT"
    --dataset_base_dir "$DATASET_BASE_DIR"
    --log_every "$LOG_EVERY"
    --auto_manifest_output "$AUTO_MANIFEST_OUTPUT"
  )

  if [ -n "$MANIFEST_PATH" ]; then
    CMD+=(--manifest "$MANIFEST_PATH")
  fi

  if [ -n "$DATASET_ROOT" ]; then
    CMD+=(--dataset_root "$DATASET_ROOT")
  fi

  if [ -n "$RUNS_ROOT" ]; then
    CMD+=(--runs_root "$RUNS_ROOT")
  fi

  "${CMD[@]}"
} 2>&1 | tee "$LOG_PATH"

echo "============================================================"
echo "[run_export_lens_scores] Done"
echo "[run_export_lens_scores] Output written to: $OUTPUT_PATH"
echo "[run_export_lens_scores] Log written to   : $LOG_PATH"
echo "============================================================"
