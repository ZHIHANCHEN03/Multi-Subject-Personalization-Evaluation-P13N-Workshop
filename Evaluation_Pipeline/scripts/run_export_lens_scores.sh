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

python_can_import_unsloth() {
  local py_bin="$1"
  if [ ! -x "$py_bin" ]; then
    return 1
  fi
  PYTHONNOUSERSITE=1 "$py_bin" - <<'PY' >/dev/null 2>&1
import unsloth
print(getattr(unsloth, "__version__", "unknown"))
PY
}

bootstrap_eval_venv() {
  local venv_dir="$1"
  local base_python="$2"
  local pyvenv_cfg="$venv_dir/pyvenv.cfg"

  echo "[run_export_lens_scores] Bootstrapping local eval venv: $venv_dir"
  if [ -f "$pyvenv_cfg" ] && grep -q "include-system-site-packages = true" "$pyvenv_cfg"; then
    echo "[run_export_lens_scores] Removing contaminated eval venv (system-site-packages enabled)"
    rm -rf "$venv_dir"
  fi
  if [ ! -d "$venv_dir" ]; then
    "$base_python" -m venv "$venv_dir"
  fi

  local venv_python="$venv_dir/bin/python"
  local constraints_file="$venv_dir/constraints-export-lens.txt"

  cat > "$constraints_file" <<'EOF'
fsspec<=2025.9.0
trl>=0.18.2,<=0.24.0,!=0.19.0
torchao>=0.13.0
EOF

  "$venv_python" -m pip install --upgrade pip setuptools wheel
  "$venv_python" -m pip install --upgrade --force-reinstall --no-cache-dir \
    -c "$constraints_file" \
    "unsloth" "unsloth_zoo" "transformers==5.5.0" "peft" "accelerate" "pillow" "tqdm" "trl" "torchao"

  echo "[run_export_lens_scores] Eval venv ready: $venv_python"
  printf '%s\n' "$venv_python"
}

discover_python_bin() {
  local candidates=()
  local local_eval_venv="$SCRIPT_DIR/.venv-export-lens"
  local local_eval_python="$local_eval_venv/bin/python"

  if [ -n "${PYTHON_BIN:-}" ] && [ "$PYTHON_BIN" != "python3" ]; then
    candidates+=("$PYTHON_BIN")
  fi
  candidates+=(
    "$REPO_ROOT/Model_Training/.venv-a100-unsloth/bin/python"
    "$REPO_ROOT/.venv-a100-unsloth/bin/python"
    "$local_eval_python"
  )

  for candidate in "${candidates[@]}"; do
    if python_can_import_unsloth "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  if python_can_import_unsloth "python3"; then
    printf '%s\n' "python3"
    return 0
  fi

  bootstrap_eval_venv "$local_eval_venv" "python3"
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
  OUTPUT_PATH="$SCRIPT_DIR/${MANIFEST_STEM}_lens_scores_all6.jsonl"
fi

if [ -n "$MANIFEST_PATH" ]; then
  MANIFEST_BASENAME="$(basename "$MANIFEST_PATH")"
  MANIFEST_STEM="${MANIFEST_BASENAME%.jsonl}"
else
  MANIFEST_STEM="auto_manifest"
fi
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_PATH="${LOG_PATH:-$SCRIPT_DIR/${MANIFEST_STEM}_lens_scores_all6_${TIMESTAMP}.log}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNS_ROOT="${RUNS_ROOT:-}"
DATASET_BASE_DIR="${DATASET_BASE_DIR:-$REPO_ROOT}"
LOG_EVERY="${LOG_EVERY:-20}"
PAIR_BATCH_SIZE="${PAIR_BATCH_SIZE:-5}"
PAIR_BATCH_SIZE_4B="${PAIR_BATCH_SIZE_4B:-2}"
MODEL_GROUP="${MODEL_GROUP:-all}"
PYTHON_BIN="$(discover_python_bin)"

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
echo "[run_export_lens_scores] Metric models  : all 6 ready models"
echo "[run_export_lens_scores] Per-model out  : ${OUTPUT_PATH%.jsonl}__<metrics_alias>.jsonl"
echo "[run_export_lens_scores] Log every      : $LOG_EVERY pairs"
echo "[run_export_lens_scores] Pair batch     : $PAIR_BATCH_SIZE"
echo "[run_export_lens_scores] 4B pair batch  : $PAIR_BATCH_SIZE_4B"
echo "[run_export_lens_scores] Model group    : $MODEL_GROUP"
echo "============================================================"

{
  PYTHONNOUSERSITE=1 "$PYTHON_BIN" - <<'PY'
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
    --pair_batch_size "$PAIR_BATCH_SIZE"
    --pair_batch_size_4b "$PAIR_BATCH_SIZE_4B"
    --metric_model_group "$MODEL_GROUP"
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

  PYTHONNOUSERSITE=1 "${CMD[@]}"
} 2>&1 | tee "$LOG_PATH"

echo "============================================================"
echo "[run_export_lens_scores] Done"
echo "[run_export_lens_scores] Output written to: $OUTPUT_PATH"
echo "[run_export_lens_scores] Log written to   : $LOG_PATH"
echo "============================================================"
