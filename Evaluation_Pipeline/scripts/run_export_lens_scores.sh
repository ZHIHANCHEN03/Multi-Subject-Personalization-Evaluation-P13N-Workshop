#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$PIPELINE_ROOT/.." && pwd)"

discover_manifest() {
  shopt -s nullglob
  local candidates=(
    "$PIPELINE_ROOT/data/manifests"/*_manifest.jsonl
    "$SCRIPT_DIR"/*_manifest.jsonl
  )
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

python_can_import_runtime_stack() {
  local py_bin="$1"
  local extra_path="${2:-}"
  if [ ! -x "$py_bin" ]; then
    return 1
  fi
  PYTHONNOUSERSITE=1 PYTHONPATH="$extra_path${PYTHONPATH:+:$PYTHONPATH}" "$py_bin" - <<'PY' >/dev/null 2>&1
import importlib
modules = ["unsloth", "torch", "transformers", "peft", "PIL", "tqdm"]
for name in modules:
    importlib.import_module(name)
PY
}

bootstrap_eval_venv() {
  local venv_dir="$1"
  local py_bin="$venv_dir/bin/python"
  local marker="$venv_dir/.deps_ready_export"
  local constraints_file="$venv_dir/constraints-export.txt"

  echo "[run_export_lens_scores] Bootstrapping INDEPENDENT local eval venv at: $venv_dir" >&2
  mkdir -p "$venv_dir"
  if [ ! -x "$py_bin" ]; then
    python3 -m venv "$venv_dir"
  fi

  if [ -f "$marker" ] && python_can_import_runtime_stack "$py_bin"; then
    echo "[run_export_lens_scores] Reusing ready eval venv: $py_bin" >&2
    printf '%s\n' "$py_bin"
    return 0
  fi

  cat > "$constraints_file" <<'EOF'
torch==2.10.0
torchvision==0.25.0
torchaudio==2.10.0
transformers==5.5.0
fsspec<=2025.9.0
EOF

  echo "[run_export_lens_scores] Upgrading pip, setuptools, wheel..." >&2
  "$py_bin" -m pip install --upgrade pip setuptools wheel >&2
  echo "[run_export_lens_scores] Installing packaging, ninja..." >&2
  "$py_bin" -m pip install --upgrade --no-cache-dir packaging ninja >&2
  
  echo "[run_export_lens_scores] Installing PyTorch stack (with progress bar)..." >&2
  "$py_bin" -m pip install --upgrade --force-reinstall --progress-bar raw \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128 >&2
    
  echo "[run_export_lens_scores] Installing Transformers, Peft, Pillow..." >&2
  "$py_bin" -m pip install --upgrade --force-reinstall --progress-bar raw \
    -c "$constraints_file" transformers==5.5.0 peft pillow 'fsspec<=2025.9.0' >&2
    
  echo "[run_export_lens_scores] Installing Unsloth..." >&2
  "$py_bin" -m pip install --upgrade --force-reinstall --progress-bar raw \
    -c "$constraints_file" unsloth unsloth_zoo >&2

  if ! python_can_import_runtime_stack "$py_bin"; then
    echo "[run_export_lens_scores] Eval runtime install finished, but required imports still fail." >&2
    return 1
  fi

  touch "$marker"
  echo "[run_export_lens_scores] Eval runtime ready: $py_bin" >&2
  printf '%s\n' "$py_bin"
}

discover_python_bin() {
  local local_eval_venv="$SCRIPT_DIR/.venv-export-lens"
  local local_eval_python="$local_eval_venv/bin/python"

  if [ -n "${PYTHON_BIN:-}" ] && [ "$PYTHON_BIN" != "python3" ]; then
    if python_can_import_runtime_stack "$PYTHON_BIN"; then
      printf '%s\n' "$PYTHON_BIN"
      return 0
    fi
  fi

  # Always use the strictly independent local eval venv
  bootstrap_eval_venv "$local_eval_venv"
}

MANIFEST_PATH="${MANIFEST_PATH:-}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
DATASET_ROOT="${DATASET_ROOT:-}"
AUTO_MANIFEST_OUTPUT="${AUTO_MANIFEST_OUTPUT:-$PIPELINE_ROOT/data/manifests/auto_manifest.jsonl}"

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
  mkdir -p "$PIPELINE_ROOT/outputs/jsonl"
  if [ -n "$MANIFEST_PATH" ]; then
    MANIFEST_BASENAME="$(basename "$MANIFEST_PATH")"
    MANIFEST_STEM="${MANIFEST_BASENAME%.jsonl}"
  else
    MANIFEST_STEM="auto_manifest"
  fi
  OUTPUT_PATH="$PIPELINE_ROOT/outputs/jsonl/${MANIFEST_STEM}_lens_scores_all6.jsonl"
fi

if [ -n "$MANIFEST_PATH" ]; then
  MANIFEST_BASENAME="$(basename "$MANIFEST_PATH")"
  MANIFEST_STEM="${MANIFEST_BASENAME%.jsonl}"
else
  MANIFEST_STEM="auto_manifest"
fi
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
mkdir -p "$PIPELINE_ROOT/outputs/logs"
LOG_PATH="${LOG_PATH:-$PIPELINE_ROOT/outputs/logs/${MANIFEST_STEM}_lens_scores_all6_${TIMESTAMP}.log}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNS_ROOT="${RUNS_ROOT:-}"
DATASET_BASE_DIR="${DATASET_BASE_DIR:-$PROJECT_ROOT}"
LOG_EVERY="${LOG_EVERY:-20}"
BATCH_SIZE="${BATCH_SIZE:-}"
if [ -n "$BATCH_SIZE" ]; then
  PAIR_BATCH_SIZE="${PAIR_BATCH_SIZE:-$BATCH_SIZE}"
  PAIR_BATCH_SIZE_4B="${PAIR_BATCH_SIZE_4B:-$BATCH_SIZE}"
else
  PAIR_BATCH_SIZE="${PAIR_BATCH_SIZE:-5}"
  PAIR_BATCH_SIZE_4B="${PAIR_BATCH_SIZE_4B:-2}"
fi
MODEL_GROUP="${MODEL_GROUP:-all}"
PYTHON_BIN="$(discover_python_bin)"

case "$MODEL_GROUP" in
  all) METRIC_MODEL_LABEL="all 6 ready models" ;;
  08b) METRIC_MODEL_LABEL="0.8B group (2 models)" ;;
  2b) METRIC_MODEL_LABEL="2B group (2 models)" ;;
  4b) METRIC_MODEL_LABEL="4B group (2 models)" ;;
  *) METRIC_MODEL_LABEL="$MODEL_GROUP" ;;
esac

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
echo "[run_export_lens_scores] Metric models  : $METRIC_MODEL_LABEL"
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
