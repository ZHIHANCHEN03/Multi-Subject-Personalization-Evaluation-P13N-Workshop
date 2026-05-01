#!/bin/bash
# ==============================================================================
# LENS Training & Evaluation Pipeline 
# Hardware: NVIDIA A100 (80GB) | Environment: Linux / RunPod
# ==============================================================================

# Exit on errors, undefined vars, and failed piped commands.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "🚀 Initializing LENS Pipeline on A100 Server..."
echo "======================================================================"

# 0. Backbone Option
# Usage examples:
#   bash run_a100_pipeline.sh
#   MODEL_NAME=unsloth/Qwen3.5-4B bash run_a100_pipeline.sh
#   MODEL_NAME=unsloth/Qwen3.5-2B RUN_LORA_LAYER=0 bash run_a100_pipeline.sh
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3.5-0.8B}"
DATA_VERSION="${DATA_VERSION:-v2}"
RUN_LAYER_ONLY="${RUN_LAYER_ONLY:-1}"
RUN_LORA_LAYER="${RUN_LORA_LAYER:-0}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
INSTALL_FASTPATH_DEPS="${INSTALL_FASTPATH_DEPS:-0}"
MAX_JOBS="${MAX_JOBS:-8}"
IMAGE_A_ROOT="${IMAGE_A_ROOT:-/workspace/data/A}"
IMAGE_B_ROOT="${IMAGE_B_ROOT:-/workspace/data/B}"
IMAGE_A_EXT="${IMAGE_A_EXT:-.png}"
IMAGE_B_EXT="${IMAGE_B_EXT:-.jpg}"
REFS_ROOT="${REFS_ROOT:-$SCRIPT_DIR/data_v2/refs}"
TRAIN_PATH="${TRAIN_PATH:-}"
VAL_PATH="${VAL_PATH:-}"
TEST_PATH="${TEST_PATH:-}"
RUNS_ROOT="${RUNS_ROOT:-/workspace/Model_Training_runs}"
AUTO_SCALE_TARGET_EBS="${AUTO_SCALE_TARGET_EBS:-16}"
AUTO_SCALE_TARGET_UPDATES="${AUTO_SCALE_TARGET_UPDATES:-6000}"
AUTO_SCALE_MIN_EPOCHS="${AUTO_SCALE_MIN_EPOCHS:-2}"
AUTO_SCALE_MAX_EPOCHS="${AUTO_SCALE_MAX_EPOCHS:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-3407}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
ALPHA="${ALPHA:-1.0}"
BETA="${BETA:-0.5}"
ADAPTIVE_BETA="${ADAPTIVE_BETA:-false}"
BETA_FINAL="${BETA_FINAL:-0.5}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-1}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.001}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"

if [ "$ADAPTIVE_BETA" = "true" ]; then
  ADAPTIVE_BETA_FLAG="--adaptive_beta"
else
  ADAPTIVE_BETA_FLAG="--no-adaptive_beta"
fi

if [ -z "${BATCH_SIZE:-}" ]; then
  case "$MODEL_NAME" in
    "unsloth/Qwen3.5-0.8B") BATCH_SIZE=1 ;;  # VRAM Limit on A100 80GB (Due to Siamese + Flash Attention Fallback)
    "unsloth/Qwen3.5-2B")   BATCH_SIZE=1 ;;  # VRAM Limit
    "unsloth/Qwen3.5-4B")   BATCH_SIZE=1 ;;  # VRAM Limit
    "unsloth/Qwen3.5-9B")   BATCH_SIZE=1 ;;  # VRAM Limit
    *)                      BATCH_SIZE=1 ;;
  esac
fi

echo "✅ [0/5] Selected backbone: $MODEL_NAME"
echo "✅ [0/5] Selected data version: $DATA_VERSION"
echo "✅ [0/5] Selected physical batch size: $BATCH_SIZE (Effective Batch Size & Epochs will be AUTO-SCALED)"
echo "✅ [0/5] Selected image size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "✅ [0/5] Fair-train protocol: EBS=${AUTO_SCALE_TARGET_EBS}, target_updates=${AUTO_SCALE_TARGET_UPDATES}, lr=${LR}, wd=${WEIGHT_DECAY}, seed=${SEED}"
echo "✅ [0/5] Optional FlashAttention install: ${INSTALL_FLASH_ATTN}"
echo "✅ [0/5] Optional fast-path deps install: ${INSTALL_FASTPATH_DEPS}"

if [ "$DATA_VERSION" != "v2" ]; then
  echo "❌ This public release only supports DATA_VERSION=v2."
  exit 1
fi

# 1. Environment Setup (Safe Cache Storage)
# IMPORTANT:
# Some RunPod environments expose /workspace with large capacity in `df -h`,
# but Hugging Face Xet downloads can still fail there with "Disk quota exceeded".
# We therefore default to the local overlay disk, which has ample free space.
export HF_HOME="$HOME/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"
echo "✅ [1/5] HF cache directory set to: $HF_HOME"
echo "✅ [1/5] Xet disabled via HF_HUB_DISABLE_XET=1"

# 2. Dependency Management
echo "⏳ [2/5] Creating isolated A100 training environment..."
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv-a100-unsloth}"
REBUILD_VENV="${REBUILD_VENV:-0}"
if [ -d "$VENV_DIR" ] && [ "$REBUILD_VENV" = "1" ]; then
  echo "🧹 [2/5] Rebuilding virtualenv as requested: $VENV_DIR"
  rm -rf "$VENV_DIR"
fi
if [ ! -d "$VENV_DIR" ]; then
  echo "🧱 [2/5] Creating fresh virtualenv at: $VENV_DIR"
  python3 -m venv "$VENV_DIR"
else
  echo "♻️  [2/5] Reusing existing virtualenv: $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "🐍 [2/5] Using python: $(which python)"
echo "📦 [2/5] Using pip: $(which pip)"
python --version
pip --version
CONSTRAINTS_FILE="$VENV_DIR/constraints-a100.txt"
echo "🧱 [2/5] Writing A100 training constraints..."
cat > "$CONSTRAINTS_FILE" <<'EOF'
torch==2.10.0
torchvision==0.25.0
torchaudio==2.10.0
transformers==5.5.0
fsspec<=2025.9.0
EOF
if [ "$REBUILD_VENV" = "1" ] || [ ! -f "$VENV_DIR/.deps_ready" ]; then
  echo "⬆️  [2/5] Upgrading pip/setuptools/wheel..."
  python -m pip install --upgrade pip setuptools wheel
  echo "🧰 [2/5] Installing common build helpers..."
  python -m pip install --upgrade --no-cache-dir packaging ninja
  echo "🔥 [2/5] Installing pinned torch stack for A100 training (cu128)..."
  python -m pip install --upgrade --force-reinstall --no-cache-dir torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
  echo "🧩 [2/5] Installing pinned Python stack..."
  python -m pip install --upgrade --force-reinstall --no-cache-dir -c "$CONSTRAINTS_FILE" transformers==5.5.0 peft pillow 'fsspec<=2025.9.0'
  echo "🦥 [2/5] Installing Unsloth under the same constraints..."
  python -m pip install --upgrade --force-reinstall --no-cache-dir -c "$CONSTRAINTS_FILE" unsloth unsloth_zoo

  if [ "$INSTALL_FLASH_ATTN" = "1" ]; then
    echo "⚡ [2/5] Attempting FlashAttention 2 installation (best effort)..."
    if MAX_JOBS="$MAX_JOBS" python -m pip install --upgrade --no-cache-dir flash-attn==2.8.3 --no-build-isolation; then
      echo "✅ [2/5] FlashAttention 2 installed successfully."
    else
      echo "⚠️  [2/5] FlashAttention 2 install failed; training will fall back to Xformers/PyTorch SDPA."
    fi
  else
    echo "⏭️  [2/5] Skipping FlashAttention 2 installation by request."
  fi

  if [ "$INSTALL_FASTPATH_DEPS" = "1" ]; then
    echo "🚀 [2/5] Attempting optional fast-path deps install (best effort)..."
    if python -m pip install --upgrade --no-cache-dir 'flash-linear-attention[conv1d]'; then
      echo "✅ [2/5] Optional fast-path deps installed successfully."
    else
      echo "⚠️  [2/5] Optional fast-path deps install failed; Unsloth may use slower fallback kernels."
    fi
  else
    echo "⏭️  [2/5] Skipping optional fast-path deps installation by request."
  fi

  touch "$VENV_DIR/.deps_ready"
else
  echo "⚡ [2/5] Skipping dependency reinstall because cached environment is ready."
fi
echo "🔎 [2/5] Verifying final package versions..."
python - <<'PY'
import unsloth
import importlib
packages = ["torch", "torchvision", "torchaudio", "transformers", "peft", "unsloth", "unsloth_zoo", "fsspec"]
for name in packages:
    mod = importlib.import_module(name)
    print(f"   - {name}: {getattr(mod, '__version__', 'unknown')}")
PY
echo "🩺 [2/5] Running pip check..."
python -m pip check
echo "✅ [2/5] Isolated environment is ready at: $VENV_DIR"

# 3. Data Preparation
echo "⏳ [3/5] Cleaning raw data and generating Train/Val/Test splits..."
python scripts/build_v2_dataset.py --image_a_root "$IMAGE_A_ROOT" --image_b_root "$IMAGE_B_ROOT" --image_a_ext "$IMAGE_A_EXT" --image_b_ext "$IMAGE_B_EXT" --refs_root "$REFS_ROOT"
DEFAULT_TRAIN_PATH="$SCRIPT_DIR/data_v2/train_v2.json"
DEFAULT_VAL_PATH="$SCRIPT_DIR/data_v2/val_v2.json"
DEFAULT_TEST_PATH="$SCRIPT_DIR/data_v2/test_v2.json"
TRAIN_PATH="${TRAIN_PATH:-$DEFAULT_TRAIN_PATH}"
VAL_PATH="${VAL_PATH:-$DEFAULT_VAL_PATH}"
TEST_PATH="${TEST_PATH:-$DEFAULT_TEST_PATH}"
echo "✅ [3/5] Dataset built successfully."
echo "✅ [3/5] Train manifest: $TRAIN_PATH"
echo "✅ [3/5] Val manifest: $VAL_PATH"
echo "✅ [3/5] Test manifest: $TEST_PATH"

# Enable gradient checkpointing and memory expansion for massive VLM training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false # Limits threads to prevent CPU RAM OOM during compilation/loading

# Create logs directory
SAFE_MODEL_NAME=$(echo "$MODEL_NAME" | tr '/' '_')
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${RUNS_ROOT}/${DATA_VERSION}/${SAFE_MODEL_NAME}/${TIMESTAMP}"
OUTPUTS_DIR="${RUN_DIR}/outputs"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "$OUTPUTS_DIR" "$LOG_DIR"
echo "✅ [3.5/5] Run directory: $RUN_DIR"
echo "✅ [3.5/5] Logs directory: $LOG_DIR"
echo "✅ [3.5/5] Outputs directory: $OUTPUTS_DIR"

# 4. EXPERIMENT A: Layer-only
if [ "$RUN_LAYER_ONLY" = "1" ]; then
  echo "======================================================================"
  echo "⏳ [4/6] EXPERIMENT A: Initiating Joint Training (Layer-only mode)..."
  echo "======================================================================"
  LOG_FILE_A="$LOG_DIR/${SAFE_MODEL_NAME}_layer_only_${TIMESTAMP}.log"
  echo "📝 Logging output to $LOG_FILE_A"
  
  python scripts/train.py --model_name "$MODEL_NAME" --mode layer_only --unfreeze_layers 4 --batch_size "$BATCH_SIZE" --image_size "$IMAGE_SIZE" --train_path "$TRAIN_PATH" --val_path "$VAL_PATH" --outputs_dir "$OUTPUTS_DIR" --num_workers "$NUM_WORKERS" --seed "$SEED" --lr "$LR" --weight_decay "$WEIGHT_DECAY" --warmup_ratio "$WARMUP_RATIO" --alpha "$ALPHA" --beta "$BETA" "$ADAPTIVE_BETA_FLAG" --beta_final "$BETA_FINAL" --grad_clip_norm "$GRAD_CLIP_NORM" --early_stopping_patience "$EARLY_STOPPING_PATIENCE" --early_stopping_min_delta "$EARLY_STOPPING_MIN_DELTA" --auto_scale --auto_scale_target_ebs "$AUTO_SCALE_TARGET_EBS" --auto_scale_target_updates "$AUTO_SCALE_TARGET_UPDATES" --auto_scale_min_epochs "$AUTO_SCALE_MIN_EPOCHS" --auto_scale_max_epochs "$AUTO_SCALE_MAX_EPOCHS" 2>&1 | tee "$LOG_FILE_A"
  echo "✅ [4/6] Training A completed."

  echo "⏳ Running Benchmark Evaluation for Experiment A..."
  python scripts/evaluate_pipeline.py --model_name "$MODEL_NAME" --mode layer_only --test_path "$TEST_PATH" --outputs_dir "$OUTPUTS_DIR" --image_size "$IMAGE_SIZE" 2>&1 | tee -a "$LOG_FILE_A"
  echo "✅ Evaluation A completed."
else
  echo "⏭️  [4/6] Skipping layer-only experiment. Set RUN_LAYER_ONLY=1 to enable."
fi

# 5. EXPERIMENT B: LoRA + Layer
echo "======================================================================"
echo "⏳ [5/6] EXPERIMENT B: Initiating Joint Training (LoRA + Layer mode)..."
echo "======================================================================"
if [ "$RUN_LORA_LAYER" = "1" ]; then
  LOG_FILE_B="$LOG_DIR/${SAFE_MODEL_NAME}_lora_layer_${TIMESTAMP}.log"
  echo "📝 Logging output to $LOG_FILE_B"
  
  python scripts/train.py --model_name "$MODEL_NAME" --mode lora_layer --unfreeze_layers 4 --batch_size "$BATCH_SIZE" --image_size "$IMAGE_SIZE" --train_path "$TRAIN_PATH" --val_path "$VAL_PATH" --outputs_dir "$OUTPUTS_DIR" --num_workers "$NUM_WORKERS" --seed "$SEED" --lr "$LR" --weight_decay "$WEIGHT_DECAY" --warmup_ratio "$WARMUP_RATIO" --alpha "$ALPHA" --beta "$BETA" "$ADAPTIVE_BETA_FLAG" --beta_final "$BETA_FINAL" --grad_clip_norm "$GRAD_CLIP_NORM" --early_stopping_patience "$EARLY_STOPPING_PATIENCE" --early_stopping_min_delta "$EARLY_STOPPING_MIN_DELTA" --auto_scale --auto_scale_target_ebs "$AUTO_SCALE_TARGET_EBS" --auto_scale_target_updates "$AUTO_SCALE_TARGET_UPDATES" --auto_scale_min_epochs "$AUTO_SCALE_MIN_EPOCHS" --auto_scale_max_epochs "$AUTO_SCALE_MAX_EPOCHS" 2>&1 | tee "$LOG_FILE_B"
  echo "✅ [5/6] Training B completed."
else
  echo "⏭️  [5/6] Skipping LoRA + Layer experiment. Set RUN_LORA_LAYER=1 to enable."
fi

if [ "$RUN_LORA_LAYER" = "1" ]; then
  echo "⏳ Running Benchmark Evaluation for Experiment B..."
  python scripts/evaluate_pipeline.py --model_name "$MODEL_NAME" --mode lora_layer --test_path "$TEST_PATH" --outputs_dir "$OUTPUTS_DIR" --image_size "$IMAGE_SIZE" 2>&1 | tee -a "$LOG_FILE_B"
  echo "✅ Evaluation B completed."
fi

echo "======================================================================"
echo "🎉 Pipeline finished successfully! Check terminal output for Ablation metrics."
echo "======================================================================"
