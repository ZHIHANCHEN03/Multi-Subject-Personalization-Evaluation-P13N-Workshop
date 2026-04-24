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
RUN_LAYER_ONLY="${RUN_LAYER_ONLY:-1}"
RUN_LORA_LAYER="${RUN_LORA_LAYER:-1}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"

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
echo "✅ [0/5] Selected physical batch size: $BATCH_SIZE (Effective Batch Size & Epochs will be AUTO-SCALED)"
echo "✅ [0/5] Selected image size: ${IMAGE_SIZE}x${IMAGE_SIZE}"

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
  echo "🔥 [2/5] Installing pinned torch stack for A100 training (cu128)..."
  python -m pip install --upgrade --force-reinstall --no-cache-dir torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
  echo "🧩 [2/5] Installing pinned Python stack..."
  python -m pip install --upgrade --force-reinstall --no-cache-dir -c "$CONSTRAINTS_FILE" transformers==5.5.0 peft pillow 'fsspec<=2025.9.0'
  echo "🦥 [2/5] Installing Unsloth under the same constraints..."
  python -m pip install --upgrade --force-reinstall --no-cache-dir -c "$CONSTRAINTS_FILE" unsloth unsloth_zoo
  echo "⚡ [2/5] Skipping Flash Attention 2 strict installation (Falling back to Xformers/PyTorch SDPA)..."
  # python -m pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir
  touch "$VENV_DIR/.deps_ready"
else
  echo "⚡ [2/5] Skipping dependency reinstall because cached environment is ready."
fi
echo "🔎 [2/5] Verifying final package versions..."
python - <<'PY'
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
python scripts/build_v1_dataset.py
echo "✅ [3/5] Dataset built successfully."

# Enable gradient checkpointing and memory expansion for massive VLM training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false # Limits threads to prevent CPU RAM OOM during compilation/loading

# Create logs directory
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
SAFE_MODEL_NAME=$(echo "$MODEL_NAME" | tr '/' '_')
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 4. EXPERIMENT A: Layer-only
if [ "$RUN_LAYER_ONLY" = "1" ]; then
  echo "======================================================================"
  echo "⏳ [4/6] EXPERIMENT A: Initiating Joint Training (Layer-only mode)..."
  echo "======================================================================"
  LOG_FILE_A="$LOG_DIR/${SAFE_MODEL_NAME}_layer_only_${TIMESTAMP}.log"
  echo "📝 Logging output to $LOG_FILE_A"
  
  python scripts/train.py --model_name "$MODEL_NAME" --mode layer_only --unfreeze_layers 4 --batch_size "$BATCH_SIZE" --image_size "$IMAGE_SIZE" --auto_scale 2>&1 | tee "$LOG_FILE_A"
  echo "✅ [4/6] Training A completed."

  echo "⏳ Running Benchmark Evaluation for Experiment A..."
  python scripts/evaluate_pipeline.py --model_name "$MODEL_NAME" --mode layer_only --image_size "$IMAGE_SIZE" 2>&1 | tee -a "$LOG_FILE_A"
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
  
  python scripts/train.py --model_name "$MODEL_NAME" --mode lora_layer --unfreeze_layers 4 --batch_size "$BATCH_SIZE" --image_size "$IMAGE_SIZE" --auto_scale 2>&1 | tee "$LOG_FILE_B"
  echo "✅ [5/6] Training B completed."
else
  echo "⏭️  [5/6] Skipping LoRA + Layer experiment. Set RUN_LORA_LAYER=1 to enable."
fi

if [ "$RUN_LORA_LAYER" = "1" ]; then
  echo "⏳ Running Benchmark Evaluation for Experiment B..."
  python scripts/evaluate_pipeline.py --model_name "$MODEL_NAME" --mode lora_layer --image_size "$IMAGE_SIZE" 2>&1 | tee -a "$LOG_FILE_B"
  echo "✅ Evaluation B completed."
fi

echo "======================================================================"
echo "🎉 Pipeline finished successfully! Check terminal output for Ablation metrics."
echo "======================================================================"
