#!/bin/bash
# ==============================================================================
# LENS Training & Evaluation Pipeline 
# Hardware: NVIDIA A100 (80GB) | Environment: Linux / RunPod
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

echo "======================================================================"
echo "🚀 Initializing LENS Pipeline on A100 Server..."
echo "======================================================================"

# 0. Backbone Option
# Usage examples:
#   bash run_a100_pipeline.sh
#   MODEL_NAME=unsloth/Qwen3.5-4B bash run_a100_pipeline.sh
#   MODEL_NAME=unsloth/Qwen3.5-2B BATCH_SIZE=8 bash run_a100_pipeline.sh
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3.5-0.8B}"

if [ -z "${BATCH_SIZE:-}" ]; then
  case "$MODEL_NAME" in
    "unsloth/Qwen3.5-0.8B") BATCH_SIZE=16 ;;
    "unsloth/Qwen3.5-2B")   BATCH_SIZE=8 ;;
    "unsloth/Qwen3.5-4B")   BATCH_SIZE=4 ;;
    "unsloth/Qwen3.5-9B")   BATCH_SIZE=2 ;;
    *)                      BATCH_SIZE=2 ;;
  esac
fi

echo "✅ [0/5] Selected backbone: $MODEL_NAME"
echo "✅ [0/5] Selected batch size: $BATCH_SIZE"

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
VENV_DIR="${VENV_DIR:-$PWD/.venv-a100-unsloth}"
if [ -d "$VENV_DIR" ]; then
  echo "🧹 [2/5] Removing stale virtualenv: $VENV_DIR"
  rm -rf "$VENV_DIR"
fi
echo "🧱 [2/5] Creating fresh virtualenv at: $VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
echo "🐍 [2/5] Using python: $(which python)"
echo "📦 [2/5] Using pip: $(which pip)"
python --version
pip --version
echo "⬆️  [2/5] Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel
echo "🧱 [2/5] Writing A100 training constraints..."
cat > "$VENV_DIR/constraints-a100.txt" <<'EOF'
torch==2.10.0
torchvision==0.25.0
torchaudio==2.10.0
transformers==5.5.0
fsspec==2025.9.0
EOF
echo "🔥 [2/5] Installing pinned torch stack for A100 training (cu128)..."
python -m pip install --upgrade --force-reinstall --no-cache-dir torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
echo "🧩 [2/5] Installing pinned Python stack..."
python -m pip install --upgrade --force-reinstall --no-cache-dir -c "$VENV_DIR/constraints-a100.txt" transformers==5.5.0 peft pillow fsspec==2025.9.0
echo "🦥 [2/5] Installing Unsloth under the same constraints..."
python -m pip install --upgrade --force-reinstall --no-cache-dir -c "$VENV_DIR/constraints-a100.txt" unsloth unsloth_zoo
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # Limits threads to prevent CPU RAM OOM during compilation/loading

# 4. Model Training & Eval: EXPERIMENT A (Layer Unfreezing)
echo "======================================================================"
echo "⏳ [4/6] EXPERIMENT A: Initiating Joint Training (Layer Unfreezing mode)..."
echo "======================================================================"
# Reduce batch size as backbone grows; can be overridden via BATCH_SIZE=...
python scripts/train.py --model_name "$MODEL_NAME" --mode partial --unfreeze_layers 4 --batch_size "$BATCH_SIZE" --epochs 5
echo "✅ [4/6] Training A completed."

echo "⏳ Running Benchmark Evaluation for Experiment A..."
python scripts/evaluate_pipeline.py
echo "✅ Evaluation A completed."

# 5. Model Training & Eval: EXPERIMENT B (LoRA Adapters)
echo "======================================================================"
echo "⏳ [5/6] EXPERIMENT B: Initiating Joint Training (LoRA mode)..."
echo "======================================================================"
python scripts/train.py --model_name "$MODEL_NAME" --mode lora --batch_size "$BATCH_SIZE" --epochs 5
echo "✅ [5/6] Training B completed."

echo "⏳ Running Benchmark Evaluation for Experiment B..."
python scripts/evaluate_pipeline.py
echo "✅ Evaluation B completed."

echo "======================================================================"
echo "🎉 Pipeline finished successfully! Check terminal output for Ablation metrics."
echo "======================================================================"
