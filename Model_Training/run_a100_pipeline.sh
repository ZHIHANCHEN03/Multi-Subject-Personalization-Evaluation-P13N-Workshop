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
echo "⏳ [2/5] Force aligning PyTorch and Unsloth ecosystem..."
# 1. Align core torch ecosystem to 2.4.0 with CU121
pip install -q torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# 2. Install correct xformers and torchao for unsloth 2026.4.4
pip install -q xformers==0.0.27.post2 torchao==0.13.0
# 3. Install stable transformers (avoiding the bleeding edge 5.6.0.dev0 that breaks unsloth)
pip install -q transformers==5.5.0
# 4. Install unsloth
pip install -q unsloth unsloth_zoo
echo "✅ [2/5] Environment aligned and ready."

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
