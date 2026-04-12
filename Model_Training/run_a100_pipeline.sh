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

# 1. Environment Setup (Safe Cache Storage)
# IMPORTANT:
# Some RunPod environments expose /workspace with large capacity in `df -h`,
# but Hugging Face Xet downloads can still fail there with "Disk quota exceeded".
# We therefore default to the local overlay disk, which has ample free space.
export HF_HOME="/root/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"
echo "✅ [1/5] HF cache directory set to: $HF_HOME"
echo "✅ [1/5] Xet disabled via HF_HUB_DISABLE_XET=1"

# 2. Dependency Management
echo "⏳ [2/5] Updating transformers to development branch (required for Qwen3.5/Qwen3-VL)..."
pip install -q git+https://github.com/huggingface/transformers.git
echo "✅ [2/5] Transformers updated."

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
# Reduce batch size for 9B multi-image training
python scripts/train.py --model_name Qwen/Qwen3.5-9B-Base --mode partial --unfreeze_layers 4 --batch_size 2 --epochs 5
echo "✅ [4/6] Training A completed."

echo "⏳ Running Benchmark Evaluation for Experiment A..."
python scripts/evaluate_pipeline.py
echo "✅ Evaluation A completed."

# 5. Model Training & Eval: EXPERIMENT B (LoRA Adapters)
echo "======================================================================"
echo "⏳ [5/6] EXPERIMENT B: Initiating Joint Training (LoRA mode)..."
echo "======================================================================"
python scripts/train.py --model_name Qwen/Qwen3.5-9B-Base --mode lora --batch_size 2 --epochs 5
echo "✅ [5/6] Training B completed."

echo "⏳ Running Benchmark Evaluation for Experiment B..."
python scripts/evaluate_pipeline.py
echo "✅ Evaluation B completed."

echo "======================================================================"
echo "🎉 Pipeline finished successfully! Check terminal output for Ablation metrics."
echo "======================================================================"
