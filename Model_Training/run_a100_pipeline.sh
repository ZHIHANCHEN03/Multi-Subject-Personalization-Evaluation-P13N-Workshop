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
# We map HuggingFace downloads to /workspace to prevent filling up the root overlay
# and to ensure weights persist across instance restarts.
export HF_HOME="/workspace/huggingface_cache"
echo "✅ [1/5] Environment Variable HF_HOME set to: $HF_HOME"

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
python scripts/train.py --mode partial --unfreeze_layers 4 --batch_size 2 --epochs 5
echo "✅ [4/6] Training A completed."

echo "⏳ Running Benchmark Evaluation for Experiment A..."
python scripts/evaluate_pipeline.py
echo "✅ Evaluation A completed."

# 5. Model Training & Eval: EXPERIMENT B (LoRA Adapters)
echo "======================================================================"
echo "⏳ [5/6] EXPERIMENT B: Initiating Joint Training (LoRA mode)..."
echo "======================================================================"
python scripts/train.py --mode lora --batch_size 2 --epochs 5
echo "✅ [5/6] Training B completed."

echo "⏳ Running Benchmark Evaluation for Experiment B..."
python scripts/evaluate_pipeline.py
echo "✅ Evaluation B completed."

echo "======================================================================"
echo "🎉 Pipeline finished successfully! Check terminal output for Ablation metrics."
echo "======================================================================"
