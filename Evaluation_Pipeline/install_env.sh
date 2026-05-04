#!/bin/bash

# 遇到错误立即停止
set -e

echo "============================================================"
echo "🛠️  [Step 1/3] Installing Base Deep Learning Libraries..."
echo "============================================================"
# 确保安装了最新版的 PyTorch (Unsloth 需要 PyTorch >= 2.4.0 来支持 _inductor.config)
pip install --upgrade --force-reinstall "torch>=2.4.0" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "============================================================"
echo "🦥 [Step 2/3] Installing Unsloth (for Extreme Fast Qwen3.5 4-bit Inference)..."
echo "============================================================"
# 按照 Unsloth 官方要求安装
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes

echo "============================================================"
echo "📊 [Step 3/3] Installing Classic Metrics Dependencies..."
echo "============================================================"
# 安装 pandas, transformers, ImageReward, insightface 等
pip install -r requirements.txt

echo "============================================================"
echo "✅ All dependencies installed successfully!"
echo "🚀 You can now run: ./run_all_evals.sh"
echo "============================================================"
