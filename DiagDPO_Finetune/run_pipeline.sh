#!/bin/bash
set -e

echo "====================================================="
echo "   Time-Aware Diag-DPO Full Pipeline for FLUX.2"
echo "====================================================="

# 1. 提纯数据集
echo ">> Step 1: Data Curation (Hard Negative Mining)"
python prepare_dataset.py \
    --input ../Model_Training/data_v2/train_v2.json \
    --output data/dpo_train_filtered.json \
    --min_delta 0.1

# 2. 开始端到端训练
echo ">> Step 2: Unsloth FLUX Fine-Tuning (Download, Load Data, Train)"
# 注意：在真实环境运行时，建议用 accelerate launch 启动
# accelerate launch train_flux_unsloth.py
python train_flux_unsloth.py

echo "====================================================="
echo "   Pipeline Complete! LoRA saved in outputs/"
echo "====================================================="
