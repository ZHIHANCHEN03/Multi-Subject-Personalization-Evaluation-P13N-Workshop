#!/bin/bash

# 遇到错误立即停止
set -e

# 定义路径 (动态获取当前脚本所在目录)
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$PIPELINE_DIR")" # 父目录 /Multi-Subject-Personalization-Evaluation-P13N-Workshop

V10_MANIFEST="$WORKSPACE_DIR/v10_manifest.jsonl"
V13_MANIFEST="$WORKSPACE_DIR/v13_manifest.jsonl"

echo "============================================================"
echo "� [Step 0/3] Checking Environment Dependencies..."
echo "============================================================"
if ! python3 -c "import unsloth" &> /dev/null; then
    echo "⚠️ Unsloth or other dependencies not found. Auto-installing..."
    bash "$PIPELINE_DIR/install_env.sh"
else
    echo "✅ Environment dependencies look good!"
fi
echo ""

echo "============================================================"
echo "�🚀 [Step 1/3] Generating Metadata Manifests (v10 and v13.2)"
echo "============================================================"
python3 $PIPELINE_DIR/dataset_loader.py
echo "✅ Metadata manifests generated successfully!"
echo ""

echo "============================================================"
echo "🚀 [Step 2/3] Running All Advanced Metrics (CLIP, DINO, SCR, REFVNLI...)"
echo "============================================================"
# 遍历需要跑的所有非 LLM 指标
for METRIC in clip_t clip_i dinov2 arcface image_reward scr refvnli; do
    echo "👉 Running [$METRIC] for v10..."
    python3 $PIPELINE_DIR/run_classic_metrics.py --manifest $V10_MANIFEST --metric $METRIC
    
    echo "👉 Running [$METRIC] for v13.2..."
    python3 $PIPELINE_DIR/run_classic_metrics.py --manifest $V13_MANIFEST --metric $METRIC
done
echo "✅ All classic & advanced metrics evaluation completed!"
echo ""

echo "============================================================"
echo "🚀 [Step 3/3] Running LLM-as-Judge (Unsloth Qwen2-VL)"
echo "============================================================"
# 遍历 Qwen2-VL 的两个模型尺寸
for SIZE in 2 7; do
    echo "👉 Running [Qwen2-VL ${SIZE}B] for v10..."
    python3 $PIPELINE_DIR/run_llm_judge_unsloth.py --manifest $V10_MANIFEST --size $SIZE
    
    echo "👉 Running [Qwen2-VL ${SIZE}B] for v13.2..."
    python3 $PIPELINE_DIR/run_llm_judge_unsloth.py --manifest $V13_MANIFEST --size $SIZE
done
echo "✅ LLM-as-Judge evaluation completed!"
echo ""

echo "============================================================"
echo "🎉 All pipelines executed successfully!"
echo "📂 Final output files with all metadata and scores:"
echo "   - $V10_MANIFEST"
echo "   - $V13_MANIFEST"
echo "============================================================"
