#!/usr/bin/env bash
# =============================================================================
# Diag-DPO / FLUX.2 全新 RunPod 环境搭建(只搭环境)
#
# 用法(新 pod 终端):
#   export HF_TOKEN=hf_xxxxx        # 已在网页接受 FLUX.2-dev license
#   bash setup_pod.sh
#
# 幂等: 重复运行安全(已装/已下的跳过)。
# =============================================================================
set -uo pipefail

FLUX_DIR=/root/flux2-dev           # 权重落点(overlay; volume 够大可改 /workspace/flux2-dev)
HF_CACHE=/workspace/hf_cache       # 下载中转缓存(大盘, 完事即删)

echo "==== 1. Python 环境 ===="
# ★ torch 必须 >=2.5: 2.4 的 infer_schema 不认 flash-attn-3 的 'float|None' 签名,
#   会在 import diffusers(Flux2) 时直接崩。2.5 才支持 PEP604 union。
python - <<'PY' || NEED_TORCH=1
import torch, sys
maj, mn = torch.__version__.split('.')[:2]
sys.exit(0 if (int(maj), int(mn)) >= (2, 5) else 1)
PY
if [ "${NEED_TORCH:-0}" = "1" ]; then
  pip install -q "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu124
fi
pip install -q diffusers transformers peft accelerate sentencepiece protobuf "huggingface_hub>=0.26"

echo "==== 2. FLUX.2-dev 权重(~178GB) ===="
if [ -f "$FLUX_DIR/model_index.json" ]; then
  echo "已存在, 跳过下载"
else
  [ -n "${HF_TOKEN:-}" ] || { echo "!! 需先 export HF_TOKEN=hf_xxx (且已接受 license)"; exit 1; }
  HF_HOME="$HF_CACHE" hf download black-forest-labs/FLUX.2-dev --local-dir "$FLUX_DIR" || exit 1
  rm -rf "$HF_CACHE"
fi

echo "==== 自检 ===="
python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, '|', torch.cuda.get_device_name(0))" || exit 1
python -c "from diffusers import Flux2Pipeline, Flux2Transformer2DModel; print('FLUX2 import OK')" || exit 1
[ -f "$FLUX_DIR/model_index.json" ] && echo "权重 OK: $FLUX_DIR" || { echo "权重缺失"; exit 1; }
echo "环境就绪。"
