#!/usr/bin/env bash
# Prepare every Round-1 dependency, including the Qwen/Unsloth MIE runtime.
# Run this on the CUDA GPU machine from any working directory.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT="$ROOT/external"
MODELS="$ROOT/models"
VENVS="$ROOT/.venvs"
PYTHON="${PYTHON:-$(command -v python3.11 || command -v python3)}"

mkdir -p "$EXT" "$MODELS" "$VENVS"
export HF_HOME="${HF_HOME:-$MODELS/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-$MODELS/torch_cache}"
mkdir -p "$HF_HOME" "$TORCH_HOME"

if [[ ! -d "$EXT/UMO/.git" ]]; then
  git clone --depth 1 https://github.com/bytedance/UMO.git "$EXT/UMO"
fi
git -C "$EXT/UMO" submodule update --init --depth 1 projects/OmniGen2

if [[ ! -d "$EXT/FreeGraftor/.git" ]]; then
  git clone --depth 1 https://github.com/Nihukat/FreeGraftor.git "$EXT/FreeGraftor"
fi

echo "[setup] creating OmniGen2/UMO environment"
if [[ ! -f "$VENVS/omni/.deps_ready" ]]; then
  "$PYTHON" -m venv "$VENVS/omni"
  "$VENVS/omni/bin/pip" install --upgrade pip
  "$VENVS/omni/bin/pip" install \
    torch==2.6.0 torchvision==0.21.0 \
    --extra-index-url https://download.pytorch.org/whl/cu124
  "$VENVS/omni/bin/pip" install \
    -r "$EXT/UMO/projects/OmniGen2/requirements.txt" \
    peft safetensors huggingface_hub transformers accelerate pillow
  touch "$VENVS/omni/.deps_ready"
fi

echo "[setup] creating FreeGraftor environment"
if [[ ! -f "$VENVS/freegraftor/.deps_ready" ]]; then
  "$PYTHON" -m venv "$VENVS/freegraftor"
  "$VENVS/freegraftor/bin/pip" install --upgrade pip
  "$VENVS/freegraftor/bin/pip" install -r "$EXT/FreeGraftor/requirements.txt"
  "$VENVS/freegraftor/bin/pip" install \
    einops tqdm safetensors huggingface_hub transformers accelerate
  touch "$VENVS/freegraftor/.deps_ready"
fi

echo "[setup] creating isolated Qwen/Unsloth MIE environment"
if [[ ! -f "$VENVS/mie/.deps_ready" ]]; then
  "$PYTHON" -m venv "$VENVS/mie"
  "$VENVS/mie/bin/pip" install --upgrade pip setuptools wheel packaging ninja
  "$VENVS/mie/bin/pip" install \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url "${MIE_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
  "$VENVS/mie/bin/pip" install \
    transformers==5.5.0 peft pillow 'fsspec<=2025.9.0' tqdm
  "$VENVS/mie/bin/pip" install unsloth unsloth_zoo
  touch "$VENVS/mie/.deps_ready"
fi
"$VENVS/mie/bin/python" -c \
  'import unsloth, torch, transformers, peft; print("[setup] MIE runtime imports OK")'

HF_ARGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_ARGS+=(--token "$HF_TOKEN")
fi

echo "[setup] downloading OmniGen2 and UMO weights into $MODELS"
"$VENVS/omni/bin/hf" download OmniGen2/OmniGen2 \
  --local-dir "$MODELS/OmniGen2" "${HF_ARGS[@]}"
"$VENVS/omni/bin/hf" download bytedance-research/UMO \
  UMO_OmniGen2.safetensors --local-dir "$MODELS/UMO" "${HF_ARGS[@]}"

echo "[setup] downloading Grounding-DINO (shared by SCR and FreeGraftor)"
"$VENVS/omni/bin/hf" download IDEA-Research/grounding-dino-tiny \
  --local-dir "$MODELS/grounding-dino-tiny" "${HF_ARGS[@]}"

echo "[setup] downloading FreeGraftor dependencies"
# FLUX.1-dev is gated: accept its HF license and provide HF_TOKEN if required.
"$VENVS/freegraftor/bin/hf" download black-forest-labs/FLUX.1-dev \
  --local-dir "$MODELS/FLUX.1-dev" "${HF_ARGS[@]}"
if [[ ! -f "$MODELS/sam_vit_h_4b8939.pth" ]]; then
  curl -L --fail --retry 3 \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
    -o "$MODELS/sam_vit_h_4b8939.pth"
fi

echo
echo "[setup] all non-MIE Round-1 code and weights are ready."
echo "        Omni Python: $VENVS/omni/bin/python"
echo "        FG Python:   $VENVS/freegraftor/bin/python"
echo "        MIE Python:  $VENVS/mie/bin/python"
echo "        Next: provide only MIE_CKPT, then run round1/run_round1.sh."
