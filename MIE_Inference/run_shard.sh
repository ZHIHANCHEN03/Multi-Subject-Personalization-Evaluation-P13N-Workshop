#!/bin/bash
# Launch one scoring shard.  Usage: ./run_shard.sh <num_shards> <shard_index>
#
# Thread caps matter more than they look. torch/unsloth default to one thread per
# core, so six processes on this 128-core box each spawned ~128 threads: load
# average hit 178, and model load never finished in 11 minutes (a single process
# loads in ~2). Capping at 16 keeps 6 x 16 = 96 threads under the core count.
set -euo pipefail

NUM_SHARDS="${1:?usage: run_shard.sh <num_shards> <shard_index>}"
SHARD="${2:?usage: run_shard.sh <num_shards> <shard_index>}"

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=false

export HF_HOME=/workspace/misc/models/hf_cache
export MIE_CODE="/workspace/misc/MIBE_Core/Multi-Subject-Personalization-Evaluation-P13N-Workshop-feat-neurips-lens/Model_Training_Paper_Coding"
export MIE_CKPT=/workspace/Model_Training_runs/v2/unsloth_Qwen3.5-4B/20260503_045230/outputs/unsloth_Qwen3.5-4B-lora_layer-best

cd /workspace/MIE_Inference
exec /workspace/misc/.venvs/mie/bin/python score_nips_rebuttal.py \
  --root /workspace/nips_rebuttal/nips_rebuttal_ref_extension_exp \
  --out  /workspace/mie_results/mie_scores.json \
  --num_shards "$NUM_SHARDS" --shard "$SHARD"
