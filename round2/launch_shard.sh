#!/usr/bin/env bash
# Round-2 per-server shard launcher.
# Run ONE shard on ONE GPU. Launch this on each server (in tmux).
#
# Usage:
#   bash launch_shard.sh <SHARD_INDEX>
#     SHARD_INDEX in {0,1,2,3}  (shard 0 already running on the first server)
#
# Prereqs on the new server:
#   - /workspace mounted to the SAME network volume as the first server
#     (so models/ at /workspace/misc/models and venvs at /workspace/misc/.venvs
#      are already present — no re-download needed)
#   - tmux installed
set -euo pipefail

SHARD="${1:?usage: launch_shard.sh <0|1|2|3> [SEEDS]}"
SEEDS="${2:-0}"   # default seed 0; pass "1,2" for Pass 2 of staged rollout

REPO="/workspace/misc"
R1="$REPO/round1"
R2="$REPO/round2"
MANIFEST="$R2/results_r2/shards/shard_${SHARD}.jsonl"
RESULTS="$R2/results_r2/shard_${SHARD}"
CALIBRATION="$R1/results/calibration/mie_baselines.json"
MIE_CKPT="/workspace/Model_Training_runs/v2/unsloth_Qwen3.5-4B/20260503_045230/outputs/unsloth_Qwen3.5-4B-lora_layer-best"
MIE_PYTHON="$REPO/.venvs/mie/bin/python"

echo "[launch] shard=$SHARD manifest=$MANIFEST results=$RESULTS"
[ -f "$MANIFEST" ]   || { echo "ERROR: manifest not found: $MANIFEST"; exit 2; }
[ -f "$CALIBRATION" ] || { echo "ERROR: calibration not found: $CALIBRATION"; exit 2; }
[ -d "$MIE_CKPT" ]   || { echo "ERROR: MIE checkpoint not found: $MIE_CKPT"; exit 2; }
[ -x "$MIE_PYTHON" ] || { echo "ERROR: MIE python not found: $MIE_PYTHON (run round1/setup_round1.sh)"; exit 2; }

tmux kill-session -t "r2_shard${SHARD}" 2>/dev/null || true
tmux new-session -d -s "r2_shard${SHARD}" "
  cd $R1 && \
  CUDA_VISIBLE_DEVICES=0 \
  SHARD_MANIFEST=$MANIFEST \
  RESULTS_DIR=$RESULTS \
  CALIBRATION=$CALIBRATION \
  MIE_CKPT=$MIE_CKPT \
  MIE_PYTHON=$MIE_PYTHON \
  SEEDS=$SEEDS \
  bash $R2/run_shard.sh > $RESULTS.log 2>&1
"
echo "[launch] shard $SHARD launched in tmux session 'r2_shard${SHARD}' (seeds=$SEEDS)."
echo "[launch] monitor: tmux attach -t r2_shard${SHARD}  |  tail -f $RESULTS.log"
