#!/usr/bin/env bash
# Round-2 MAIN run orchestrator (500 tasks, sharded across GPUs).
# Builds the full manifest, REUSES round1 frozen calibration, splits into shards,
# and launches one tmux session per GPU. Merge + analyze after all finish.
#
# Run on the server (from repo round2/ dir). Set NGPU to the number of A100s.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO="$(cd .. && pwd)"
R1="$REPO/round1"
DATA_SRC="${DATA_SRC:-$REPO/prompt/train_60k_v13_2.jsonl}"
REFS_DIR="${REFS_DIR:-$REPO/refs}"
CALIBRATION="${CALIBRATION:-$R1/results/calibration/mie_baselines.json}"
MIE_CKPT="${MIE_CKPT:-/workspace/Model_Training_runs/v2/unsloth_Qwen3.5-4B/20260503_045230/outputs/unsloth_Qwen3.5-4B-lora_layer-best}"
NGPU="${NGPU:-4}"
N_HARD="${N_HARD:-250}"   # 4-entity
N_EASY="${N_EASY:-250}"   # 2-entity
SEEDS="${SEEDS:-0}"       # comma-separated; AAAI P0 wants "0,1,2" (staged: 0 first, then 1,2)
OUT="${OUT:-$PWD/results_r2}"
MAN="$OUT/manifests"
PY="$R1/../.venvs/omni/bin/python"

mkdir -p "$MAN" "$OUT/shards"
[ -f "$CALIBRATION" ] || { echo "ERROR: frozen calibration missing: $CALIBRATION (run round1 first)"; exit 2; }

# 1) build full manifest (exclude the round1 calibration tasks to avoid leakage)
if [ ! -f "$MAN/round2_full.jsonl" ]; then
  EXCL=""
  [ -f "$MAN/../../../round1/results/manifests/calibration_cases.jsonl" ] && \
    EXCL="--exclude $R1/results/manifests/calibration_cases.jsonl"
  "$PY" "$R1/select_hard_cases.py" --src "$DATA_SRC" --refs "$REFS_DIR" \
    --exact_entities 4 --n "$N_HARD" --seed 200 $EXCL --out "$MAN/r2_hard.jsonl"
  "$PY" "$R1/select_hard_cases.py" --src "$DATA_SRC" --refs "$REFS_DIR" \
    --exact_entities 2 --n "$N_EASY" --seed 201 $EXCL --out "$MAN/r2_easy.jsonl"
  cat "$MAN/r2_hard.jsonl" "$MAN/r2_easy.jsonl" > "$MAN/round2_full.jsonl"
fi
echo "[r2] full manifest: $(wc -l < "$MAN/round2_full.jsonl") tasks"

# 2) split into NGPU shards
"$PY" split_manifest.py --data "$MAN/round2_full.jsonl" --shards "$NGPU" --out_dir "$OUT/shards"

# 3) launch one tmux per GPU
for i in $(seq 0 $((NGPU-1))); do
  sess="r2_shard_$i"
  tmux kill-session -t "$sess" 2>/dev/null || true
  tmux new-session -d -s "$sess" \
    "CUDA_VISIBLE_DEVICES=$i SHARD_MANIFEST=$OUT/shards/shard_$i.jsonl \
     RESULTS_DIR=$OUT/shard_$i CALIBRATION=$CALIBRATION MIE_CKPT=$MIE_CKPT \
     SEEDS=$SEEDS \
     bash $PWD/run_shard.sh > $OUT/shard_$i.log 2>&1; echo SHARD_${i}_DONE=\$? >> $OUT/shard_$i.log"
  echo "[r2] launched $sess on GPU $i -> $OUT/shard_$i.log"
done
echo
echo "[r2] all shards launched (seeds=$SEEDS). When all *_DONE appear, run:"
echo "  $PY merge_shards.py --shard_glob '$OUT/shard_*' --out $OUT/merged --seeds ${SEEDS//,/ }"
echo "  $PY analyze.py --results $OUT/merged --main ours_v2 --others umo best_of_n one_shot --seeds ${SEEDS//,/ }"
