#!/usr/bin/env bash
# Round-3 P1-3: FLUX.2 scaling experiment (6/8 subjects).
#
# Story: "the more subjects, the worse the collapse, the larger our gain."
# On FLUX.2 there is NO same-base retrained SOTA, so we only compare:
#   ours_v2  vs  one_shot  vs  best_of_n
# (no UMO here — UMO is OmniGen2-only.)
#
# Prerequisites:
#   1. probe_flux2.sh  -> confirm FLUX.2 supports >=6 refs (check probe.json)
#   2. calibrate_flux2.sh -> produces mie_baselines_flux2.json
#
# Multi-seed + multi-GPU parallel:
#   SHARD=0 -> 6-entity, seeds 0,1,2
#   SHARD=1 -> 8-entity, seeds 0,1,2
#   SHARD unset -> run both sequentially.
# Or split by seed:
#   SHARD=0 -> seeds 0 (both 6+8)
#   SHARD=1 -> seeds 1
#   SHARD=2 -> seeds 2
#
# Required env:
#   MIE_CKPT       MIE checkpoint dir
#   CALIBRATION    mie_baselines_flux2.json (from calibrate_flux2.sh)
# Optional:
#   OUT, SEEDS, SHARD, N6, N8, B
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../round1" && pwd)"

REPO="$(cd .. && pwd)"
DATA_SRC="${DATA_SRC:-$REPO/prompt/train_60k_v13_2.jsonl}"
REFS_DIR="${REFS_DIR:-$REPO/refs}"
MIE_CKPT="${MIE_CKPT:?set MIE_CKPT}"
CALIBRATION="${CALIBRATION:?set CALIBRATION (mie_baselines_flux2.json from calibrate_flux2.sh)}"
OUT="${OUT:-$PWD/../round2/results_flux2}"
SEEDS="${SEEDS:-0,1,2}"
N6="${N6:-100}"
N8="${N8:-100}"
B="${B:-8}"
PY="${PY:-../.venvs/omni/bin/python}"

export ROUND1_WORK="$OUT"
export OMNIGEN2_STEPS="${OMNIGEN2_STEPS:-28}" ROUND1_CPU_OFFLOAD=0
export MIE_PYTHON="${MIE_PYTHON:-/workspace/misc/.venvs/mie/bin/python}"
export MIE_CKPT
export FLUX2_STEPS="${FLUX2_STEPS:-28}" FLUX2_GUIDANCE="${FLUX2_GUIDANCE:-4.0}"

MAN="$OUT/manifests"
mkdir -p "$MAN"

# 1) build 6/8 manifests (disjoint from all prior splits via seed 300/301)
EXCL=""
for ex in "$REPO/round1/results/manifests/calibration_cases.jsonl" \
          "$REPO/round2/results_r2/manifests/round2_full.jsonl" \
          "$OUT/calibration/manifests/calib_flux2.jsonl"; do
  [ -f "$ex" ] && EXCL="$EXCL --exclude $ex"
done
if [ ! -f "$MAN/scaling_6_8.jsonl" ]; then
  "$PY" ../round2/select_hard_cases_6_8.py --src "$DATA_SRC" --refs "$REFS_DIR" \
    --n6 "$N6" --n8 "$N8" --out_dir "$MAN" $EXCL
fi
echo "[flux2-scaling] manifest: $(wc -l < "$MAN/scaling_6_8.jsonl") tasks"

# 2) split into 6-entity and 8-entity manifests (if not already)
[ -f "$MAN/scaling_6.jsonl" ] || head -"$N6" "$MAN/scaling_6_8.jsonl" > "$MAN/scaling_6.jsonl"
[ -f "$MAN/scaling_8.jsonl" ] || tail -"$N8" "$MAN/scaling_6_8.jsonl" > "$MAN/scaling_8.jsonl"

# OURS env (same v2.3 weaksel winner; uses FLUX.2 calibration)
export V2_DUAL_SIGNAL=1 V2_ACTION_PORTFOLIO=1 V2_DUAL_ACCEPT=1 V2_ACCEPT_MODE=relaxed
export V2_SELECT_MODE=weak_subject V2_TOTAL_TOL=0.0
export V2_ACTIONS_PER_STEP=3 V2_SEEDS_PER_ACTION=1
export OURS_DEFICIT_MIN=0.5 OURS_REFSET_MODE=front_dup3 OURS_LAYOUT=1 OURS_PROPOSALS=2

run_slice() {
  local manifest="$1"; local label="$2"; local seed="$3"
  echo "[flux2-scaling] === $label seed=$seed ==="
  "$PY" p2_oneshot.py  --name "flux2_${label}_oneshot_s${seed}"  --data "$manifest" --generator flux2 --seed_offset "$seed"
  "$PY" p3_bestofn.py  --name "flux2_${label}_bon_s${seed}"      --data "$manifest" --generator flux2 --budget "$B" --seed_offset "$seed"
  "$PY" p1_ours_v2.py  --name "flux2_${label}_ours_s${seed}"     --data "$manifest" --generator flux2 \
    --n_init 2 --k 2 --calibration "$CALIBRATION" --seed_offset "$seed"
}

IFS=',' read -ra SEED_LIST <<< "$SEEDS"

# SHARD modes:
#   0 -> 6-entity (all seeds)
#   1 -> 8-entity (all seeds)
#   unset -> both sequentially
case "${SHARD:-}" in
  0) for s in "${SEED_LIST[@]}"; do run_slice "$MAN/scaling_6.jsonl" "6" "$s"; done;;
  1) for s in "${SEED_LIST[@]}"; do run_slice "$MAN/scaling_8.jsonl" "8" "$s"; done;;
  "")
    for s in "${SEED_LIST[@]}"; do run_slice "$MAN/scaling_6.jsonl" "6" "$s"; done
    for s in "${SEED_LIST[@]}"; do run_slice "$MAN/scaling_8.jsonl" "8" "$s"; done
    ;;
  *) echo "SHARD must be 0,1, or unset"; exit 2;;
esac

echo "[flux2-scaling] done -> $OUT"
echo "[flux2-scaling] analyze with:"
echo "  $PY ../round2/merge_shards.py --shard_glob '$OUT' --out '$OUT/merged' --seeds ${SEEDS//,/ }"
echo "  $PY ../round2/analyze.py --results '$OUT/merged' --main flux2_6_ours --others flux2_6_oneshot flux2_6_bon --entities 6 --metric scr --seeds ${SEEDS//,/ }"
echo "  $PY ../round2/analyze.py --results '$OUT/merged' --main flux2_8_ours --others flux2_8_oneshot flux2_8_bon --entities 8 --metric scr --seeds ${SEEDS//,/ }"
