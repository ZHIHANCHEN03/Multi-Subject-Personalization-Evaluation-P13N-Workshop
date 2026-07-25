#!/usr/bin/env bash
# Run the 4 OmniGen2 same-base methods on ONE manifest shard, on ONE GPU,
# for one or more seeds. Reuses round1 pipelines + the FROZEN round1
# calibration baselines. Launch one of these per GPU (set CUDA_VISIBLE_DEVICES
# + SHARD). Output dirs are namespaced by seed so multiple seeds never collide.
#
# Required env:
#   SHARD_MANIFEST   path to shard_*.jsonl
#   RESULTS_DIR      where this shard writes (e.g. round2/results_r2/shard_0)
#   MIE_CKPT         MIE checkpoint dir
#   CALIBRATION      frozen round1 baselines json (reused)
# Optional:
#   CUDA_VISIBLE_DEVICES, B (default 8), SEEDS (default "0", comma-separated)
#
# Staged rollout for AAAI P0 (>=3 seeds + 95%CI):
#   Pass 1: SEEDS=0           -> ~1.4 days for 4 shards in parallel; preliminary signal.
#   Pass 2: SEEDS=1,2          -> another ~2.8 days; full 3-seed bootstrap CIs.
# Re-running a seed is safe: it overwrites that seed's dirs only.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../round1" && pwd)"   # run from round1/

SHARD_MANIFEST="${SHARD_MANIFEST:?set SHARD_MANIFEST}"
RESULTS_DIR="${RESULTS_DIR:?set RESULTS_DIR}"
CALIBRATION="${CALIBRATION:?set CALIBRATION (frozen round1 mie_baselines.json)}"
B="${B:-8}"
SEEDS="${SEEDS:-0}"
PY="${PY:-../.venvs/omni/bin/python}"
export ROUND1_WORK="$RESULTS_DIR"
export OMNIGEN2_STEPS="${OMNIGEN2_STEPS:-28}" ROUND1_CPU_OFFLOAD="${ROUND1_CPU_OFFLOAD:-0}"
export MIE_PYTHON="${MIE_PYTHON:-.venvs/mie/bin/python}"
export MIE_CKPT="${MIE_CKPT:?set MIE_CKPT}"

echo "[shard] manifest=$SHARD_MANIFEST results=$RESULTS_DIR device=${CUDA_VISIBLE_DEVICES:-?} seeds=$SEEDS"

# OURS env (Round-1.1 winner: v2.3 weaksel dual-signal diagnose-and-target).
# Budget 8 = n_init 2 + k 2 + actions_per_step 3 (compute-matched to best-of-N=8).
export V2_DUAL_SIGNAL=1 V2_ACTION_PORTFOLIO=1 V2_DUAL_ACCEPT=1 V2_ACCEPT_MODE=relaxed
export V2_SELECT_MODE=weak_subject V2_TOTAL_TOL=0.0
export V2_ACTIONS_PER_STEP=3 V2_SEEDS_PER_ACTION=1
export OURS_DEFICIT_MIN=0.5 OURS_REFSET_MODE=front_dup3 OURS_LAYOUT=1

IFS=',' read -ra SEED_LIST <<< "$SEEDS"
for s in "${SEED_LIST[@]}"; do
  s="${s// /}"  # trim whitespace
  echo "[shard] === seed $s ==="
  "$PY" p2_oneshot.py  --name "one_shot_s${s}"  --data "$SHARD_MANIFEST" --generator omnigen2 --seed_offset "$s"
  "$PY" p3_bestofn.py  --name "best_of_n_s${s}" --data "$SHARD_MANIFEST" --generator omnigen2 --budget "$B" --seed_offset "$s"
  "$PY" p1_ours_v2.py  --name "ours_v2_s${s}"   --data "$SHARD_MANIFEST" --generator omnigen2 \
    --n_init "${OURS_N_INIT:-2}" --k "${OURS_K:-2}" --calibration "$CALIBRATION" \
    --seed_offset "$s"
  "$PY" p4_umo.py      --name "umo_s${s}"       --data "$SHARD_MANIFEST" --seed_offset "$s"
done

echo "[shard] done -> $RESULTS_DIR (seeds: $SEEDS)"
