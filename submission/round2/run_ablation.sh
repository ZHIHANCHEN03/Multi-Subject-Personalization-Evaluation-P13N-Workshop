#!/usr/bin/env bash
# Round-3 ABLATION of OURS on a fixed hard subset (reuse round1/round2 frozen
# calibration). All variants are training-free + MIE-guided; the point is to
# show each design choice matters. Runs multi-seed for bootstrap CIs.
#
# Variants (all at budget B=8, compute-matched):
#   ours_full          : calibrated routing + prompt+refset + dual-signal + portfolio  (main = ours_v2)
#   ours_rawroute      : raw-argmin routing              (ablate calibration)
#   ours_promptonly    : prompt-only actions             (ablate reference-set lever)
#   ours_nodual        : round-robin subject selection   (ablate SCR dual-signal diagnosis)
#   ours_noportfolio   : single action per step          (ablate action portfolio)
#   ours_strictaccept  : strict v1 acceptance            (ablate relaxed dual-signal acceptance)
#
# Multi-seed: set SEEDS="0,1,2" (default "0,1"). Output dirs namespaced by seed.
#
# Parallel: set SHARD to run a subset of variants on this GPU.
#   SHARD=0 -> ours_full, ours_rawroute
#   SHARD=1 -> ours_promptonly, ours_nodual
#   SHARD=2 -> ours_noportfolio, ours_strictaccept
#   SHARD unset -> run all 6 sequentially.
#
# Required env:
#   DATA          manifest jsonl (e.g. round2 hard subset, ~150 tasks)
#   CALIBRATION   frozen mie_baselines.json
#   MIE_CKPT      MIE checkpoint dir
# Optional:
#   OUT (default round2/results_ablation), SEEDS, SHARD, N_INIT, K
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../round1" && pwd)"

DATA="${DATA:?set DATA (manifest jsonl)}"
CALIBRATION="${CALIBRATION:?set CALIBRATION}"
MIE_CKPT="${MIE_CKPT:?set MIE_CKPT}"
OUT="${OUT:-$PWD/../round2/results_ablation}"
SEEDS="${SEEDS:-0,1}"
PY="${PY:-../.venvs/omni/bin/python}"
N_INIT="${N_INIT:-2}"
K="${K:-2}"

export ROUND1_WORK="$OUT"
export OMNIGEN2_STEPS="${OMNIGEN2_STEPS:-28}" ROUND1_CPU_OFFLOAD=0
export MIE_PYTHON="${MIE_PYTHON:-.venvs/mie/bin/python}"
export MIE_CKPT
COMMON=(--data "$DATA" --generator omnigen2 --n_init "$N_INIT" --k "$K" --calibration "$CALIBRATION")

# Shared winner env (v2.3 weaksel). Each variant overrides ONE knob.
base_env() {
  export V2_DUAL_SIGNAL=1 V2_ACTION_PORTFOLIO=1 V2_DUAL_ACCEPT=1 V2_ACCEPT_MODE=relaxed
  export V2_SELECT_MODE=weak_subject V2_TOTAL_TOL=0.0
  export V2_ACTIONS_PER_STEP=3 V2_SEEDS_PER_ACTION=1
  export OURS_DEFICIT_MIN=0.5 OURS_REFSET_MODE=front_dup3 OURS_LAYOUT=1
  export OURS_PROPOSALS=2
}

run_variant() {
  local name="$1"; shift
  local seed="$1"; shift
  echo "[ablation] === $name seed=$seed ==="
  base_env
  "$@" "$PY" p1_ours_v2.py --name "${name}_s${seed}" "${COMMON[@]}" --seed_offset "$seed"
}

# Variant definitions: name -> extra env + extra args
run_ours_full()        { base_env; "$PY" p1_ours_v2.py --name "ours_full_s${1}"        "${COMMON[@]}" --routing calibrated --action both        --seed_offset "$1"; }
run_ours_rawroute()    { base_env; "$PY" p1_ours_v2.py --name "ours_rawroute_s${1}"    "${COMMON[@]}" --routing raw        --action both        --seed_offset "$1"; }
run_ours_promptonly()  { base_env; "$PY" p1_ours_v2.py --name "ours_promptonly_s${1}"  "${COMMON[@]}" --routing calibrated --action prompt_only --seed_offset "$1"; }
run_ours_nodual()      { base_env; V2_DUAL_SIGNAL=0;            "$PY" p1_ours_v2.py --name "ours_nodual_s${1}"      "${COMMON[@]}" --routing calibrated --action both --seed_offset "$1"; }
run_ours_noportfolio() { base_env; V2_ACTION_PORTFOLIO=0; V2_ACTIONS_PER_STEP=1; "$PY" p1_ours_v2.py --name "ours_noportfolio_s${1}" "${COMMON[@]}" --routing calibrated --action both --seed_offset "$1"; }
run_ours_strictaccept(){ base_env; V2_ACCEPT_MODE=strict; V2_TOTAL_TOL=0.0; "$PY" p1_ours_v2.py --name "ours_strictaccept_s${1}" "${COMMON[@]}" --routing calibrated --action both --seed_offset "$1"; }

VARIANTS=(ours_full ours_rawroute ours_promptonly ours_nodual ours_noportfolio ours_strictaccept)

# Shard: split 6 variants across GPUs (2 per shard for 3 GPUs; SHARD unset = all)
if [ -n "${SHARD:-}" ]; then
  case "$SHARD" in
    0) SHARD_VARIANTS=(ours_full ours_rawroute);;
    1) SHARD_VARIANTS=(ours_promptonly ours_nodual);;
    2) SHARD_VARIANTS=(ours_noportfolio ours_strictaccept);;
    *) echo "SHARD must be 0,1,2"; exit 2;;
  esac
else
  SHARD_VARIANTS=("${VARIANTS[@]}")
fi

IFS=',' read -ra SEED_LIST <<< "$SEEDS"
for v in "${SHARD_VARIANTS[@]}"; do
  for s in "${SEED_LIST[@]}"; do
    s="${s// /}"
    "run_${v}" "$s"
  done
done

echo "[ablation] done -> $OUT (variants: ${SHARD_VARIANTS[*]}; seeds: $SEEDS)"
echo "[ablation] analyze with:"
echo "  $PY ../round2/analyze.py --results $OUT --main ours_full --others ours_rawroute ours_promptonly ours_nodual ours_noportfolio ours_strictaccept --entities 4 --metric scr --seeds ${SEEDS//,/ }"
