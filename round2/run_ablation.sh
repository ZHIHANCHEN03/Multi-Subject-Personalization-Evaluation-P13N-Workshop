#!/usr/bin/env bash
# Round-2 ablation of OURS on a fixed subset (reuse round1 frozen calibration).
# Variants (all training-free + MIE): shows each design choice matters.
#   ours_full          : calibrated routing + prompt+refset  (main)
#   ours_rawroute      : raw-argmin routing  (ablate calibration)
#   ours_promptonly    : prompt-only actions (ablate reference-set lever)
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../round1" && pwd)"

DATA="${DATA:?set DATA (manifest jsonl, e.g. round2 hard subset)}"
CALIBRATION="${CALIBRATION:?set CALIBRATION}"
OUT="${OUT:-$PWD/../round2/results_ablation}"
PY="${PY:-../.venvs/omni/bin/python}"
export ROUND1_WORK="$OUT"
export OMNIGEN2_STEPS="${OMNIGEN2_STEPS:-28}" ROUND1_CPU_OFFLOAD=0
export MIE_PYTHON="${MIE_PYTHON:-/workspace/misc/.venvs/mie/bin/python}"
export MIE_CKPT="${MIE_CKPT:?set MIE_CKPT}"
COMMON=(--data "$DATA" --generator omnigen2 --n_init 2 --k 3 --calibration "$CALIBRATION")

OURS_PROPOSALS=2 OURS_DEFICIT_MIN=0.75 "$PY" p1_ours.py --name ours_full       "${COMMON[@]}" --routing calibrated --action both
OURS_PROPOSALS=2 OURS_DEFICIT_MIN=0.75 "$PY" p1_ours.py --name ours_rawroute   "${COMMON[@]}" --routing raw        --action both
OURS_PROPOSALS=2 OURS_DEFICIT_MIN=0.75 "$PY" p1_ours.py --name ours_promptonly "${COMMON[@]}" --routing calibrated --action prompt_only

echo "ablation done -> $OUT ; analyze with round2/analyze.py --main ours_full --others ours_rawroute ours_promptonly"
