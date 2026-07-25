#!/usr/bin/env bash
# Round-2 compute-scaling curve: ours vs best-of-N across budgets B on a fixed
# subset. Frames test-time compute as "zero training cost" (not vs Ma et al.).
# For OURS, budget B = n_init + k*proposals; we keep proposals=2 and vary (n_init,k).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../round1" && pwd)"

DATA="${DATA:?set DATA (fixed subset manifest)}"
CALIBRATION="${CALIBRATION:?set CALIBRATION}"
OUT="${OUT:-$PWD/../round2/results_scaling}"
PY="${PY:-../.venvs/omni/bin/python}"
export ROUND1_WORK="$OUT"
export OMNIGEN2_STEPS="${OMNIGEN2_STEPS:-28}" ROUND1_CPU_OFFLOAD=0
export MIE_PYTHON="${MIE_PYTHON:-.venvs/mie/bin/python}"
export MIE_CKPT="${MIE_CKPT:?set MIE_CKPT}"

# budget -> (n_init,k) with proposals=2 : B = n_init + 2k
declare -A NI=( [4]=2 [6]=2 [8]=2 )
declare -A KK=( [4]=1 [6]=2 [8]=3 )
for B in 4 6 8; do
  "$PY" p3_bestofn.py --name "bon_B${B}" --data "$DATA" --generator omnigen2 --budget "$B"
  OURS_PROPOSALS=2 OURS_DEFICIT_MIN=0.75 "$PY" p1_ours.py --name "ours_B${B}" \
    --data "$DATA" --generator omnigen2 --n_init "${NI[$B]}" --k "${KK[$B]}" --calibration "$CALIBRATION"
done
echo "scaling done -> $OUT ; plot SCR vs B for ours_B* and bon_B*"
