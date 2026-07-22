#!/usr/bin/env bash
# Calibrate MIE routing baselines for FLUX.2 (per-entity-count median/MAD).
#
# Round-1/2 calibration was frozen on OmniGen2 one-shot images. FLUX.2 has a
# different score distribution, so we need a SEPARATE calibration for the
# FLUX.2 scaling experiment. This script:
#   1. Generates one-shot FLUX.2 images on a held-out calibration split.
#   2. Scores them with MIE.
#   3. Fits per-entity-count median/MAD -> mie_baselines_flux2.json.
#
# The calibration split is disjoint from the round2 test tasks AND from the
# round1 calibration tasks (use a different --seed).
#
# Run on the server (from round2/ dir):
#   MIE_CKPT=<ckpt> bash calibrate_flux2.sh
#
# Output: results_flux2/calibration/mie_baselines_flux2.json
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../round1" && pwd)"

REPO="$(cd .. && pwd)"
DATA_SRC="${DATA_SRC:-$REPO/prompt/train_60k_v13_2.jsonl}"
REFS_DIR="${REFS_DIR:-$REPO/refs}"
MIE_CKPT="${MIE_CKPT:?set MIE_CKPT}"
OUT="${OUT:-$PWD/../round2/results_flux2}"
PY="${PY:-../.venvs/omni/bin/python}"
N_CALIB="${N_CALIB:-60}"   # 30 @ 6-entity + 30 @ 8-entity
SEED="${SEED:-777}"        # disjoint from round1 (200/201) and round2 (200/201)

export ROUND1_WORK="$OUT"
export MIE_PYTHON="${MIE_PYTHON:-/workspace/misc/.venvs/mie/bin/python}"
export MIE_CKPT
export OMNIGEN2_STEPS="${OMNIGEN2_STEPS:-28}" ROUND1_CPU_OFFLOAD=0
export FLUX2_STEPS="${FLUX2_STEPS:-28}"

CAL_DIR="$OUT/calibration"
IMG_DIR="$CAL_DIR/images"
MAN_DIR="$CAL_DIR/manifests"
mkdir -p "$CAL_DIR" "$IMG_DIR" "$MAN_DIR"

# 1) build calibration manifest (6 + 8 entities, occlusion_interaction, disjoint)
echo "[calib-flux2] building calibration manifest (n=$N_CALIB, seed=$SEED) ..."
"$PY" select_hard_cases.py --src "$DATA_SRC" --refs "$REFS_DIR" \
  --exact_entities 6 --n $((N_CALIB / 2)) --seed $SEED \
  --out "$MAN_DIR/calib_6.jsonl"
"$PY" select_hard_cases.py --src "$DATA_SRC" --refs "$REFS_DIR" \
  --exact_entities 8 --n $((N_CALIB / 2)) --seed $((SEED + 1)) \
  --out "$MAN_DIR/calib_8.jsonl"
cat "$MAN_DIR/calib_6.jsonl" "$MAN_DIR/calib_8.jsonl" > "$MAN_DIR/calib_flux2.jsonl"
echo "[calib-flux2] manifest: $(wc -l < "$MAN_DIR/calib_flux2.jsonl") tasks"

# 2) generate one-shot FLUX.2 images for calibration
echo "[calib-flux2] generating one-shot FLUX.2 images ..."
"$PY" p2_oneshot.py --name "calib_flux2_oneshot" \
  --data "$MAN_DIR/calib_flux2.jsonl" --generator flux2 --seed_offset 0

# 3) fit MIE baselines
echo "[calib-flux2] fitting MIE median/MAD baselines ..."
"$PY" calibrate_mie.py \
  --data "$MAN_DIR/calib_flux2.jsonl" \
  --images "$OUT/calib_flux2_oneshot/images" \
  --out "$CAL_DIR/mie_baselines_flux2.json" \
  --raw_out "$CAL_DIR/mie_raw_flux2.jsonl"

echo "[calib-flux2] done -> $CAL_DIR/mie_baselines_flux2.json"
echo "[calib-flux2] use this as CALIBRATION for run_flux2_scaling.sh"
