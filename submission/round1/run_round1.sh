#!/usr/bin/env bash
# Round-1: does the claim have signal? (cheap, fast, no human)
#
# Runs all five pipelines directly and scores every final image with the same
# detection-aware SCR(DINOv2) implementation.
#
# Prereqs:
#   - GPU box, `bash setup_round1.sh`
#   - MIE checkpoint: export MIE_CKPT=/path/to/mie-best
#   - Data src + refs from MIBE_Core/.../data_v2
#
# Dependency-free dry run (laptop, no GPU/weights):
#   GEN=mock bash run_round1.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- config ----
RESULTS="${ROUND1_RESULTS:-$PWD/results}"
MANIFESTS="$RESULTS/manifests"
mkdir -p "$RESULTS" "$MANIFESTS"
export ROUND1_WORK="$RESULTS"
export HF_HOME="${HF_HOME:-$PWD/../models/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-$PWD/../models/torch_cache}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$TORCH_HOME"

LOG_DIR="${ROUND1_LOG_DIR:-misc}"
if ! mkdir -p "$LOG_DIR" 2>/dev/null; then
  LOG_DIR="$RESULTS/logs"
  mkdir -p "$LOG_DIR"
  echo "[run] warning: misc unavailable; using $LOG_DIR"
fi
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_LOG="$LOG_DIR/round1_${RUN_ID}.log"
ln -sfn "$RUN_LOG" "$LOG_DIR/round1_latest.log"
ln -sfn "$RUN_LOG" "$RESULTS/round1.log"
exec > >(tee -a "$RUN_LOG") 2>&1

on_exit() {
  status=$?
  echo "[run] finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ) exit_code=$status"
  echo "[run] full_log=$RUN_LOG"
  return "$status"
}
trap on_exit EXIT

echo "[run] run_id=$RUN_ID"
echo "[run] results=$RESULTS"
echo "[run] log=$RUN_LOG"

stage_begin() {
  STAGE_LABEL="$1"
  STAGE_STARTED="$(date +%s)"
  echo
  echo "================================================================================"
  echo "[STAGE START] $STAGE_LABEL"
  echo "================================================================================"
}

stage_end() {
  local finished elapsed
  finished="$(date +%s)"
  elapsed=$((finished - STAGE_STARTED))
  echo "[STAGE END] $STAGE_LABEL elapsed=${elapsed}s status=OK"
}

stage_skip() {
  echo "[STAGE SKIP] $1 reason=$2"
}

DATA_SRC="${DATA_SRC:-../prompt/train_60k_v13_2.jsonl}"
REFS_DIR="${REFS_DIR:-../refs}"
GEN="${GEN:-omnigen2}"
B="${B:-8}"                     # aligned budget: best-of-N=8, ours=n_init4+k4
DATA="${DATA:-$MANIFESTS/hard_cases.jsonl}"
# OmniGen2 supports at most 5 reference images (image_index_embedding size=5),
# so 6/8-entity cases cannot run on OmniGen2-based methods. The hardest tier we
# can run natively is 4 entities; 2 entities is the easy contrast. 6/8-entity
# extreme-collapse stress moves to Round 2 on a base that supports more refs.
HARD_ENTITIES="${HARD_ENTITIES:-4}"
EASY_ENTITIES="${EASY_ENTITIES:-2}"
N_SUPPORTED="${N_SUPPORTED:-30}" # hard tier (4 entities)
N_STRESS="${N_STRESS:-30}"       # easy contrast tier (2 entities)
N_CAL_SUPPORTED="${N_CAL_SUPPORTED:-30}"
N_CAL_STRESS="${N_CAL_STRESS:-30}"
PY_OMNI="${PY_OMNI:-../.venvs/omni/bin/python}"
PY_FG="${PY_FG:-../.venvs/freegraftor/bin/python}"
PY_MIE="${PY_MIE:-../.venvs/mie/bin/python}"
export MIE_PYTHON="$PY_MIE"
DEFAULT_MIE_CKPT="Model_Training_runs/v2/unsloth_Qwen3.5-4B/20260503_045230/outputs/unsloth_Qwen3.5-4B-lora_layer-best"
export MIE_CKPT="${MIE_CKPT:-$DEFAULT_MIE_CKPT}"
CAL_DATA="$MANIFESTS/calibration_cases.jsonl"
CAL_BASELINE="$RESULTS/calibration/mie_baselines.json"

SCR_FLAG=""
if [[ "$GEN" == "mock" ]]; then
  PY_OMNI="${PYTHON:-python3}"
  SCR_FLAG="--no_scr"
  unset MIE_CKPT
else
  stage_begin "01/11 Preflight — verify GPU, disk, data, refs and MIE checkpoint"
  if [[ ! -d "$MIE_CKPT" ]]; then
    echo "ERROR: MIE checkpoint directory not found: $MIE_CKPT" >&2
    exit 2
  fi
  "${PYTHON:-python3}" preflight.py \
    --checkpoint "$MIE_CKPT" \
    --prompt "$DATA_SRC" \
    --refs "$REFS_DIR" \
    --out "$RESULTS/preflight.json"
  stage_end
  if [[ ! -f "../.venvs/omni/.deps_ready" \
        || ! -f "../.venvs/freegraftor/.deps_ready" \
        || ! -f "../.venvs/mie/.deps_ready" ]]; then
    if [[ -z "${HF_TOKEN:-}" ]]; then
      echo "ERROR: dependencies are absent; export a NEW HF_TOKEN first." >&2
      exit 2
    fi
    stage_begin "02/11 Setup — install isolated runtimes and download model weights"
    bash setup_round1.sh
    stage_end
  else
    stage_skip "02/11 Setup" "dependency markers already present"
  fi
fi

# ---- 0. pre-register a mixed hard set: fair 4-entity + n>4 stress ----
if [[ ! -f "$DATA" ]]; then
  stage_begin "03/11 Manifest — select $N_SUPPORTED hard(${HARD_ENTITIES}) + $N_STRESS easy(${EASY_ENTITIES}) cases"
  "$PY_OMNI" select_hard_cases.py \
    --src "$DATA_SRC" --refs "$REFS_DIR" \
    --exact_entities "$HARD_ENTITIES" --n "$N_SUPPORTED" --seed 0 \
    --out "$MANIFESTS/hard_supported.jsonl"
  "$PY_OMNI" select_hard_cases.py \
    --src "$DATA_SRC" --refs "$REFS_DIR" \
    --exact_entities "$EASY_ENTITIES" --n "$N_STRESS" --seed 1 \
    --out "$MANIFESTS/easy_contrast.jsonl"
  "$PY_OMNI" -c 'from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(Path(sys.argv[2]).read_bytes() + Path(sys.argv[3]).read_bytes())' \
    "$DATA" "$MANIFESTS/hard_supported.jsonl" "$MANIFESTS/easy_contrast.jsonl"
  stage_end
else
  stage_skip "03/11 Manifest" "frozen manifest already exists"
fi

# ---- 1. held-out MIE dimension calibration (never overlaps test 60) ----
if [[ "$GEN" != "mock" && ! -f "$CAL_BASELINE" ]]; then
  stage_begin "04/11 Calibration generation — build held-out split and run OmniGen2 one-shot"
  "$PY_OMNI" select_hard_cases.py \
    --src "$DATA_SRC" --refs "$REFS_DIR" --exact_entities "$HARD_ENTITIES" \
    --n "$N_CAL_SUPPORTED" --seed 100 --exclude "$DATA" \
    --out "$MANIFESTS/calibration_hard.jsonl"
  "$PY_OMNI" select_hard_cases.py \
    --src "$DATA_SRC" --refs "$REFS_DIR" --exact_entities "$EASY_ENTITIES" \
    --n "$N_CAL_STRESS" --seed 101 --exclude "$DATA" \
    --out "$MANIFESTS/calibration_easy.jsonl"
  "$PY_OMNI" -c 'from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(Path(sys.argv[2]).read_bytes() + Path(sys.argv[3]).read_bytes())' \
    "$CAL_DATA" "$MANIFESTS/calibration_hard.jsonl" "$MANIFESTS/calibration_easy.jsonl"
  "$PY_OMNI" p2_oneshot.py \
    --name calibration --data "$CAL_DATA" --generator "$GEN" --no_scr
  stage_end
  stage_begin "05/11 MIE calibration — score E/A/I and freeze median/MAD baselines"
  "$PY_OMNI" calibrate_mie.py \
    --data "$CAL_DATA" \
    --images "$RESULTS/calibration/images" \
    --out "$CAL_BASELINE" \
    --raw_out "$RESULTS/calibration/mie_scores.jsonl"
  stage_end
elif [[ "$GEN" != "mock" ]]; then
  stage_skip "04-05/11 Calibration" "frozen MIE baselines already exist"
fi

# ---- 2. in-house pipelines on the SAME base (OmniGen2) ----
stage_begin "06/11 One-shot — OmniGen2 direct generation baseline"
"$PY_OMNI" p2_oneshot.py --data "$DATA" --generator "$GEN" $SCR_FLAG
stage_end
stage_begin "07/11 Best-of-N — generate B candidates and select by MIE preference"
"$PY_OMNI" p3_bestofn.py --data "$DATA" --generator "$GEN" --budget "$B" $SCR_FLAG
stage_end
stage_begin "08/11 OURS — calibrated MIE diagnose, semantic correction and verification"
# Budget matched to best-of-N (B): n_init + k*proposals = 2 + 3*2 = 8.
OURS_PROPOSALS="${OURS_PROPOSALS:-2}" OURS_DEFICIT_MIN="${OURS_DEFICIT_MIN:-0.75}" \
"$PY_OMNI" p1_ours.py \
  --data "$DATA" --generator "$GEN" --n_init "${OURS_N_INIT:-2}" --k "${OURS_K:-3}" \
  --calibration "$CAL_BASELINE" $SCR_FLAG
stage_end

# ---- 3. released external baselines, direct inference ----
if [[ "$GEN" != "mock" ]]; then
  stage_begin "09/11 UMO — released retrained OmniGen2 baseline"
  "$PY_OMNI" p4_umo.py --data "$DATA" || echo "[warn] UMO stage failed; continuing to evaluation"
  stage_end
  # Interim verdict from the OmniGen2 same-base methods (signals 1&2), so a
  # GO/STOP is available before the slow optional FreeGraftor stage.
  echo "[run] interim evaluation (OmniGen2 methods, before FreeGraftor):"
  "$PY_OMNI" compare_round1.py --runs "$RESULTS" --out_dir "$RESULTS" || true
  stage_begin "10/11 FreeGraftor — released training-free open-loop baseline (optional)"
  if [[ -f "../models/sam_vit_h_4b8939.pth" && -d "../models/FLUX.1-dev" ]]; then
    "$PY_FG" p5_freegraftor.py --data "$DATA" || echo "[warn] FreeGraftor stage failed; continuing to evaluation"
  else
    echo "[warn] FreeGraftor prerequisites (SAM/FLUX) missing; skipping optional baseline"
  fi
  stage_end
else
  stage_skip "09-10/11 External baselines" "mock mode"
fi

# ---- 4. compare + three signals ----
stage_begin "11/11 Evaluation — aggregate independent SCR/DINO and decide Round 2"
"$PY_OMNI" compare_round1.py --runs "$RESULTS" --out_dir "$RESULTS"
stage_end

echo "[run] every artifact is under: $RESULTS"
