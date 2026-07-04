#!/usr/bin/env bash
# Full experiment matrix (idea.md section 8). Compute-aligned by budget B.
#
# Configure via env before running:
#   FLUX2_MODEL_ID   e.g. black-forest-labs/FLUX.2-klein-4B  (main) / FLUX.2-dev (ceiling)
#   MISC_CRITIC      mie_checkpoint   MIE_ADAPTER=<your.module>
#   MISC_DATA        /path/to/mib_gold.jsonl
#   LLM_API_BASE / LLM_API_KEY   (for prompt_llm action + GPT-4o judge)
#
# For a dependency-free dry run of the whole matrix, set:
#   export GEN=mock CRITIC=mock LIMIT="--limit 8"
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GEN="${GEN:-flux2}"
CRITIC="${CRITIC:-${MISC_CRITIC:-mie_checkpoint}}"
LIMIT="${LIMIT:-}"                    # e.g. "--limit 500" for main test set
B="${B:-8}"                            # aligned generation budget
NO_METRICS="${NO_METRICS:-}"          # set to 1 for a dependency-free dry run

EXTRA=""
[[ -n "$NO_METRICS" ]] && EXTRA="--no_metrics"

run() { echo ">>> $*"; python run.py $LIMIT $EXTRA "$@"; }

# ---- 1. main comparison (compute-aligned B) ----
run --name one_shot        --method one_shot        --generator "$GEN" --critic "$CRITIC"
run --name bon_scalar      --method best_of_n       --generator "$GEN" --critic "$CRITIC" --budget "$B"
run --name caption_upsample --method caption_upsample --generator "$GEN" --critic "$CRITIC" --budget "$B"
run --name misc            --method misc            --generator "$GEN" --critic "$CRITIC" --budget "$B"

# ---- 2. routing ablation (calibrated is default in `misc`; show it beats the rest) ----
run --name misc_route_diagnostic --method misc --routing diagnostic --generator "$GEN" --critic "$CRITIC" --budget "$B"  # raw argmin -> degenerates to always-Interaction
run --name misc_route_random --method misc --routing random --generator "$GEN" --critic "$CRITIC" --budget "$B"
run --name misc_route_static --method misc --routing static --generator "$GEN" --critic "$CRITIC" --budget "$B"

# ---- 3. action ablation ----
run --name misc_act_ref  --method misc --action reference  --generator "$GEN" --critic "$CRITIC" --budget "$B"
run --name misc_act_llm  --method misc --action prompt_llm --generator "$GEN" --critic "$CRITIC" --budget "$B"

# ---- 4. seed mode ----
run --name misc_resampled --method misc --seed_mode resampled --generator "$GEN" --critic "$CRITIC" --budget "$B"

# ---- 4b. signal-usage ablations: how much of MIE's 4 numbers we actually use ----
MISC_GRADED=0 run --name misc_fixed_intensity --method misc --generator "$GEN" --critic "$CRITIC" --budget "$B"
MISC_KNOB_ESCALATION=0 run --name misc_no_knob --method misc --generator "$GEN" --critic "$CRITIC" --budget "$B"

# ---- 5. scaling curve: structured (misc) vs scalar (bon) across budgets ----
for b in 2 4 6 8; do
  run --name "scale_misc_B${b}" --method misc      --generator "$GEN" --critic "$CRITIC" --budget "$b"
  run --name "scale_bon_B${b}"  --method best_of_n --generator "$GEN" --critic "$CRITIC" --budget "$b"
done

# ---- 6. generality: swap verifier to vlm_judge (needs LLM API) ----
if [[ -n "${LLM_API_BASE:-}" ]]; then
  run --name misc_vlmjudge --method misc --critic vlm_judge --generator "$GEN" --budget "$B"
fi

echo "==> all runs done. Aggregate the load-bearing comparison:"
echo "    python aggregate.py winrate runs/misc runs/bon_scalar --metric final_total"
echo "    python aggregate.py winrate runs/misc runs/one_shot   --metric clip_i"
