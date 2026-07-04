"""Global configuration for the MISC pipeline.

All hyperparameters mirror idea.md section 7. Everything that depends on
assets you have not provided yet (MIE checkpoint, LLM API, data) is read from
environment variables and falls back to mock/no-op so the whole pipeline runs
end-to-end on CPU without them.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
PAPER_DIR = CODE_DIR.parent
WORK_DIR = Path(os.environ.get("MISC_WORK_DIR", str(PAPER_DIR / "runs")))
MODELS_DIR = Path(os.environ.get("MISC_MODELS_DIR", str(PAPER_DIR / "models")))
WORK_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dimensions (Existence / Appearance / Interaction) -- MIE typed signal
# ---------------------------------------------------------------------------
DIMS = ["existence", "appearance", "interaction"]
# Static prior used only for tie-breaking when routing==static or on ties.
PRIORITY = ["existence", "appearance", "interaction"]

# ---------------------------------------------------------------------------
# MISC hyperparameters (idea.md section 7). Defaults are frozen; only the
# scaling experiment sweeps N / K.
# ---------------------------------------------------------------------------
N_INIT = int(os.environ.get("MISC_N", 4))            # Best-of-N init candidates
K_STEPS = int(os.environ.get("MISC_K", 3))           # max correction steps
TAU_STOP = float(os.environ.get("MISC_TAU", 0.80))   # stop when total score >= tau
EPS_ACCEPT = float(os.environ.get("MISC_EPS", 0.0))  # require strict improvement (> s_t + eps)
DELTA_FLOOR = float(os.environ.get("MISC_DELTA", 0.0))  # per-dim allowed drop tolerance
THETA_GATE = float(os.environ.get("MISC_THETA", 0.70))  # per-dim output pass gate

# Generator knobs
GUIDANCE_SCALE = float(os.environ.get("MISC_GUIDANCE", 4.0))
STEPS_KLEIN = int(os.environ.get("MISC_STEPS_KLEIN", 4))
STEPS_DEV = int(os.environ.get("MISC_STEPS_DEV", 50))
RESOLUTION = int(os.environ.get("MISC_RES", 1024))
CAPTION_UPSAMPLE_TEMPERATURE = float(os.environ.get("MISC_UPSAMPLE_T", 0.7))

# Seed handling
SEED_MODES = ["fixed", "resampled"]
SEED_MODE_DEFAULT = os.environ.get("MISC_SEED_MODE", "fixed")
BASE_SEED = int(os.environ.get("MISC_BASE_SEED", 0))

# ---------------------------------------------------------------------------
# FLUX.2 model selection
#   dev  = black-forest-labs/FLUX.2-dev   (32B, quality ceiling, H100)
#   klein= black-forest-labs/FLUX.2-klein-4B (fast workhorse, A100/consumer)
# ---------------------------------------------------------------------------
FLUX2_MODEL_ID = os.environ.get("FLUX2_MODEL_ID", "black-forest-labs/FLUX.2-dev")
FLUX2_IS_KLEIN = "klein" in FLUX2_MODEL_ID.lower()
FLUX2_QUANTIZE = os.environ.get("FLUX2_QUANTIZE", "").lower()  # "", "nf4", "int8"
TORCH_DTYPE = os.environ.get("MISC_DTYPE", "bfloat16")

# ---------------------------------------------------------------------------
# MIE critic backend
#   mie_checkpoint : your Paper-1 MIE, loaded via an adapter module (see critic.py)
#   vlm_judge      : an LLM/VLM API used as judge (fallback / generality study)
#   mock           : deterministic fake, for CPU smoke tests
# MIE_ADAPTER : dotted path to a python module exposing
#               score(image, prompt, subject_refs, subject_names) -> dict
# ---------------------------------------------------------------------------
CRITIC_BACKEND = os.environ.get("MISC_CRITIC", "mock")
MIE_ADAPTER = os.environ.get("MIE_ADAPTER", "")       # e.g. "my_mie.adapter"
MIE_CKPT_DIR = Path(os.environ.get("MIE_CKPT_DIR", str(MODELS_DIR / "mie_ckpt")))

# ---------------------------------------------------------------------------
# LLM API (for vlm_judge critic and the LLM-rewriter action variant).
# Not provided by you yet -> if unset, those paths raise a clear error and the
# rule-based / mock fallbacks are used instead.
# ---------------------------------------------------------------------------
LLM_API_BASE = os.environ.get("LLM_API_BASE", "")     # e.g. https://api.openai.com/v1
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o")  # independent judge (§8.5)

# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------
ACTION_MODES = ["prompt_rule", "prompt_llm", "reference"]  # idea.md §8.2
ACTION_MODE_DEFAULT = os.environ.get("MISC_ACTION", "prompt_rule")

INTERACTION_PHRASES = [
    "standing right next to each other, shoulder to shoulder",
    "directly facing each other, clearly interacting",
    "positioned together in the same scene, close to one another",
]

# Routing mode
ROUTING_MODES = ["diagnostic", "random", "static"]
ROUTING_DEFAULT = os.environ.get("MISC_ROUTING", "diagnostic")


def steps_for_model() -> int:
    return STEPS_KLEIN if FLUX2_IS_KLEIN else STEPS_DEV
