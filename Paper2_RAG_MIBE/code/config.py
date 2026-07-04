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
# Accept requires the MIE preference score (unbounded margin-ranking scalar) to
# strictly improve; only comparisons are meaningful, never an absolute threshold.
EPS_ACCEPT = float(os.environ.get("MISC_EPS", 0.0))  # require strict improvement (> s_t + eps)
DELTA_FLOOR = float(os.environ.get("MISC_DELTA", 0.0))  # per-dim allowed drop tolerance (on [0,1] E/A/I)
# Stop / output gate on the [0,1] category scores (E/A/I). This is the valid
# absolute threshold (preference score is unbounded, cannot be thresholded).
THETA_GATE = float(os.environ.get("MISC_THETA", 0.70))

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
#   klein= black-forest-labs/FLUX.2-klein-4B (DEFAULT: fast workhorse, Apache-2.0,
#          ungated -- no HF token / license click needed, sub-second, ~8GB)
#   dev  = black-forest-labs/FLUX.2-dev   (32B, quality ceiling, gated, H100)
# ---------------------------------------------------------------------------
FLUX2_MODEL_ID = os.environ.get("FLUX2_MODEL_ID", "black-forest-labs/FLUX.2-klein-4B")
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

# Score-difference -> edit intensity. The dimension deficit (THETA_GATE - score)
# is bucketed into rewrite levels 1/2/3; larger deficit => stronger edit.
# Set MISC_GRADED=0 to disable (fixed level-1) for the "graded vs fixed" ablation.
GRADED_INTENSITY = os.environ.get("MISC_GRADED", "1") != "0"
INTENSITY_T1 = float(os.environ.get("MISC_INTENSITY_T1", 0.15))  # deficit < T1 -> level 1
INTENSITY_T2 = float(os.environ.get("MISC_INTENSITY_T2", 0.35))  # deficit < T2 -> level 2, else 3

# Escalation ladder into FLUX.2-native knobs: when prompt edits on a dimension
# keep getting rejected (score stalls), stop just rewording and instead push
# FLUX.2's own levers -- raise guidance (stronger prompt adherence) + jitter the
# seed. Set MISC_KNOB_ESCALATION=0 to disable (prompt-only ladder).
KNOB_ESCALATION = os.environ.get("MISC_KNOB_ESCALATION", "1") != "0"
STUCK_TO_KNOB = int(os.environ.get("MISC_STUCK_TO_KNOB", 2))   # rejects on a dim before knobs kick in
GUIDANCE_STEP = float(os.environ.get("MISC_GUIDANCE_STEP", 1.0))
GUIDANCE_MAX = float(os.environ.get("MISC_GUIDANCE_MAX", 7.0))

# Routing mode
#   calibrated : gain-calibrated -- shortfall vs per-dim baseline x fixability.
#                Default. Fixes the "raw argmin always picks Interaction" trap.
#   diagnostic : raw argmin over absolute E/A/I (kept as the degenerate ablation).
#   random / static : baselines.
ROUTING_MODES = ["calibrated", "diagnostic", "random", "static"]
ROUTING_DEFAULT = os.environ.get("MISC_ROUTING", "calibrated")

# Per-dimension reference levels (typical scores). In practice E > A > I, so raw
# argmin degenerates to always-Interaction. Calibrated routing compares each dim
# to its OWN typical level. Ideally these are calibrated on the dev set; the
# defaults encode the empirical E>A>I ordering.
MIE_REF = {
    "existence": float(os.environ.get("MIE_REF_E", 0.85)),
    "appearance": float(os.environ.get("MIE_REF_A", 0.70)),
    "interaction": float(os.environ.get("MIE_REF_I", 0.50)),
}
# Fixability priors (how much an action on this dim tends to help). I is hardest.
FIXABILITY = {
    "existence": float(os.environ.get("MIE_FIX_E", 1.0)),
    "appearance": float(os.environ.get("MIE_FIX_A", 0.8)),
    "interaction": float(os.environ.get("MIE_FIX_I", 0.4)),
}
# Optional: update fixability online from observed gains (training-free bandit).
# Off by default for reproducibility.
ONLINE_GAIN = os.environ.get("MISC_ONLINE_GAIN", "0") != "0"
ONLINE_GAIN_LR = float(os.environ.get("MISC_ONLINE_GAIN_LR", 0.3))


def steps_for_model() -> int:
    return STEPS_KLEIN if FLUX2_IS_KLEIN else STEPS_DEV
