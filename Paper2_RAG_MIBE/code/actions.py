"""Typed actions: the controller maps the weak MIE *dimension* to an input-side
edit. Actions operate ONLY on FLUX.2 inputs (prompt + reference set) -- never on
pixels (FLUX.2 has no inpaint). See idea.md sections 2 and 5.

Key point (why this is honest about MIE): MIE only returns scores, so it tells
us WHICH dimension is weakest and HOW weak (the score gap), not WHICH subject is
wrong or HOW to fix it. So:
  - the weakest dimension chooses the *kind* of edit (propose),
  - the score DEFICIT (theta - score) + escalation choose the edit *intensity*
    (levels 1/2/3) -- this uses the graded scores, not just the ranking,
  - subject-level targeting is done by SEARCH: across steps we cycle which
    subject to emphasize / boost, and MIE-as-verifier keeps what helps.
This is propose-verify, not diagnose-fix.

A pipeline "state" is a dict: {"prompt": str, "ref_images": [PIL]}.
All prompt edits are additive/reordering only -> bounded semantic drift.
"""
from __future__ import annotations

import random
from typing import Optional

import config
import llm


def _subject_for_step(subject_names, step: int) -> Optional[str]:
    """Search over subjects: pick one to target this step (round-robin)."""
    if not subject_names:
        return None
    return subject_names[step % len(subject_names)]


def level_from_score(score: float, escalation: int = 0) -> int:
    """Graded intensity from the score DEFICIT (theta - score) + escalation.

    Larger deficit -> stronger edit. Escalation bumps the level when previous
    edits on this dimension were not accepted. Returns 1..3.
    """
    if not config.GRADED_INTENSITY:
        return 1
    deficit = max(0.0, config.THETA_GATE - float(score))
    if deficit < config.INTENSITY_T1:
        base = 1
    elif deficit < config.INTENSITY_T2:
        base = 2
    else:
        base = 3
    return max(1, min(3, base + escalation))


# --------------------------------------------------------------------------
# Rule-based prompt rewrites (deterministic). `level` (1/2/3) scales emphasis.
# --------------------------------------------------------------------------
def rewrite_existence_rule(prompt: str, subject: Optional[str], level: int) -> str:
    tag = subject or "every described subject"
    pre = {
        1: f"Clearly include {tag}.",
        2: f"The image must clearly show {tag}.",
        3: f"It is essential that {tag} is fully and prominently visible.",
    }[level]
    return f"{pre} {prompt}"


def rewrite_appearance_rule(prompt: str, subject: Optional[str], level: int) -> str:
    tag = subject or "each subject"
    post = {
        1: f"Ensure {tag} matches its reference identity.",
        2: f"{tag} must exactly match the reference identity, including distinctive features.",
        3: f"{tag} must be a faithful, exact match to the reference identity in every detail.",
    }[level]
    return f"{prompt} {post}"


def rewrite_interaction_rule(prompt: str, level: int, rng: random.Random) -> str:
    phrase = rng.choice(config.INTERACTION_PHRASES)
    post = {
        1: f"The subjects are {phrase}.",
        2: f"The subjects are clearly {phrase}, visibly interacting.",
        3: f"The subjects are unmistakably {phrase}, directly interacting with each other.",
    }[level]
    return f"{prompt} {post}"


def rewrite_prompt_rule(prompt: str, weak_dim: str, subject: Optional[str],
                        level: int, rng: random.Random) -> str:
    if weak_dim == "existence":
        return rewrite_existence_rule(prompt, subject, level)
    if weak_dim == "appearance":
        return rewrite_appearance_rule(prompt, subject, level)
    return rewrite_interaction_rule(prompt, level, rng)


# --------------------------------------------------------------------------
# LLM-based prompt rewrite (generality: not tied to hand-written templates).
# The score/level is passed so the LLM can scale its emphasis too.
# --------------------------------------------------------------------------
def rewrite_prompt_llm(prompt: str, weak_dim: str, level: int, original_prompt: str,
                       diag: Optional[dict] = None) -> str:
    """MIE-steered prompt upsampling: feed the FULL score profile to the rewriter
    (ideally FLUX.2's own Mistral upsampler) so it expands the prompt into the
    dense, structured style FLUX.2 prefers, biased toward the weak dimension."""
    strength = {1: "slightly", 2: "clearly", 3: "strongly"}[level]
    profile = ""
    if diag:
        profile = (
            f"MIE scores (0-1): preference={diag.get('total'):.2f}, "
            f"existence={diag.get('existence'):.2f}, appearance={diag.get('appearance'):.2f}, "
            f"interaction={diag.get('interaction'):.2f}. "
        )
    text = (
        "You expand image prompts into the dense, structured style FLUX.2 prefers. "
        "Rewrite the prompt to fix the weakest aspect while preserving the original "
        "meaning. Do NOT remove any subject or attribute; only add clarity/detail.\n"
        f"{profile}Weakest dimension to fix: {weak_dim} (emphasize {strength}).\n"
        f"Original (must stay semantically equivalent): {original_prompt}\n"
        f"Current prompt: {prompt}\n"
        "Reply ONLY the rewritten prompt, no quotes, no explanation."
    )
    out = llm.chat([{"role": "user", "content": text}], model=config.LLM_MODEL,
                   temperature=0.4, max_tokens=256)
    return out.strip().strip('"')


# --------------------------------------------------------------------------
# Reference-set reweighting (P2, FLUX.2-native). `level` controls how many
# duplicated copies of the boosted subject's reference are prepended.
# --------------------------------------------------------------------------
def reweight_reference(state: dict, subject: Optional[str], level: int,
                       refs_by_subject: dict) -> Optional[dict]:
    if not subject or subject not in refs_by_subject:
        return None
    subj_refs = refs_by_subject.get(subject, [])
    if not subj_refs:
        return None
    others = [img for name, imgs in refs_by_subject.items() if name != subject for img in imgs]
    copies = [subj_refs[0]] * level  # stronger level -> more duplication
    new_state = dict(state)
    new_state["ref_images"] = copies + subj_refs[1:] + others
    return new_state


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------
def apply_action(state: dict, weak_dim: str, level: int, action_mode: str,
                 rng: random.Random, original_prompt: str, refs_by_subject: dict,
                 subject_names, step: int, diag: Optional[dict] = None) -> dict:
    """Return a new state with the chosen action applied.

    `weak_dim` = weakest dimension (routing); `level` = graded intensity from
    the score deficit (+escalation); `step` = index used to search over subjects;
    `diag` = full MIE score profile, passed to the steered LLM rewriter.
    """
    subject = _subject_for_step(subject_names, step)

    if action_mode == "reference":
        if weak_dim != "interaction":
            boosted = reweight_reference(state, subject, level, refs_by_subject)
            if boosted is not None:
                return boosted
        new_state = dict(state)
        new_state["prompt"] = rewrite_prompt_rule(state["prompt"], weak_dim, subject, level, rng)
        return new_state

    if action_mode == "prompt_llm" and llm.is_configured():
        new_state = dict(state)
        new_state["prompt"] = rewrite_prompt_llm(state["prompt"], weak_dim, level,
                                                 original_prompt, diag)
        return new_state

    new_state = dict(state)
    new_state["prompt"] = rewrite_prompt_rule(state["prompt"], weak_dim, subject, level, rng)
    return new_state
