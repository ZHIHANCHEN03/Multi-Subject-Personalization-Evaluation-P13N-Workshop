"""Typed actions: the controller maps the weak MIE dimension to an input-side
edit. Actions operate ONLY on FLUX.2 inputs (prompt + reference set) -- never
on pixels (FLUX.2 has no inpaint). See idea.md sections 2 and 5.2.

Two prompt variants (rule / LLM) + one reference-set variant (P2). All are
additive/reordering only, which is what gives the bounded-drift property.

A pipeline "state" here is a dict:
    {"prompt": str, "ref_images": [PIL], "ref_order": [subject_name, ...]}
"""
from __future__ import annotations

import random
from typing import Optional

import config
import llm


# --------------------------------------------------------------------------
# Rule-based prompt rewrites (deterministic; zero extra model calls)
# --------------------------------------------------------------------------
def rewrite_existence_rule(prompt: str, subject: Optional[str]) -> str:
    tag = subject or "every described subject"
    return f"Clearly include {tag}. {prompt}"


def rewrite_appearance_rule(prompt: str, subject: Optional[str]) -> str:
    tag = subject or "each subject"
    return f"{prompt} Ensure {tag} matches its reference identity exactly."


def rewrite_interaction_rule(prompt: str, rng: random.Random) -> str:
    phrase = rng.choice(config.INTERACTION_PHRASES)
    return f"{prompt} The subjects are {phrase}."


def rewrite_prompt_rule(prompt: str, weak_dim: str, subject: Optional[str],
                        rng: random.Random) -> str:
    if weak_dim == "existence":
        return rewrite_existence_rule(prompt, subject)
    if weak_dim == "appearance":
        return rewrite_appearance_rule(prompt, subject)
    return rewrite_interaction_rule(prompt, rng)


# --------------------------------------------------------------------------
# LLM-based prompt rewrite (generality: not tied to hand-written templates)
# --------------------------------------------------------------------------
def rewrite_prompt_llm(prompt: str, weak_dim: str, subject: Optional[str],
                       original_prompt: str) -> str:
    tgt = f" (focus on subject: {subject})" if subject else ""
    text = (
        "Rewrite the image prompt to fix ONE weakness while preserving the "
        "original meaning. Do not remove any described subject or attribute; "
        "only add clarity/emphasis.\n"
        f"Weak dimension: {weak_dim}{tgt}\n"
        f"Original (must stay semantically equivalent): {original_prompt}\n"
        f"Current prompt: {prompt}\n"
        "Reply ONLY the rewritten prompt, no quotes, no explanation."
    )
    out = llm.chat([{"role": "user", "content": text}], model=config.LLM_MODEL,
                   temperature=0.4, max_tokens=256)
    return out.strip().strip('"')


# --------------------------------------------------------------------------
# Reference-set reweighting (P2, FLUX.2-native)
# --------------------------------------------------------------------------
def reweight_reference(state: dict, weak_dim: str, subject: Optional[str],
                       refs_by_subject: dict) -> dict:
    """Move the weak subject's reference to the front and duplicate it once,
    strengthening its conditioning. No-op for interaction (a relation, not a
    subject) -> caller should fall back to a prompt action.
    """
    if weak_dim == "interaction" or not subject or subject not in refs_by_subject:
        return state
    subj_refs = refs_by_subject.get(subject, [])
    if not subj_refs:
        return state
    others = [
        img
        for name, imgs in refs_by_subject.items()
        if name != subject
        for img in imgs
    ]
    # boosted subject first (duplicated once), then the rest
    new_refs = [subj_refs[0], subj_refs[0]] + subj_refs[1:] + others
    new_state = dict(state)
    new_state["ref_images"] = new_refs
    return new_state


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------
def apply_action(state: dict, weak_dim: str, subject: Optional[str],
                 action_mode: str, rng: random.Random,
                 original_prompt: str, refs_by_subject: dict) -> dict:
    """Return a new state with the chosen action applied."""
    new_state = dict(state)

    if action_mode == "reference":
        boosted = reweight_reference(state, weak_dim, subject, refs_by_subject)
        if boosted is not state and boosted.get("ref_images") is not state.get("ref_images"):
            return boosted
        # fall back to rule prompt rewrite (e.g. interaction / no refs)
        new_state["prompt"] = rewrite_prompt_rule(state["prompt"], weak_dim, subject, rng)
        return new_state

    if action_mode == "prompt_llm" and llm.is_configured():
        new_state["prompt"] = rewrite_prompt_llm(state["prompt"], weak_dim, subject, original_prompt)
        return new_state

    # default: prompt_rule (also the fallback when LLM not configured)
    new_state["prompt"] = rewrite_prompt_rule(state["prompt"], weak_dim, subject, rng)
    return new_state
