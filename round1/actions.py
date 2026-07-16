"""Training-free correction actions for the closed loop (p1_ours).

Two levers, both on the INPUT side (no weights touched):
  1. prompt rewrite  — emphasize the weakest MIE dimension in the text.
  2. reference-set manipulation — reorder / duplicate reference images to
     push the base model to attend more to a target subject. This lever is
     unique to multi-reference generators and is a core novelty of the method.

The weakest dimension picks which lever + how to phrase it:
  - existence   -> a subject is missing/merged: strengthen presence + refs.
  - appearance  -> identity drifted: strengthen the reference set (reorder/dup).
  - interaction -> relation wrong: rewrite the prompt to stress the interaction.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from common import DIMS, Task


class CalibratedRouter:
    """Route by standardized deficit relative to each dimension's own norm."""

    def __init__(self, calibration_path: str):
        payload = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        self.groups = payload["groups"]

    def _group(self, num_subjects: int) -> dict:
        key = str(num_subjects)
        if key in self.groups:
            return self.groups[key]
        nearest = min(self.groups, key=lambda value: abs(int(value) - num_subjects))
        return self.groups[nearest]

    def deficits(self, dim_scores: dict, num_subjects: int) -> dict[str, float]:
        group = self._group(num_subjects)
        return {
            dim: (float(group[dim]["median"]) - float(dim_scores[dim]))
            / float(group[dim]["scale"])
            for dim in DIMS
        }

    def route(self, dim_scores: dict, num_subjects: int) -> tuple[str, dict]:
        deficits = self.deficits(dim_scores, num_subjects)
        return max(DIMS, key=lambda dim: deficits[dim]), deficits


class RawRouter:
    """Dry-run/ablation router; never use as the primary real experiment."""

    def route(self, dim_scores: dict, num_subjects: int) -> tuple[str, dict]:
        deficits = {dim: -float(dim_scores[dim]) for dim in DIMS}
        return max(DIMS, key=lambda dim: deficits[dim]), deficits


def rewrite_prompt(prompt: str, dim: str, subject_names: list[str]) -> str:
    subjects = ", ".join(n.replace("_", " ") for n in subject_names)
    if dim == "existence":
        return (f"{prompt} Every subject must be clearly and fully visible: "
                f"{subjects}. Do not merge or omit any subject.")
    if dim == "appearance":
        return (f"{prompt} Preserve the exact identity, face and appearance of "
                f"each reference subject: {subjects}.")
    if dim == "interaction":
        return (f"{prompt} Render the described interaction between the subjects "
                f"clearly and physically plausibly.")
    return prompt


def manipulate_refset(
    refs: list[Image.Image],
    subject_names: list[str],
    target_idx: int,
    mode: str = "front_dup",
) -> list[Image.Image]:
    """Reorder/duplicate the reference list to emphasize `target_idx`.

    front_dup: move target's ref to the front AND duplicate it (stronger weight).
    front:     move target's ref to the front only.
    """
    if not refs or not (0 <= target_idx < len(refs)):
        return refs
    tgt = refs[target_idx]
    rest = [r for i, r in enumerate(refs) if i != target_idx]
    if mode == "front":
        return [tgt] + rest
    return [tgt, tgt] + rest  # front_dup


def pick_target_subject(num_subjects: int, step: int) -> int:
    """Which subject to emphasize with the reference-set lever.

    MIE gives per-dimension (not per-subject) scores, so for Round-1 we rotate
    through subjects across steps (round-robin). This is a deliberately simple
    heuristic; per-subject attribution is a Round-2 refinement.
    """
    if num_subjects <= 0:
        return 0
    return step % num_subjects


def apply_action(
    task: Task,
    prompt: str,
    refs: list[Image.Image],
    dim: str,
    step: int,
) -> tuple[str, list[Image.Image], dict]:
    """Return (new_prompt, new_refs, action_info) for the routed dimension."""
    new_prompt = rewrite_prompt(prompt, dim, task.subject_names)
    new_refs = refs

    if dim in ("appearance", "existence"):
        tgt = pick_target_subject(task.num_subjects, step)
        new_refs = manipulate_refset(refs, task.subject_names, tgt, mode="front_dup")
        action = {"dim": dim, "lever": "prompt+refset", "target_subject": tgt}
    else:
        action = {"dim": dim, "lever": "prompt"}

    return new_prompt, new_refs, action
