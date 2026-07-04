"""Adapter that plugs your trained MIE (Qwen3.5-0.8B dual-head evaluator) into MISC.

Wire it in:
    export MISC_CRITIC=mie_checkpoint
    export MIE_ADAPTER=mie_adapter
    export MIE_REPO=/path/to/Model_Training           # dir containing your model package
    export MIE_CKPT=/path/to/outputs/..._layer_only-best

critic.py calls `score(image, prompt, subject_refs, subject_names) -> dict` and
expects:
    {
      "total":      float,   # MIE preference score (score head). UNBOUNDED,
                             #   comparison-only, higher = better.
      "existence":  float in [0,1],  # sigmoid(classification logit), higher=better
      "appearance": float in [0,1],
      "interaction":float in [0,1],
    }

MIE was trained pairwise (A vs B) but each head runs per single image, so we
score one candidate at a time: the score head gives the per-image preference
scalar (MISC only ever compares two candidates for the same prompt -> exactly
its trained use), and sigmoid(classification logits) gives the [0,1] E/A/I.
"""
from __future__ import annotations

import os
import sys

_MODEL = None
_PROC = None
_DEVICE = None
_DIMS = ["existence", "appearance", "interaction"]


def _lazy_init():
    global _MODEL, _PROC, _DEVICE
    if _MODEL is not None:
        return

    repo = os.environ.get("MIE_REPO", "")
    ckpt = os.environ.get("MIE_CKPT", "")
    if not repo or not ckpt:
        raise RuntimeError("Set MIE_REPO (your Model_Training dir) and MIE_CKPT (checkpoint dir).")
    if repo not in sys.path:
        sys.path.append(repo)

    import torch
    # Reuse the training repo's own checkpoint loader so we match training exactly.
    from scripts.evaluate_pipeline import load_lens_checkpoint as _load_ckpt  # type: ignore

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _MODEL, _PROC, _ = _load_ckpt(ckpt, _DEVICE)
    _MODEL.eval()


def _build_inputs(image, prompt, ref_images):
    """Mirror the training dataset's prompt builder for a single candidate image."""
    content = [{"type": "image"} for _ in ref_images]
    content.append({"type": "image"})  # the generated candidate
    content.append({
        "type": "text",
        "text": (
            "You are evaluating a multi-subject personalization result. "
            "The first images are subject references. The last image is the generated candidate. "
            f"Prompt: {prompt}"
        ),
    })
    messages = [{"role": "user", "content": content}]
    text = _PROC.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return _PROC(
        text=[text],
        images=list(ref_images) + [image],
        return_tensors="pt",
        padding=False,
        truncation=False,
    )


def score(image, prompt, subject_refs, subject_names):
    _lazy_init()
    import torch

    ref_images = list(subject_refs or [])
    inputs = _build_inputs(image, prompt, ref_images)
    kwargs = {
        "input_ids": inputs["input_ids"].to(_DEVICE),
        "attention_mask": inputs["attention_mask"].to(_DEVICE),
        "pixel_values": inputs["pixel_values"].to(_DEVICE),
    }
    if "image_grid_thw" in inputs:
        kwargs["image_grid_thw"] = inputs["image_grid_thw"].to(_DEVICE)
    if "mm_token_type_ids" in inputs:
        kwargs["mm_token_type_ids"] = inputs["mm_token_type_ids"].to(_DEVICE)

    with torch.no_grad():
        s, logits = _MODEL(**kwargs)
        probs = torch.sigmoid(logits.float().squeeze(0)).tolist()

    out = {"total": float(s.squeeze().item())}
    for i, k in enumerate(_DIMS):
        out[k] = float(probs[i])
    return out
