"""Independent metrics (the final arbiters -- deliberately NOT MIE).

  - CLIP-I / DINO : subject fidelity (image vs reference)
  - CLIP text-image : semantic guard vs ORIGINAL prompt p0 (drift check)
  - collateral_damage_rate : computed inside the pipeline from MIE traces

All model-backed metrics lazy-import torch/transformers so the mock/CPU path
has zero heavy dependencies. If deps are missing, functions return None and the
pipeline still runs (metrics simply omitted).
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import config

# Hard off-switch: if MISC_NO_METRIC_MODELS is set, never attempt to load/
# download CLIP/DINO (keeps mock/offline runs fast). Metrics return None.
_DISABLED = bool(os.environ.get("MISC_NO_METRIC_MODELS", ""))

_DEVICE = None


def _device():
    global _DEVICE
    if _DEVICE is None:
        try:
            import torch

            _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            _DEVICE = "cpu"
    return _DEVICE


@lru_cache(maxsize=1)
def _clip():
    if _DISABLED:
        return None
    try:
        import torch  # noqa: F401
        from transformers import CLIPModel, CLIPProcessor

        m = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(_device()).eval()
        p = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        return m, p
    except Exception:
        return None


@lru_cache(maxsize=1)
def _dino():
    if _DISABLED:
        return None
    try:
        import torch  # noqa: F401
        from transformers import AutoImageProcessor, AutoModel

        m = AutoModel.from_pretrained("facebook/dinov2-base").to(_device()).eval()
        p = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        return m, p
    except Exception:
        return None


def _cos(a, b):
    import torch

    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return float((a * b).sum(dim=-1).mean().item())


def clip_image_similarity(image, ref_images) -> Optional[float]:
    if not ref_images:
        return None
    bundle = _clip()
    if bundle is None:
        return None
    import torch

    model, proc = bundle
    with torch.no_grad():
        gi = proc(images=[image], return_tensors="pt").to(_device())
        ri = proc(images=list(ref_images), return_tensors="pt").to(_device())
        gf = model.get_image_features(**gi)
        rf = model.get_image_features(**ri).mean(dim=0, keepdim=True)
        return _cos(gf, rf)


def dino_similarity(image, ref_images) -> Optional[float]:
    if not ref_images:
        return None
    bundle = _dino()
    if bundle is None:
        return None
    import torch

    model, proc = bundle
    with torch.no_grad():
        gi = proc(images=[image], return_tensors="pt").to(_device())
        ri = proc(images=list(ref_images), return_tensors="pt").to(_device())
        gf = model(**gi).last_hidden_state[:, 0]
        rf = model(**ri).last_hidden_state[:, 0].mean(dim=0, keepdim=True)
        return _cos(gf, rf)


def clip_text_image_similarity(image, text) -> Optional[float]:
    """Semantic guard: similarity of final image to the ORIGINAL prompt p0."""
    bundle = _clip()
    if bundle is None:
        return None
    import torch

    model, proc = bundle
    with torch.no_grad():
        inp = proc(text=[text], images=[image], return_tensors="pt",
                   padding=True, truncation=True).to(_device())
        out = model(**inp)
        return _cos(out.image_embeds, out.text_embeds)


def independent_metrics(image, ref_images, original_prompt) -> dict:
    return {
        "clip_i": clip_image_similarity(image, ref_images),
        "dino": dino_similarity(image, ref_images),
        "clip_t": clip_text_image_similarity(image, original_prompt),
    }
