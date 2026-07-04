"""MIE critic: the structured (typed) verifier that drives the controller.

Returns, for a generated image:
    {
      "total": float in [0,1],
      "existence": float, "appearance": float, "interaction": float,
      "weak_subject": Optional[str],
    }

Three backends (config.CRITIC_BACKEND):
  - mie_checkpoint : your Paper-1 MIE via an adapter module (MIE_ADAPTER).
  - vlm_judge      : an LLM/VLM API judge (fallback / generality study).
  - mock           : deterministic pseudo-scores, for CPU smoke tests.
"""
from __future__ import annotations

import hashlib
import importlib
from typing import Optional

import config
import llm

DIMS = config.DIMS


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


class MIECritic:
    def __init__(self, backend: Optional[str] = None):
        self.backend = backend or config.CRITIC_BACKEND
        self._adapter = None
        if self.backend == "mie_checkpoint":
            self._load_adapter()

    def _load_adapter(self):
        if not config.MIE_ADAPTER:
            raise RuntimeError(
                "CRITIC_BACKEND=mie_checkpoint requires MIE_ADAPTER (dotted path "
                "to a module exposing score(image, prompt, subject_refs, subject_names)->dict)."
            )
        self._adapter = importlib.import_module(config.MIE_ADAPTER)
        if not hasattr(self._adapter, "score"):
            raise RuntimeError(f"MIE_ADAPTER module {config.MIE_ADAPTER} has no `score` function.")

    # ------------------------------------------------------------------
    def diagnose(self, image, prompt, subject_refs, subject_names) -> dict:
        if self.backend == "mie_checkpoint":
            d = self._diagnose_checkpoint(image, prompt, subject_refs, subject_names)
        elif self.backend == "vlm_judge":
            d = self._diagnose_vlm(image, prompt, subject_names)
        else:
            d = self._diagnose_mock(image, prompt, subject_names)
        return self._finalize(d, subject_names)

    def _finalize(self, d: dict, subject_names) -> dict:
        for k in DIMS:
            d[k] = _clip01(d.get(k, 0.0))
        # total = mean of typed dims unless the backend supplied its own.
        if "total" not in d:
            d["total"] = sum(d[k] for k in DIMS) / len(DIMS)
        d["total"] = _clip01(d["total"])
        ws = d.get("weak_subject")
        d["weak_subject"] = ws if (subject_names and ws in subject_names) else None
        return d

    # ------------------------------------------------------------------
    def _diagnose_checkpoint(self, image, prompt, subject_refs, subject_names) -> dict:
        raw = self._adapter.score(
            image=image, prompt=prompt, subject_refs=subject_refs, subject_names=subject_names
        )
        # Adapter is expected to already return the E/A/I keys; be lenient.
        return dict(raw)

    def _diagnose_vlm(self, image, prompt, subject_names) -> dict:
        names = ", ".join(subject_names) if subject_names else "the subjects"
        text = (
            "You are a strict multi-subject image evaluator. Given the target prompt "
            "and the generated image, rate three dimensions in [0,1]:\n"
            "- existence: are all required subjects present?\n"
            "- appearance: does each subject look correct/consistent?\n"
            "- interaction: is the described spatial/relational interaction correct?\n"
            f"Subjects: {names}\nPrompt: {prompt}\n"
            'Reply ONLY JSON: {"existence":x,"appearance":x,"interaction":x,'
            '"weak_subject":"<name or null>"}'
        )
        raw = llm.chat_with_images(text, [image], model=config.LLM_MODEL, temperature=0.0)
        d = llm.parse_json(raw)
        if not d:
            raise RuntimeError(f"vlm_judge returned unparseable output: {raw[:200]}")
        return d

    def _diagnose_mock(self, image, prompt, subject_names) -> dict:
        # Deterministic pseudo-scores from an image+prompt hash so runs are
        # reproducible and the control logic is exercised end-to-end.
        h = hashlib.sha256()
        h.update(prompt.encode("utf-8"))
        if image is not None:
            h.update(str(image.size).encode())
            h.update(bytes(image.resize((8, 8)).convert("L").tobytes()))
        digest = h.digest()
        scores = {DIMS[i]: (digest[i] / 255.0) for i in range(len(DIMS))}
        weak_dim = min(DIMS, key=lambda k: scores[k])
        weak_subject = None
        if subject_names:
            idx = digest[8] % len(subject_names)
            if weak_dim in ("appearance", "existence"):
                weak_subject = subject_names[idx]
        scores["weak_subject"] = weak_subject
        return scores
