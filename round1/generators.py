"""Generators. A generator turns (prompt, reference images, seed) into an image.

- OmniGen2Generator: the shared base model for p1/p2/p3 (in-context multi-ref).
  Weights auto-download from HF on first use on the GPU box:
  https://huggingface.co/OmniGen2/OmniGen2
- MockGenerator: deterministic placeholder so the pipeline logic runs on a
  laptop with no GPU / no weights.

NOTE on the OmniGen2 call: the exact kwargs can vary across diffusers versions.
The actual `pipe(...)` invocation is isolated in `_call_pipe` so you can fix it
in ONE place after `pip install`. Known knobs from the model card are wired:
text_guidance_scale, image_guidance_scale, max_pixels, negative_prompt.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
OMNIGEN2_SRC = REPO_ROOT / "external" / "UMO" / "projects" / "OmniGen2"


class BaseGenerator:
    def generate(self, prompt: str, refs: list[Image.Image], seed: int = 0) -> Image.Image:
        raise NotImplementedError


class MockGenerator(BaseGenerator):
    """Draws a deterministic image seeded by (prompt, seed). No model needed."""

    def __init__(self, size: int = 256):
        self.size = size

    def generate(self, prompt: str, refs: list[Image.Image], seed: int = 0) -> Image.Image:
        import hashlib

        h = hashlib.md5(f"{prompt}|{seed}".encode()).digest()
        img = Image.new("RGB", (self.size, self.size),
                        (h[0], h[1], h[2]))
        # paste downscaled refs so SCR sees *some* signal in mock mode
        x = 0
        for r in refs[:4]:
            t = r.resize((self.size // 4, self.size // 4))
            img.paste(t, (x, 0))
            x += self.size // 4
        return img


class OmniGen2Generator(BaseGenerator):
    """Official OmniGen2 multi-reference pipeline, loaded once per process."""

    def __init__(
        self,
        model_id: str = "OmniGen2/OmniGen2",
        dtype: str = "bfloat16",
        max_pixels: int = 1024 * 1024,
        text_guidance_scale: float = 5.0,
        image_guidance_scale: float = 2.5,
        negative_prompt: str = "blurry, low quality, text, watermark, deformed",
        cpu_offload: bool = True,
    ):
        if not OMNIGEN2_SRC.exists():
            raise FileNotFoundError(
                f"OmniGen2 source not found at {OMNIGEN2_SRC}. "
                "Run round1/setup_round1.sh first."
            )
        sys.path.insert(0, str(OMNIGEN2_SRC))

        import torch
        from transformers import CLIPProcessor
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
        from omnigen2.models.transformers.transformer_omnigen2 import (
            OmniGen2Transformer2DModel,
        )

        self.torch = torch
        td = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        self.pipe = OmniGen2Pipeline.from_pretrained(
            model_id,
            processor=CLIPProcessor.from_pretrained(
                model_id, subfolder="processor", use_fast=True
            ),
            torch_dtype=td,
            trust_remote_code=True,
        )
        self.pipe.transformer = OmniGen2Transformer2DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=td
        )
        if cpu_offload and torch.cuda.is_available():
            self.pipe.enable_model_cpu_offload()
        elif torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")
        self.max_pixels = max_pixels
        self.tgs = text_guidance_scale
        self.igs = image_guidance_scale
        self.neg = negative_prompt

    def _call_pipe(self, prompt: str, refs: list[Image.Image], generator):
        return self.pipe(
            prompt=prompt,
            input_images=refs or None,
            negative_prompt=self.neg,
            text_guidance_scale=self.tgs,
            image_guidance_scale=self.igs,
            max_pixels=self.max_pixels,
            align_res=False,
            width=1024,
            height=1024,
            num_inference_steps=50,
            max_sequence_length=1024,
            cfg_range=(0.0, 1.0),
            num_images_per_prompt=1,
            generator=generator,
            output_type="pil",
        )

    def generate(self, prompt: str, refs: list[Image.Image], seed: int = 0) -> Image.Image:
        refs = [ImageOps.exif_transpose(image.convert("RGB")) for image in refs]
        gen = self.torch.Generator(
            device="cuda" if self.torch.cuda.is_available() else "cpu"
        ).manual_seed(int(seed))
        out = self._call_pipe(prompt, refs, gen)
        return out.images[0]


def build_generator(kind: str = "omnigen2") -> BaseGenerator:
    kind = (kind or os.environ.get("ROUND1_GEN", "omnigen2")).lower()
    if kind == "mock":
        return MockGenerator()
    if kind == "omnigen2":
        local_model = REPO_ROOT / "models" / "OmniGen2"
        return OmniGen2Generator(
            model_id=os.environ.get(
                "OMNIGEN2_MODEL_ID",
                str(local_model) if local_model.exists() else "OmniGen2/OmniGen2",
            )
        )
    raise ValueError(f"unknown generator: {kind}")
