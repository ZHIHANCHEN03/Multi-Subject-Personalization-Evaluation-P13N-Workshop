"""FLUX.2 generator wrapper (diffusers Flux2Pipeline).

`generate(prompt, ref_images, seed, guidance, caption_upsample_temperature)`
returns a single PIL image. Supports:
  - multi-reference conditioning (image=[refs])  -> P2 action lever
  - native prompt upsampling (caption_upsample_temperature) if the installed
    diffusers exposes it -> caption_upsample baseline
  - optional 4-bit / 8-bit quantization for 32B [dev] on smaller GPUs
  - a "mock" backend that returns deterministic blank images (CPU smoke test)
"""
from __future__ import annotations

import inspect
from typing import Optional

from PIL import Image

import config


class MockGenerator:
    """Deterministic placeholder generator for CPU smoke tests."""

    supports_upsample = True

    def generate(self, prompt, ref_images=None, seed=0, guidance=None,
                 caption_upsample_temperature=None):
        # Solid-color image whose shade is a stable function of prompt+seed,
        # so the mock critic produces varied-but-reproducible scores.
        val = (abs(hash((prompt, int(seed)))) % 200) + 30
        return Image.new("RGB", (256, 256), (val, (val * 2) % 255, (val * 3) % 255))


class Flux2Generator:
    def __init__(self, model_id: Optional[str] = None):
        import torch
        from diffusers import Flux2Pipeline

        self.model_id = model_id or config.FLUX2_MODEL_ID
        dtype = getattr(torch, config.TORCH_DTYPE, torch.bfloat16)

        kwargs = {"torch_dtype": dtype}
        if config.FLUX2_QUANTIZE in ("nf4", "int8"):
            kwargs["quantization_config"] = self._quant_config(config.FLUX2_QUANTIZE)

        self.pipe = Flux2Pipeline.from_pretrained(self.model_id, **kwargs)
        if config.FLUX2_QUANTIZE not in ("nf4", "int8"):
            self.pipe = self.pipe.to("cuda")
        else:
            self.pipe.enable_model_cpu_offload()

        sig = inspect.signature(self.pipe.__call__)
        self.supports_upsample = "caption_upsample_temperature" in sig.parameters
        self._steps = config.steps_for_model()

    @staticmethod
    def _quant_config(mode: str):
        from diffusers import BitsAndBytesConfig  # type: ignore
        import torch

        if mode == "nf4":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        return BitsAndBytesConfig(load_in_8bit=True)

    def generate(self, prompt, ref_images=None, seed=0, guidance=None,
                 caption_upsample_temperature=None):
        import torch

        gen = torch.Generator(device="cuda").manual_seed(int(seed))
        kwargs = dict(
            prompt=prompt,
            num_inference_steps=self._steps,
            guidance_scale=config.GUIDANCE_SCALE if guidance is None else guidance,
            height=config.RESOLUTION,
            width=config.RESOLUTION,
            generator=gen,
        )
        if ref_images:
            kwargs["image"] = list(ref_images)
        if caption_upsample_temperature is not None and self.supports_upsample:
            kwargs["caption_upsample_temperature"] = caption_upsample_temperature
        out = self.pipe(**kwargs)
        return out.images[0]


def build_generator(backend: str = "flux2"):
    if backend == "mock":
        return MockGenerator()
    return Flux2Generator()
