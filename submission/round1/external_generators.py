"""Direct adapters for the two official Round-1 external baselines.

Both adapters load their released implementation and weights once, then expose
``generate(task, seed) -> PIL.Image``. No precomputed-image handoff and no
training is involved.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from PIL import Image, ImageOps

import common


REPO_ROOT = Path(__file__).resolve().parents[1]
UMO_ROOT = REPO_ROOT / "external" / "UMO"
OMNIGEN2_SRC = UMO_ROOT / "projects" / "OmniGen2"
FG_ROOT = REPO_ROOT / "external" / "FreeGraftor"
FG_SRC = FG_ROOT / "src"
MODEL_ROOT = REPO_ROOT / "models"


class UMOGenerator:
    """Released UMO-OmniGen2 LoRA fused into the official OmniGen2 pipeline."""

    def __init__(
        self,
        model_path: str | None = None,
        lora_path: str | None = None,
        cpu_offload: bool = None,
    ):
        if cpu_offload is None:
            cpu_offload = os.environ.get("ROUND1_CPU_OFFLOAD", "0") == "1"
        if not OMNIGEN2_SRC.exists():
            raise FileNotFoundError(
                f"{OMNIGEN2_SRC} is missing; run round1/setup_round1.sh."
            )

        sys.path.insert(0, str(UMO_ROOT))
        sys.path.insert(0, str(OMNIGEN2_SRC))

        import torch
        from peft import LoraConfig
        from safetensors.torch import load_file
        from transformers import CLIPProcessor
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
        from omnigen2.models.transformers.transformer_omnigen2 import (
            OmniGen2Transformer2DModel,
        )

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        local_base = MODEL_ROOT / "OmniGen2"
        model_path = model_path or os.environ.get(
            "OMNIGEN2_MODEL_ID",
            str(local_base) if local_base.exists() else "OmniGen2/OmniGen2",
        )
        lora_path = lora_path or os.environ.get(
            "UMO_LORA",
            str(MODEL_ROOT / "UMO" / "UMO_OmniGen2.safetensors"),
        )
        if not Path(lora_path).exists():
            raise FileNotFoundError(
                f"UMO LoRA not found at {lora_path}; run round1/setup_round1.sh."
            )

        self.pipe = OmniGen2Pipeline.from_pretrained(
            model_path,
            processor=CLIPProcessor.from_pretrained(
                model_path, subfolder="processor", use_fast=True
            ),
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.pipe.transformer = OmniGen2Transformer2DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=dtype
        )
        self.pipe.transformer.add_adapter(
            LoraConfig(
                r=512,
                lora_alpha=512,
                lora_dropout=0,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
        )
        state = load_file(lora_path, device=str(self.device))
        self.pipe.transformer.load_state_dict(state, strict=False)
        self.pipe.transformer.fuse_lora(
            lora_scale=1, safe_fusing=False, adapter_names=["default"]
        )
        self.pipe.transformer.unload_lora()

        if cpu_offload and self.device.type == "cuda":
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)

    def generate(self, task: common.Task, seed: int = 0) -> Image.Image:
        refs = [
            ImageOps.exif_transpose(image.convert("RGB"))
            for image in task.load_refs()
        ]
        generator = self.torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipe(
            prompt=common.omnigen_prompt(task),
            input_images=refs or None,
            width=1024,
            height=1024,
            align_res=False,
            num_inference_steps=int(os.environ.get("OMNIGEN2_STEPS", "28")),
            max_sequence_length=1024,
            text_guidance_scale=5.0,
            image_guidance_scale=2.0,
            cfg_range=(0.0, 1.0),
            negative_prompt=(
                "(((deformed))), blurry, over saturation, bad anatomy, "
                "disfigured, poorly drawn face, mutation, extra limbs, "
                "fused fingers, watermark, text"
            ),
            num_images_per_prompt=1,
            generator=generator,
            output_type="pil",
        )
        return result.images[0].convert("RGB")


class FreeGraftorGenerator:
    """Official FreeGraftor FLUX.1-dev open-loop inference pipeline."""

    def __init__(self, cpu_offload: bool = True):
        if not FG_SRC.exists():
            raise FileNotFoundError(
                f"{FG_SRC} is missing; run round1/setup_round1.sh."
            )
        required = {
            "FLUX_DEV": MODEL_ROOT / "FLUX.1-dev",
            "GROUNDING_DINO": MODEL_ROOT / "grounding-dino-tiny",
            "SAM": MODEL_ROOT / "sam_vit_h_4b8939.pth",
        }
        for env_name, default in required.items():
            os.environ.setdefault(env_name, str(default))
            if not Path(os.environ[env_name]).exists():
                raise FileNotFoundError(
                    f"{env_name}={os.environ[env_name]} does not exist; "
                    "run round1/setup_round1.sh."
                )

        sys.path.insert(0, str(FG_SRC))
        import kornia
        from kornia.config import kornia_config

        kornia_config.lazyloader.installation_mode = "auto"
        from pipeline import ConceptConfig, FreeGraftorPipeline, GenerationConfig

        self.ConceptConfig = ConceptConfig
        self.GenerationConfig = GenerationConfig
        self.output_dir = REPO_ROOT / "round1" / "results" / "freegraftor" / "native"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipe = FreeGraftorPipeline(
            models={},
            device="cuda",
            requires_offload=cpu_offload,
            image_cache_dir=str(REPO_ROOT / "round1" / ".cache" / "fg_images"),
            image_info_cache_dir=str(REPO_ROOT / "round1" / ".cache" / "fg_info"),
        )

    def generate(self, task: common.Task, seed: int = 0) -> Image.Image:
        paths = task.flat_ref_paths()
        if len(paths) != task.num_subjects:
            raise FileNotFoundError(
                f"{task.task_id}: expected {task.num_subjects} refs, got {len(paths)}"
            )
        concepts = [
            self.ConceptConfig(
                class_name=name.replace("_", " "),
                image_path=path,
            )
            for name, path in zip(task.subject_names, paths)
        ]
        config = self.GenerationConfig(
            seed=seed,
            num_steps=25,
            guidance=3.0,
            width=1024,
            height=1024,
        )
        image = self.pipe(
            concept_configs=concepts,
            prompt=task.prompt,
            template_prompt=task.prompt,
            output_dir=str(self.output_dir / task.task_id),
            clear_image_cache=False,
            clear_image_info_cache=False,
            config=config,
        )
        return image.convert("RGB")


# ---------------------------------------------------------------------------
# FLUX.2 — for Round-3 P1-3 scaling experiment (6/8 subjects).
#
# FLUX.2 (Black Forest Labs, 2026) exposes multi-image conditioning via its
# Kontext-style interface. Unlike OmniGen2 (which has a fixed 5-slot image
# index embedding), FLUX.2's multi-image capacity is governed by the pipeline's
# image-embedding budget and max_sequence_length. This adapter:
#
#   1. Loads FLUX.2 via diffusers (FLUX2Pipeline or FLUX2KontextPipeline).
#      Falls back to FLUX.1-dev if FLUX.2 weights are absent (env FLUX2_MODEL_ID
#      unset and no local models/FLUX.2).
#   2. Binds each reference image to its subject name in the prompt (same
#      "image i is the reference for <name>" convention as OmniGen2).
#   3. Passes all refs as a list to the pipeline's image conditioning input.
#
# Capacity probe: run `python external_generators.py --probe-flux2` to test
# how many reference images FLUX.2 accepts before OOM / truncation. The result
# determines whether 6/8-subject scaling is feasible on this base.
#
# Env:
#   FLUX2_MODEL_ID   HF id or local path (default: models/FLUX.2 or "black-forest-labs/FLUX.2")
#   FLUX2_STEPS      inference steps (default 28)
#   FLUX2_GUIDANCE   text guidance scale (default 4.0)
#   FLUX2_MAX_REFS   hard cap on reference images (default 8; probe first)
#   FLUX2_KONTEXT    "1" to use Kontext multi-image pipeline (default "1")
# ---------------------------------------------------------------------------


class Flux2Generator:
    """FLUX.2 multi-reference generator for 6/8-subject scaling experiments.

    Training-free, inference-only. The reference images are passed as a list
    to FLUX.2's image-conditioning input; the prompt binds each image ordinal
    to its subject name so the model knows which ref is which.
    """

    MAX_REFS = 8  # default cap; run probe to verify per-pipeline

    def __init__(
        self,
        model_id: str | None = None,
        dtype: str = "bfloat16",
        cpu_offload: bool = None,
        use_kontext: bool = None,
    ):
        import torch
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        td = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]

        if cpu_offload is None:
            cpu_offload = os.environ.get("ROUND1_CPU_OFFLOAD", "0") == "1"
        if use_kontext is None:
            use_kontext = os.environ.get("FLUX2_KONTEXT", "1") == "1"

        local_flux2 = MODEL_ROOT / "FLUX.2"
        model_id = model_id or os.environ.get(
            "FLUX2_MODEL_ID",
            str(local_flux2) if local_flux2.exists() else "black-forest-labs/FLUX.2",
        )
        self.model_id = model_id
        self.use_kontext = use_kontext
        self.MAX_REFS = int(os.environ.get("FLUX2_MAX_REFS", str(self.MAX_REFS)))
        self.steps = int(os.environ.get("FLUX2_STEPS", "28"))
        self.guidance = float(os.environ.get("FLUX2_GUIDANCE", "4.0"))

        # Try FLUX.2 Kontext (multi-image) first, then FLUX.2 base, then FLUX.1-dev fallback.
        self.pipe = None
        self._pipeline_kind = None
        try:
            self.pipe = self._load_flux2_kontext(model_id, td) if use_kontext else None
            self._pipeline_kind = "flux2_kontext"
        except Exception as exc:
            print(f"[Flux2Generator] FLUX.2 Kontext load failed: {exc}; trying FLUX.2 base")
        if self.pipe is None:
            try:
                self.pipe = self._load_flux2_base(model_id, td)
                self._pipeline_kind = "flux2_base"
            except Exception as exc:
                print(f"[Flux2Generator] FLUX.2 base load failed: {exc}; falling back to FLUX.1-dev")
        if self.pipe is None:
            self.pipe = self._load_flux1_dev_fallback(td)
            self._pipeline_kind = "flux1_dev_fallback"
            print(f"[Flux2Generator] WARNING: using FLUX.1-dev fallback. 6/8-subject multi-ref may be limited.")

        if cpu_offload and self.device.type == "cuda":
            try:
                self.pipe.enable_model_cpu_offload()
            except Exception:
                self.pipe = self.pipe.to(self.device)
        elif self.device.type == "cuda":
            self.pipe = self.pipe.to(self.device)

        print(f"[Flux2Generator] loaded pipeline={self._pipeline_kind} model={model_id} max_refs={self.MAX_REFS}")

    def _load_flux2_kontext(self, model_id, td):
        from diffusers import Flux2KontextPipeline
        pipe = Flux2KontextPipeline.from_pretrained(model_id, torch_dtype=td, trust_remote_code=True)
        return pipe

    def _load_flux2_base(self, model_id, td):
        from diffusers import Flux2Pipeline
        pipe = Flux2Pipeline.from_pretrained(model_id, torch_dtype=td, trust_remote_code=True)
        return pipe

    def _load_flux1_dev_fallback(self, td):
        """FLUX.1-dev fallback — single-image conditioning only, so multi-subject
        refs are concatenated into a single contact-sheet reference image.
        This is a degraded mode; only used if FLUX.2 is unavailable."""
        from diffusers import FluxImg2ImgPipeline
        local_flux1 = MODEL_ROOT / "FLUX.1-dev"
        mid = str(local_flux1) if local_flux1.exists() else "black-forest-labs/FLUX.1-dev"
        pipe = FluxImg2ImgPipeline.from_pretrained(mid, torch_dtype=td)
        self.MAX_REFS = min(self.MAX_REFS, 1)  # single conditioning image
        return pipe

    def _concat_refs(self, refs: list[Image.Image]) -> Image.Image:
        """Tile reference images into a single contact sheet (fallback mode only)."""
        if not refs:
            return Image.new("RGB", (1024, 1024), (128, 128, 128))
        n = len(refs)
        cols = int(n ** 0.5) + 1
        rows = (n + cols - 1) // cols
        cell = 1024 // max(cols, rows)
        sheet = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
        for i, r in enumerate(refs):
            r = r.resize((cell, cell))
            sheet.paste(r, ((i % cols) * cell, (i // cols) * cell))
        return sheet.resize((1024, 1024))

    def generate(self, prompt: str, refs: list[Image.Image], seed: int = 0) -> Image.Image:
        refs = [ImageOps.exif_transpose(image.convert("RGB")) for image in refs]
        if len(refs) > self.MAX_REFS:
            raise ValueError(
                f"FLUX.2 supports at most {self.MAX_REFS} reference images, "
                f"got {len(refs)}; reduce subjects or increase FLUX2_MAX_REFS."
            )
        gen = self.torch.Generator(
            device="cuda" if self.torch.cuda.is_available() else "cpu"
        ).manual_seed(int(seed))

        if self._pipeline_kind == "flux2_kontext":
            # Multi-image conditioning: pass refs as a list.
            out = self.pipe(
                prompt=prompt,
                image=refs,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance,
                generator=gen,
                width=1024,
                height=1024,
                output_type="pil",
            )
        elif self._pipeline_kind == "flux2_base":
            # FLUX.2 base text-to-image with image prompt tokens if supported.
            out = self.pipe(
                prompt=prompt,
                images=refs or None,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance,
                generator=gen,
                width=1024,
                height=1024,
                output_type="pil",
            )
        else:
            # FLUX.1-dev fallback: concat refs into one image, img2img.
            sheet = self._concat_refs(refs)
            out = self.pipe(
                prompt=prompt,
                image=sheet,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance,
                generator=gen,
                width=1024,
                height=1024,
                output_type="pil",
            )
        return out.images[0].convert("RGB")


def _probe_flux2_capacity():
    """Probe how many reference images FLUX.2 can condition on.

    Generates a dummy multi-subject prompt with N refs (incrementing N) until
    the pipeline errors or truncates. Prints the max supported refs. Run this
    BEFORE the 6/8-subject scaling run to confirm capacity.
    """
    import common
    print("[probe] loading FLUX.2 ...")
    gen = Flux2Generator()
    print(f"[probe] pipeline kind: {gen._pipeline_kind}")
    print(f"[probe] configured MAX_REFS: {gen.MAX_REFS}")

    # synthetic refs: colored placeholders
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)]
    max_ok = 0
    for n in [2, 4, 6, 8]:
        if n > gen.MAX_REFS:
            break
        refs = [Image.new("RGB", (512, 512), colors[i % len(colors)]) for i in range(n)]
        names = [f"subject_{i+1}" for i in range(n)]
        bindings = "; ".join(f"image {i+1} is the reference for {names[i]}" for i in range(n))
        prompt = f"{bindings}. a photo of all the subjects together in a park."
        try:
            img = gen.generate(prompt, refs, seed=0)
            print(f"[probe] n={n} OK -> image size {img.size}")
            max_ok = n
        except Exception as exc:
            print(f"[probe] n={n} FAILED: {type(exc).__name__}: {exc}")
            break
    print(f"[probe] === max supported refs = {max_ok} ===")
    print(f"[probe] recommendation: {'6/8-subject scaling FEASIBLE' if max_ok >= 6 else '6/8-subject scaling NOT feasible on this base; use FLUX.1-dev fallback or reduce to 4 subjects'}")
    return max_ok


if __name__ == "__main__" and "--probe-flux2" in sys.argv:
    _probe_flux2_capacity()
