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
        cpu_offload: bool = True,
    ):
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
            num_inference_steps=50,
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
