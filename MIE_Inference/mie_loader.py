"""Load a trained MIE checkpoint and score (references, candidate, prompt) triples.

Standalone replacement for the loader embedded in `submission/round1/mie_server.py`,
which hard-codes `<repo>/MIBE_Core/...` and broke when the repo was restructured.
Every path here is resolved from an environment variable with a documented
default, so the same file works on the training box, on the eval box, and locally.

Environment
-----------
MIE_CKPT      checkpoint directory (required; see layout below)
MIE_CODE      directory containing the importable `mie` package
              (the MIBE `Model_Training_Paper_Coding` folder)
HF_HOME       Hugging Face cache holding the base backbone
MIE_DEVICE    "cuda" | "cpu"  (default: cuda when available)

Checkpoint layout
-----------------
    <MIE_CKPT>/
      mie_config.json        or legacy lens_config.json
      mie_heads.pt           or legacy lens_heads.pt
      lora_adapter/          required when mode is lora / lora_layer
      trainable_backbone.pt  required when mode is
                             layer_only / partial / lora_layer / full

Why `import unsloth` comes first: Unsloth monkey-patches transformers on import,
so importing transformers first silently yields an unpatched model. Why stdout is
redirected: `mie_server.py` speaks a JSON-lines protocol on stdout, and both
Unsloth and transformers print banners.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

# Defaults match the RunPod boxes used for this project; override via env.
DEFAULT_MIE_CODE = (
    "/workspace/misc/MIBE_Core/"
    "Multi-Subject-Personalization-Evaluation-P13N-Workshop-feat-neurips-lens/"
    "Model_Training_Paper_Coding"
)

PROMPT_TEMPLATE = (
    "You are evaluating a multi-subject personalization result. "
    "The first images are subject references. "
    "The last image is the generated candidate. "
    "Prompt: {prompt}"
)

DIMS = ("existence", "appearance", "interaction")


class MieLoadError(RuntimeError):
    """Raised when a checkpoint is incomplete or its weights do not apply."""


def _pick(directory: Path, current: str, legacy: str) -> Path:
    """Prefer the current filename, fall back to the pre-rename one."""
    for name in (current, legacy):
        candidate = directory / name
        if candidate.exists():
            return candidate
    raise MieLoadError(
        f"{directory} contains neither {current} nor legacy {legacy}"
    )


def resolve_mie_code() -> Path:
    """Locate the importable `mie` package, checking the repo before the box default."""
    explicit = os.environ.get("MIE_CODE")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not (path / "mie").is_dir():
            raise MieLoadError(f"MIE_CODE={path} has no `mie/` package inside")
        return path

    # In-repo copy, for running against a checked-out companion tree.
    repo_root = Path(__file__).resolve().parent.parent
    in_repo = (
        repo_root / "companion" / "MIBE_Core"
        / "Multi-Subject-Personalization-Evaluation-P13N-Workshop-feat-neurips-lens"
        / "Model_Training_Paper_Coding"
    )
    for candidate in (in_repo, Path(DEFAULT_MIE_CODE)):
        if (candidate / "mie").is_dir():
            return candidate.resolve()
    raise MieLoadError(
        "cannot find the `mie` package; set MIE_CODE to the MIBE "
        "Model_Training_Paper_Coding directory"
    )


def load_runtime(checkpoint_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Build a ready-to-score MIE runtime. Verifies that weights actually applied."""
    ckpt_arg = checkpoint_dir or os.environ.get("MIE_CKPT")
    if not ckpt_arg:
        raise MieLoadError("no checkpoint given and MIE_CKPT is unset")
    ckpt = Path(ckpt_arg).expanduser().resolve()
    if not ckpt.is_dir():
        raise MieLoadError(f"checkpoint directory does not exist: {ckpt}")

    mie_code = resolve_mie_code()
    report: dict[str, Any] = {"checkpoint": str(ckpt), "mie_code": str(mie_code)}

    # Everything noisy happens inside this block so stdout stays protocol-clean.
    with contextlib.redirect_stdout(sys.stderr):
        import unsloth  # noqa: F401  (must precede transformers)
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor
        from unsloth import FastVisionModel

        sys.path.insert(0, str(mie_code))
        from mie.model import MIE
        from mie.utils.image_processing import load_image_with_safety

        config_path = _pick(ckpt, "mie_config.json", "lens_config.json")
        heads_path = _pick(ckpt, "mie_heads.pt", "lens_heads.pt")
        cfg = json.loads(config_path.read_text(encoding="utf-8"))

        base_name = cfg.get("base_model_name") or cfg.get("model_name")
        if not base_name:
            raise MieLoadError(f"{config_path} has no base_model_name/model_name")
        mode = cfg["mode"]
        report.update(base_model=base_name, mode=mode)

        want = os.environ.get("MIE_DEVICE")
        device = torch.device(want or ("cuda" if torch.cuda.is_available() else "cpu"))

        # `lora`/`lora_layer` attach the adapter after construction, so the model
        # itself is built head-only to avoid double-wrapping the backbone.
        init_mode = "head_only" if mode in {"lora", "lora_layer"} else mode
        model = MIE(
            model_name=base_name,
            num_error_classes=int(cfg.get("num_error_classes", 3)),
            mode=init_mode,
            unfreeze_layers=int(cfg.get("unfreeze_layers", 4)),
        )

        heads_state = torch.load(heads_path, map_location="cpu", weights_only=False)
        model.score_head.load_state_dict(heads_state["score_head"])
        model.classification_head.load_state_dict(heads_state["classification_head"])
        report["heads_loaded"] = sorted(heads_state.keys())

        if mode in {"lora", "lora_layer"}:
            lora_dir = ckpt / "lora_adapter"
            if not lora_dir.is_dir():
                raise MieLoadError(f"mode={mode} but no LoRA adapter at {lora_dir}")
            model.backbone = PeftModel.from_pretrained(
                model.base_model, str(lora_dir), is_trainable=False
            )
            n_lora = sum(1 for n, _ in model.backbone.named_parameters() if "lora_" in n)
            if n_lora == 0:
                raise MieLoadError(
                    f"LoRA adapter at {lora_dir} attached zero lora_* parameters"
                )
            report["lora_params"] = n_lora

        if mode in {"layer_only", "partial", "lora_layer", "full"}:
            updates_path = ckpt / "trainable_backbone.pt"
            if not updates_path.exists():
                raise MieLoadError(
                    f"mode={mode} but no trainable backbone at {updates_path}"
                )
            updates = torch.load(updates_path, map_location="cpu", weights_only=False)
            # strict=False is required (the file holds only the trainable subset),
            # so we inspect the result instead of trusting it -- a silent no-op
            # here would look exactly like a successful load.
            result = model.backbone.load_state_dict(updates, strict=False)
            applied = len(updates) - len(result.unexpected_keys)
            if applied == 0:
                raise MieLoadError(
                    f"none of the {len(updates)} backbone tensors in {updates_path.name} "
                    "matched a model parameter -- key names disagree"
                )
            report["backbone_tensors_in_file"] = len(updates)
            report["backbone_tensors_applied"] = applied
            report["backbone_unexpected_keys"] = len(result.unexpected_keys)

        model.eval()
        model.to(device)
        try:
            FastVisionModel.for_inference(model.backbone)
        except Exception as exc:  # non-fatal: only disables an inference fast path
            print(f"[mie_loader] for_inference skipped: {exc}", file=sys.stderr)
            report["for_inference"] = f"skipped: {type(exc).__name__}"
        else:
            report["for_inference"] = "ok"

        processor = AutoProcessor.from_pretrained(base_name, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    report["device"] = str(device)
    return {
        "torch": torch,
        "model": model,
        "processor": processor,
        "device": device,
        "load_image": load_image_with_safety,
        "config": cfg,
        "report": report,
    }


def score(runtime: dict[str, Any], image_path: str, ref_paths: list[str],
          prompt: str) -> dict[str, float]:
    """Score one candidate against its references. Returns total + the three facets."""
    torch = runtime["torch"]
    processor = runtime["processor"]
    load_image = runtime["load_image"]

    refs = [load_image(str(p)) for p in ref_paths]
    candidate = load_image(str(image_path))

    # Reference images first, candidate last -- the order the heads were trained on.
    content: list[dict[str, Any]] = [{"type": "image"} for _ in refs]
    content.append({"type": "image"})
    content.append({"type": "text", "text": PROMPT_TEMPLATE.format(prompt=prompt)})
    text = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=False,
    )
    inputs = processor(
        text=[text], images=refs + [candidate],
        return_tensors="pt", padding=False, truncation=False,
    )
    keep = {
        "input_ids", "attention_mask", "pixel_values",
        "mm_token_type_ids", "image_grid_thw", "pixel_values_videos",
    }
    model_inputs = {
        k: v.to(runtime["device"]) for k, v in inputs.items() if k in keep
    }

    with torch.inference_mode(), contextlib.redirect_stdout(sys.stderr):
        raw_score, logits = runtime["model"](**model_inputs)
        probs = torch.sigmoid(logits.squeeze(0)).detach().cpu().tolist()

    out = {"total": float(raw_score.squeeze().item())}
    out.update({dim: float(probs[i]) for i, dim in enumerate(DIMS)})
    return out
