"""Persistent Qwen-based MIE scoring server.

This process runs in its own Unsloth environment because the MIE Qwen3.5 stack
and OmniGen2 require incompatible torch/transformers versions. The parent
process sends JSON lines containing a candidate image path, raw prompt and
reference paths; this server returns the MIE preference score and calibrated
head probabilities (Existence/Appearance/Interaction).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path


RESULT_PREFIX = "MIE_RESULT\t"
READY_PREFIX = "MIE_READY\t"
ROOT = Path(__file__).resolve().parents[1]
MIE_CODE = (
    ROOT
    / "MIBE_Core"
    / "Multi-Subject-Personalization-Evaluation-P13N-Workshop-feat-neurips-lens"
    / "Model_Training_Paper_Coding"
)


def _checkpoint_file(directory: Path, current: str, legacy: str) -> Path:
    current_path = directory / current
    if current_path.exists():
        return current_path
    legacy_path = directory / legacy
    if legacy_path.exists():
        return legacy_path
    raise FileNotFoundError(
        f"checkpoint requires {current} (or legacy {legacy}) under {directory}"
    )


def load_runtime(checkpoint_dir: str):
    # Unsloth must be imported before transformers/MIE. Redirect package/model
    # diagnostics to stderr so stdout remains a clean JSON-lines protocol.
    with contextlib.redirect_stdout(sys.stderr):
        import unsloth
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor
        from unsloth import FastVisionModel

        sys.path.insert(0, str(MIE_CODE))
        from mie.model import MIE
        from mie.utils.image_processing import load_image_with_safety

        ckpt = Path(checkpoint_dir).expanduser().resolve()
        config_path = _checkpoint_file(ckpt, "mie_config.json", "lens_config.json")
        heads_path = _checkpoint_file(ckpt, "mie_heads.pt", "lens_heads.pt")
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        base_name = cfg.get("base_model_name") or cfg.get("model_name")
        if not base_name:
            raise KeyError(f"{config_path} has no base_model_name/model_name")
        mode = cfg["mode"]
        init_mode = "head_only" if mode in {"lora", "lora_layer"} else mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = MIE(
            model_name=base_name,
            num_error_classes=int(cfg.get("num_error_classes", 3)),
            mode=init_mode,
            unfreeze_layers=int(cfg.get("unfreeze_layers", 4)),
        )
        heads_state = torch.load(heads_path, map_location="cpu")
        model.score_head.load_state_dict(heads_state["score_head"])
        model.classification_head.load_state_dict(
            heads_state["classification_head"]
        )

        lora_dir = ckpt / "lora_adapter"
        if mode in {"lora", "lora_layer"}:
            if not lora_dir.is_dir():
                raise FileNotFoundError(f"missing LoRA adapter: {lora_dir}")
            model.backbone = PeftModel.from_pretrained(
                model.base_model, str(lora_dir), is_trainable=False
            )

        if mode in {"layer_only", "partial", "lora_layer", "full"}:
            updates_path = ckpt / "trainable_backbone.pt"
            if not updates_path.exists():
                raise FileNotFoundError(
                    f"missing trainable backbone weights: {updates_path}"
                )
            updates = torch.load(updates_path, map_location="cpu")
            model.backbone.load_state_dict(updates, strict=False)

        model.eval()
        model.to(device)
        try:
            FastVisionModel.for_inference(model.backbone)
        except Exception as exc:
            print(f"[mie_server] for_inference skipped: {exc}", file=sys.stderr)

        processor = AutoProcessor.from_pretrained(
            base_name, trust_remote_code=True
        )
        if (
            getattr(processor, "tokenizer", None)
            and processor.tokenizer.pad_token is None
        ):
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return {
        "torch": torch,
        "model": model,
        "processor": processor,
        "device": device,
        "load_image": load_image_with_safety,
        "config": cfg,
    }


def score(runtime: dict, request: dict) -> dict:
    torch = runtime["torch"]
    processor = runtime["processor"]
    device = runtime["device"]
    refs = [
        runtime["load_image"](str(path)) for path in request["ref_paths"]
    ]
    generated = runtime["load_image"](str(request["image_path"]))
    prompt = request["prompt"]

    content = [{"type": "image"} for _ in refs]
    content.append({"type": "image"})
    content.append(
        {
            "type": "text",
            "text": (
                "You are evaluating a multi-subject personalization result. "
                "The first images are subject references. "
                "The last image is the generated candidate. "
                f"Prompt: {prompt}"
            ),
        }
    )
    text = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=False,
    )
    inputs = processor(
        text=[text],
        images=refs + [generated],
        return_tensors="pt",
        padding=False,
        truncation=False,
    )
    model_inputs = {
        key: value.to(device)
        for key, value in inputs.items()
        if key
        in {
            "input_ids",
            "attention_mask",
            "pixel_values",
            "mm_token_type_ids",
            "image_grid_thw",
            "pixel_values_videos",
        }
    }
    with torch.inference_mode(), contextlib.redirect_stdout(sys.stderr):
        raw_score, logits = runtime["model"](**model_inputs)
        probs = torch.sigmoid(logits.squeeze(0)).detach().cpu().tolist()
    return {
        "total": float(raw_score.squeeze().item()),
        "existence": float(probs[0]),
        "appearance": float(probs[1]),
        "interaction": float(probs[2]),
    }


def emit(prefix: str, payload: dict) -> None:
    print(prefix + json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    try:
        runtime = load_runtime(args.checkpoint)
    except Exception as exc:
        emit(
            READY_PREFIX,
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        raise

    emit(
        READY_PREFIX,
        {
            "ok": True,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "base_model": runtime["config"].get("base_model_name"),
        },
    )
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "shutdown":
                emit(RESULT_PREFIX, {"ok": True, "shutdown": True})
                break
            result = score(runtime, request)
            emit(RESULT_PREFIX, {"ok": True, **result})
        except Exception as exc:
            emit(
                RESULT_PREFIX,
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )


if __name__ == "__main__":
    main()
