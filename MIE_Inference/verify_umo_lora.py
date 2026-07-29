"""Is the UMO LoRA actually active at inference, or a silent no-op?

The paper's footnote argues it is active from a mean absolute pixel difference of
10.44 between the base and UMO runners. That number is not conclusive: those two
runners also differ in negative prompt, image_guidance_scale (2.5 vs 2.0) and
max_pixels, so the pixel difference could be entirely explained by config drift
with the LoRA contributing nothing.

This isolates the LoRA. One pipeline, one config (UMO's), one seed; generate,
fuse the LoRA, generate again. Any difference is attributable to the adapter and
nothing else.

It also reports what `UMOGenerator` throws away: `load_state_dict(..., strict=False)`
returns the missing/unexpected key lists, and a complete key-name mismatch --
which would load nothing at all -- is indistinguishable from success unless you
look at them.

    python verify_umo_lora.py --data <manifest.jsonl> --out /tmp/umo_check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROUND1 = Path(os.environ.get("MIDC_ROUND1", "/workspace/misc/round1"))
sys.path.insert(0, str(ROUND1))

MODEL_ROOT = Path(os.environ.get("MIDC_MODELS", "/workspace/misc/models"))
UMO_ROOT = Path(os.environ.get("UMO_ROOT", "/workspace/misc/external/UMO"))
OMNIGEN2_SRC = UMO_ROOT / "projects" / "OmniGen2"

# Exactly the call UMOGenerator.generate makes, so the only thing that varies
# between the two generations below is whether the adapter is fused.
UMO_CALL = dict(
    width=1024, height=1024, align_res=False,
    num_inference_steps=int(os.environ.get("OMNIGEN2_STEPS", "28")),
    max_sequence_length=1024,
    text_guidance_scale=5.0,
    image_guidance_scale=2.0,
    cfg_range=(0.0, 1.0),
    negative_prompt=("(((deformed))), blurry, over saturation, bad anatomy, "
                     "disfigured, poorly drawn face, mutation, extra limbs, "
                     "fused fingers, watermark, text"),
    num_images_per_prompt=1,
    output_type="pil",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="task manifest JSONL")
    ap.add_argument("--out", default="/tmp/umo_check")
    ap.add_argument("--n_tasks", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora", default=str(MODEL_ROOT / "UMO" / "UMO_OmniGen2.safetensors"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(UMO_ROOT))
    sys.path.insert(0, str(OMNIGEN2_SRC))

    import torch
    from PIL import Image, ImageOps
    from peft import LoraConfig
    from safetensors.torch import load_file
    from transformers import CLIPProcessor
    from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
    from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel

    import common

    tasks = common.load_tasks(args.data, limit=args.n_tasks)
    print(f"[1/5] loaded {len(tasks)} tasks from {args.data}", flush=True)

    model_path = str(MODEL_ROOT / "OmniGen2")
    dtype = torch.bfloat16
    pipe = OmniGen2Pipeline.from_pretrained(
        model_path,
        processor=CLIPProcessor.from_pretrained(model_path, subfolder="processor", use_fast=True),
        torch_dtype=dtype, trust_remote_code=True,
    )
    pipe.transformer = OmniGen2Transformer2DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=dtype)
    pipe = pipe.to("cuda")
    print("[2/5] base OmniGen2 pipeline on cuda", flush=True)

    def gen(task, tag):
        refs = [ImageOps.exif_transpose(im.convert("RGB")) for im in task.load_refs()]
        g = torch.Generator(device="cuda").manual_seed(args.seed)
        res = pipe(prompt=common.omnigen_prompt(task), input_images=refs or None,
                   generator=g, **UMO_CALL)
        img = res.images[0].convert("RGB")
        img.save(out_dir / f"{task.task_id}__{tag}.png")
        return img

    base_imgs = {t.task_id: gen(t, "base") for t in tasks}
    print(f"[3/5] generated {len(base_imgs)} images with BASE weights", flush=True)

    # --- attach + fuse the released UMO adapter, exactly as UMOGenerator does ---
    pipe.transformer.add_adapter(LoraConfig(
        r=512, lora_alpha=512, lora_dropout=0, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    state = load_file(args.lora, device="cuda")
    result = pipe.transformer.load_state_dict(state, strict=False)
    n_lora_params = sum(1 for n, _ in pipe.transformer.named_parameters() if "lora_" in n)
    diag = {
        "lora_file": args.lora,
        "tensors_in_file": len(state),
        "unexpected_keys": len(result.unexpected_keys),
        "applied": len(state) - len(result.unexpected_keys),
        "lora_params_on_model": n_lora_params,
        "missing_keys_total": len(result.missing_keys),
        "missing_keys_that_are_lora": sum(1 for k in result.missing_keys if "lora_" in k),
    }
    print("[4/5] adapter load diagnostics:\n" + json.dumps(diag, indent=2), flush=True)
    if diag["applied"] == 0:
        print("!! none of the LoRA tensors matched a model parameter", flush=True)

    pipe.transformer.fuse_lora(lora_scale=1, safe_fusing=False, adapter_names=["default"])
    pipe.transformer.unload_lora()

    rows = []
    for t in tasks:
        umo_img = gen(t, "umo")
        a, b = base_imgs[t.task_id], umo_img
        import numpy as np
        A = np.asarray(a, dtype=np.int16)
        B = np.asarray(b, dtype=np.int16)
        d = np.abs(A - B)
        rows.append({
            "task_id": t.task_id,
            "mean_abs_diff": round(float(d.mean()), 4),
            "max_abs_diff": int(d.max()),
            "pct_pixels_identical": round(float((d.sum(axis=2) == 0).mean()) * 100, 2),
            "bitwise_identical": bool(d.max() == 0),
        })
        print(f"    {rows[-1]}", flush=True)

    verdict = ("LoRA IS ACTIVE" if all(not r["bitwise_identical"] for r in rows)
               and max(r["mean_abs_diff"] for r in rows) > 0.5
               else "LoRA LOOKS LIKE A NO-OP")
    summary = {"diagnostics": diag, "per_task": rows, "verdict": verdict,
               "config": {k: str(v) for k, v in UMO_CALL.items()}, "seed": args.seed}
    (out_dir / "umo_lora_check.json").write_text(json.dumps(summary, indent=1))
    print(f"\n[5/5] {verdict}")
    print(f"      wrote {out_dir / 'umo_lora_check.json'}")


if __name__ == "__main__":
    main()
