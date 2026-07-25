"""Dependency-free validation before downloading models or starting Round 1."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    errors, warnings = [], []
    ckpt = Path(args.checkpoint)
    config = next(
        (ckpt / name for name in ("mie_config.json", "lens_config.json")
         if (ckpt / name).exists()),
        None,
    )
    heads = next(
        (ckpt / name for name in ("mie_heads.pt", "lens_heads.pt")
         if (ckpt / name).exists()),
        None,
    )
    cfg = {}
    if not ckpt.is_dir():
        errors.append(f"checkpoint directory missing: {ckpt}")
    if config is None:
        errors.append("checkpoint missing mie_config.json/lens_config.json")
    else:
        cfg = json.loads(config.read_text(encoding="utf-8"))
    if heads is None:
        errors.append("checkpoint missing mie_heads.pt/lens_heads.pt")
    mode = cfg.get("mode")
    if mode in {"layer_only", "partial", "lora_layer", "full"}:
        if not (ckpt / "trainable_backbone.pt").exists():
            errors.append("checkpoint missing trainable_backbone.pt")
    if mode in {"lora", "lora_layer"} and not (ckpt / "lora_adapter").is_dir():
        errors.append("checkpoint missing lora_adapter/")

    prompt = Path(args.prompt)
    refs = Path(args.refs)
    if not prompt.is_file():
        errors.append(f"prompt JSONL missing: {prompt}")
    if not refs.is_dir():
        errors.append(f"refs directory missing: {refs}")
    ref_count = (
        len([path for path in refs.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
        if refs.is_dir()
        else 0
    )
    if ref_count < 80:
        warnings.append(f"only {ref_count} reference images found (expected >=80)")

    gpu = None
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()
    except Exception:
        errors.append("nvidia-smi unavailable; run this workflow on the H100 server")

    free_gib = shutil.disk_usage(Path(args.out).parent).free / (1024**3)
    if free_gib < 100:
        warnings.append(f"only {free_gib:.1f} GiB free; 150+ GiB recommended")

    report = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "checkpoint": str(ckpt),
        "checkpoint_mode": mode,
        "checkpoint_base": cfg.get("base_model_name", cfg.get("model_name")),
        "prompt": str(prompt),
        "refs": str(refs),
        "ref_count": ref_count,
        "gpu": gpu,
        "free_disk_gib": round(free_gib, 1),
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
