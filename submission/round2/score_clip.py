"""Score generated images with CLIP-T and CLIP-I.

Adds two things the paper is currently missing.

CLIP-T (image vs the *requested* prompt) answers the obvious objection to MIDC:
it rewrites prompts, front-duplicates references and injects layout hints, so does
it buy subject presence by drifting away from what was asked? Note that the prompt
used here is always the original manifest prompt, never MIDC's rewritten one --
otherwise the metric would reward the rewrite instead of testing it.

CLIP-I (image vs each reference) is a second identity signal that does not come
from the DINOv2 pipeline. SCR and DINO are both derived from the same per-subject
DINOv2 similarity array (SCR is that array thresholded at 0.5), so they are one
measurement reported two ways, correlated at r = -0.81 across our records. CLIP-I
is genuinely independent.

    python score_clip.py --manifest <manifest.jsonl> --images <dir> \
        --out clip_scores.jsonl [--refs_root <dir>]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

MODEL = os.environ.get("CLIP_MODEL", "openai/clip-vit-large-patch14")


def read_manifest(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("{"):
            r = json.loads(line)
            out[r["task_id"]] = r
    return out


def done_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    seen = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                seen.add(json.loads(line)["task_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--images", required=True, help="dir of <task_id>.png candidates")
    ap.add_argument("--out", required=True)
    ap.add_argument("--refs_root", default=None,
                    help="re-root manifest ref paths here when they do not resolve")
    ap.add_argument("--image_suffix", default=".png")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(MODEL)
    print(f"[clip] {MODEL} on {dev}", file=sys.stderr)

    tasks = read_manifest(Path(args.manifest))
    images = Path(args.images)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    already = done_ids(out_path)

    todo = [t for t in sorted(tasks) if t not in already]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[clip] manifest {len(tasks)} | done {len(already)} | todo {len(todo)}",
          file=sys.stderr)

    # Reference embeddings are reused across every task that cites the same subject,
    # and the reference pool is tiny (tens of images) next to thousands of candidates.
    ref_cache: dict[str, "torch.Tensor"] = {}

    def embed_image(path: str) -> "torch.Tensor":
        im = Image.open(path).convert("RGB")
        with torch.no_grad():
            px = proc(images=im, return_tensors="pt").to(dev)
            e = model.get_image_features(**px)
        return e / e.norm(dim=-1, keepdim=True)

    def embed_text(text: str) -> "torch.Tensor":
        with torch.no_grad():
            tk = proc(text=[text], return_tensors="pt", padding=True,
                      truncation=True, max_length=77).to(dev)
            e = model.get_text_features(**tk)
        return e / e.norm(dim=-1, keepdim=True)

    ok = failed = 0
    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as sink:
        for i, tid in enumerate(todo, 1):
            row = {"task_id": tid}
            try:
                img_path = images / f"{tid}{args.image_suffix}"
                if not img_path.exists():
                    raise FileNotFoundError(f"no image at {img_path}")
                task = tasks[tid]
                img_e = embed_image(str(img_path))

                # CLIP-T against the prompt as originally requested.
                row["clip_t"] = float((img_e @ embed_text(task["prompt"]).T).item())

                sims = []
                for subj in task.get("subjects", []):
                    for ref in subj.get("ref_images", []):
                        p = Path(ref)
                        if not p.exists() and args.refs_root:
                            p = Path(args.refs_root) / p.name
                        if not p.exists():
                            continue
                        k = str(p)
                        if k not in ref_cache:
                            ref_cache[k] = embed_image(k)
                        sims.append(float((img_e @ ref_cache[k].T).item()))
                if not sims:
                    raise ValueError("no reference images resolved")
                row["clip_i_per_subject"] = [round(s, 5) for s in sims]
                row["clip_i"] = round(sum(sims) / len(sims), 5)
                row["n_refs"] = len(sims)
                ok += 1
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                failed += 1
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            if i % 100 == 0 or i == len(todo):
                r = i / max(time.time() - t0, 1e-9)
                print(f"[clip] {i}/{len(todo)} ok={ok} failed={failed} "
                      f"{r:.1f}/s eta~{(len(todo)-i)/max(r,1e-9)/60:.1f}min",
                      file=sys.stderr)
    print(f"[clip] done: {ok} scored, {failed} failed -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
