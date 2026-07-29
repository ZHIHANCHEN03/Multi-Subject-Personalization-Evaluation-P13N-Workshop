"""Is the UMO-vs-one-shot near-tie an artifact of image sharpness?

UMO's runner (its own official config: image_guidance_scale 2.0, no max_pixels,
a longer negative prompt) produces systematically softer images than ours: on a
120-task sample UMO is softer than one_shot on 120/120 tasks, median Laplacian
variance 456 vs 878 (sign test p = 1.2e-12). Since SCR and DINO are both computed
from DINOv2 features, a reviewer can reasonably ask whether that softness -- not
identity behaviour -- is what the metrics are picking up.

This measures the sensitivity directly. Take one_shot images, blur them down to
UMO's sharpness level, re-score with the same detection-aware DINOv2 scorer, and
report how far SCR and DINO move. If they barely move, the reported near-tie is
not a blur artifact and the paper can say so with a number attached.

    python blur_counterfactual.py --data <manifest.jsonl> \
        --images <one_shot images dir> --out blur_cf.jsonl [--limit 120]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image, ImageFilter

ROUND1 = Path(os.environ.get(
    "MIDC_ROUND1", Path(__file__).resolve().parent.parent / "round1"))
sys.path.insert(0, str(ROUND1))

LAP = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)


def sharpness(im: Image.Image) -> float:
    """Laplacian variance -- the standard cheap focus measure."""
    a = np.asarray(im.convert("L"), dtype=float)
    return float((sliding_window_view(a, (3, 3)) * LAP).sum(axis=(-1, -2)).var())


def blur_to_target(im: Image.Image, target: float,
                   lo: float = 0.0, hi: float = 4.0, iters: int = 12) -> tuple[Image.Image, float]:
    """Bisect the Gaussian radius until sharpness matches `target`.

    Matching per image rather than applying one fixed radius keeps the comparison
    honest: every blurred image lands at UMO's measured sharpness, so the test is
    'same sharpness, original content'.
    """
    if sharpness(im) <= target:
        return im, 0.0
    best, best_r = im, 0.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        cand = im.filter(ImageFilter.GaussianBlur(radius=mid))
        s = sharpness(cand)
        best, best_r = cand, mid
        if s > target:
            lo = mid
        else:
            hi = mid
    return best, best_r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="task manifest JSONL")
    ap.add_argument("--images", required=True, help="dir of one_shot <task_id>.png")
    ap.add_argument("--umo_images", default=None,
                    help="dir of UMO images; if given, each task's blur target is "
                         "its own UMO sharpness instead of the global median")
    ap.add_argument("--target_sharpness", type=float, default=456.0,
                    help="fallback target (UMO median on the 120-task sample)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import common

    tasks = common.load_tasks(args.data, limit=None)
    by_id = {t.task_id: t for t in tasks}
    images = Path(args.images)
    umo_dir = Path(args.umo_images) if args.umo_images else None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    have = sorted(p for p in images.glob("*.png") if p.stem in by_id)
    if args.limit:
        have = have[:args.limit]
    print(f"[cf] {len(have)} one_shot images to test", file=sys.stderr)

    scorer = common.DinoScorer()
    print("[cf] DINOv2 + Grounding-DINO scorer ready", file=sys.stderr)

    t0 = time.time()
    n = 0
    with out_path.open("a", encoding="utf-8") as sink:
        for i, p in enumerate(have, 1):
            tid = p.stem
            task = by_id[tid]
            try:
                im = Image.open(p).convert("RGB")
                s_orig = sharpness(im)

                target = args.target_sharpness
                if umo_dir is not None:
                    up = umo_dir / f"{tid}.png"
                    if up.exists():
                        target = sharpness(Image.open(up).convert("RGB"))

                blurred, radius = blur_to_target(im, target)
                s_blur = sharpness(blurred)

                sims_o, _ = scorer.score_task(im, task)
                sims_b, _ = scorer.score_task(blurred, task)
                rec = {
                    "task_id": tid,
                    "sharp_orig": round(s_orig, 1),
                    "sharp_target": round(target, 1),
                    "sharp_blurred": round(s_blur, 1),
                    "blur_radius": round(radius, 3),
                    "scr_orig": common.scr_from_sims(sims_o, common.SCR_THRESH),
                    "scr_blurred": common.scr_from_sims(sims_b, common.SCR_THRESH),
                    "dino_orig": round(sum(sims_o) / len(sims_o), 5) if sims_o else None,
                    "dino_blurred": round(sum(sims_b) / len(sims_b), 5) if sims_b else None,
                }
                n += 1
            except Exception as exc:
                rec = {"task_id": tid, "error": f"{type(exc).__name__}: {exc}"}
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            if i % 20 == 0 or i == len(have):
                r = i / max(time.time() - t0, 1e-9)
                print(f"[cf] {i}/{len(have)} ok={n} {r:.2f}/s "
                      f"eta~{(len(have)-i)/max(r,1e-9)/60:.1f}min", file=sys.stderr)

    rows = [json.loads(l) for l in out_path.read_text().splitlines()
            if l.strip().startswith("{")]
    rows = [r for r in rows if "scr_orig" in r]
    if rows:
        ds = [r["scr_blurred"] - r["scr_orig"] for r in rows]
        dd = [r["dino_blurred"] - r["dino_orig"] for r in rows]
        print(f"\n[cf] n={len(rows)}", file=sys.stderr)
        print(f"[cf] sharpness {statistics.median(r['sharp_orig'] for r in rows):.0f}"
              f" -> {statistics.median(r['sharp_blurred'] for r in rows):.0f}",
              file=sys.stderr)
        print(f"[cf] SCR  {statistics.mean(r['scr_orig'] for r in rows):.4f}"
              f" -> {statistics.mean(r['scr_blurred'] for r in rows):.4f}"
              f"   mean delta {statistics.mean(ds):+.4f}", file=sys.stderr)
        print(f"[cf] DINO {statistics.mean(r['dino_orig'] for r in rows):.4f}"
              f" -> {statistics.mean(r['dino_blurred'] for r in rows):.4f}"
              f"   mean delta {statistics.mean(dd):+.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
