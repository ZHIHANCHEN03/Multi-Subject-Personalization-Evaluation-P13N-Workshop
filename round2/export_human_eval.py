"""Export blind A/B pairs for human evaluation.

For each sampled task, build a side-by-side image [left | right] where one side
is OURS and the other is a comparison method (UMO / best_of_n / ...), with the
left/right order RANDOMIZED and hidden. Writes:
  - pairs/<pair_id>.png            side-by-side blind image
  - pairs/<pair_id>_refs.png       the reference subjects (context for judging identity)
  - ballot.csv                     one row per pair for annotators to fill "A" or "B"
  - key.json                       hidden mapping pair_id -> which side is ours (for aggregation)

Annotator instruction (put in your form): "Which image better preserves the
identity of ALL referenced subjects while following the prompt? Choose Left or
Right." 3 annotators per pair recommended.

Usage:
  python export_human_eval.py --results /workspace/misc/round1/results \
      --main ours_v2 --vs umo best_of_n --entities 4 --per 60 --out round2/human_eval
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random

from PIL import Image, ImageDraw


def load(results_dir, method):
    path = os.path.join(results_dir, method, "records.jsonl")
    out = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["task_id"]] = r
    return out


def side_by_side(imgA, imgB, size=512, gap=16):
    a = imgA.convert("RGB").resize((size, size))
    b = imgB.convert("RGB").resize((size, size))
    canvas = Image.new("RGB", (size * 2 + gap, size + 40), (255, 255, 255))
    canvas.paste(a, (0, 40))
    canvas.paste(b, (size + gap, 40))
    d = ImageDraw.Draw(canvas)
    d.text((size // 2 - 20, 12), "LEFT", fill=(0, 0, 0))
    d.text((size + gap + size // 2 - 24, 12), "RIGHT", fill=(0, 0, 0))
    return canvas


def ref_strip(ref_paths, size=192, gap=8):
    imgs = []
    for p in ref_paths:
        if os.path.exists(p):
            imgs.append(Image.open(p).convert("RGB").resize((size, size)))
    if not imgs:
        return None
    canvas = Image.new("RGB", (size * len(imgs) + gap * (len(imgs) - 1), size), (255, 255, 255))
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += size + gap
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--main", default="ours_v2")
    ap.add_argument("--vs", nargs="+", default=["umo", "best_of_n"])
    ap.add_argument("--entities", nargs="+", type=int, default=[4])
    ap.add_argument("--per", type=int, default=60, help="pairs per comparison method")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="human_eval")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    main = load(args.results, args.main)
    pairs_dir = os.path.join(args.out, "pairs")
    os.makedirs(pairs_dir, exist_ok=True)

    ballot_rows = []
    key = {}
    for other in args.vs:
        od = load(args.results, other)
        keys = [k for k in main if od.get(k) and main[k].get("num_subjects") in args.entities
                and main[k].get("image_path") and od[k].get("image_path")]
        rng.shuffle(keys)
        keys = keys[: args.per]
        for k in keys:
            pid = f"{args.main}_vs_{other}__{k}"
            try:
                a_img = Image.open(main[k]["image_path"])
                b_img = Image.open(od[k]["image_path"])
            except Exception as exc:
                print(f"skip {pid}: {exc}")
                continue
            ours_on_left = rng.random() < 0.5
            left, right = (a_img, b_img) if ours_on_left else (b_img, a_img)
            side_by_side(left, right).save(os.path.join(pairs_dir, pid + ".png"))
            refs = main[k].get("meta", {}).get("ref_paths") or []
            # fall back: reconstruct from records if ref list absent
            strip = ref_strip(refs) if refs else None
            if strip is not None:
                strip.save(os.path.join(pairs_dir, pid + "_refs.png"))
            key[pid] = {"task_id": k, "main": args.main, "other": other,
                        "ours_side": "LEFT" if ours_on_left else "RIGHT"}
            ballot_rows.append({"pair_id": pid, "annotator_choice(LEFT/RIGHT)": ""})

    with open(os.path.join(args.out, "ballot.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "annotator_choice(LEFT/RIGHT)"])
        w.writeheader()
        w.writerows(ballot_rows)
    with open(os.path.join(args.out, "key.json"), "w") as f:
        json.dump(key, f, indent=2)
    print(f"exported {len(ballot_rows)} blind pairs -> {pairs_dir}")
    print(f"ballot: {os.path.join(args.out, 'ballot.csv')}  key: {os.path.join(args.out, 'key.json')}")
    print("Give each pair to >=3 annotators (copy ballot.csv per annotator).")


if __name__ == "__main__":
    main()
