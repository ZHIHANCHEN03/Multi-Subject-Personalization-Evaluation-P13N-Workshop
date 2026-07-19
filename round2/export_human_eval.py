"""Export blind A/B pairs for human evaluation (Round 2, seed 0).

Reads the sharded Round-2 layout (results_r2/shard_{0..3}/<method>_s0/) and
builds a self-contained human-eval package:

  <out>/
    pairs/<pair_id>.png        composite side-by-side (LEFT | RIGHT), blinded
    pairs/<pair_id>_refs.png   reference subjects strip (context for identity)
    manifest.js                window.MANIFEST = [...]  (loaded by index.html)
    key.json                   hidden pair_id -> ours_side / comparison / task_id
    sample.jsonl               fixed sampled task_ids (anti-p-hacking)
    ballot.csv                 (legacy) blank ballot for CSV-based labeling

Sampling is FIXED by --seed (default 0) and restricted to num_subjects in
--entities (default 4 = hard_4, the pre-declared main slice). Comparisons
default to ours_v2 vs {umo, best_of_n}. Per-comparison sample size = --per.

The frontend (index.html) loads manifest.js, shows each pair with its refs +
prompt, asks two forced-binary questions (identity / overall), and exports a
votes_<labeler>.json. aggregate_human_eval.py combines the 3 labelers' JSONs
with key.json into win-rate + bootstrap CI + Fleiss' kappa.

Usage (run on the server, from /workspace/misc/round2):
  python export_human_eval.py \
      --results results_r2 --shards 0 1 2 3 \
      --main ours_v2_s0 --vs umo_s0 best_of_n_s0 \
      --entities 4 --per 100 --seed 0 --out human_eval
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import random

from PIL import Image, ImageDraw


def load_shards(results_dir, shards, method):
    """Return task_id -> record dict, merged across shards for one method."""
    out = {}
    for i in shards:
        p = os.path.join(results_dir, f"shard_{i}", method, "records.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["task_id"]] = r
    return out


def load_manifest_meta(results_dir, shards):
    """Return task_id -> {refs:[...], prompt:str} from the shard manifests."""
    out = {}
    for i in shards:
        p = os.path.join(results_dir, "shards", f"shard_{i}.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tid = r["task_id"]
            paths = []
            for s in r.get("subjects", []):
                paths.extend(s.get("ref_images", []))
            out[tid] = {"refs": paths, "prompt": r.get("prompt", "")}
    return out


def side_by_side(imgA, imgB, size=512, gap=16):
    a = imgA.convert("RGB").resize((size, size))
    b = imgB.convert("RGB").resize((size, size))
    canvas = Image.new("RGB", (size * 2 + gap, size + 40), (255, 255, 255))
    canvas.paste(a, (0, 40))
    canvas.paste(b, (size + gap, 40))
    d = ImageDraw.Draw(canvas)
    d.text((size // 2 - 20, 12), "LEFT", fill=(0, 0, 0))
    d.text((size + gap + size // 2 - 28, 12), "RIGHT", fill=(0, 0, 0))
    return canvas


def ref_strip(ref_paths, size=192, gap=8):
    imgs = []
    for p in ref_paths:
        if os.path.exists(p):
            try:
                imgs.append(Image.open(p).convert("RGB").resize((size, size)))
            except Exception:
                pass
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
    ap.add_argument("--results", required=True, help="results_r2 dir")
    ap.add_argument("--shards", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--main", default="ours_v2_s0")
    ap.add_argument("--vs", nargs="+", default=["umo_s0", "best_of_n_s0"])
    ap.add_argument("--entities", nargs="+", type=int, default=[4],
                    help="num_subjects to keep (4=hard_4, 2=easy_2)")
    ap.add_argument("--per", type=int, default=100, help="pairs per comparison")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for sampling")
    ap.add_argument("--out", default="human_eval")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    main = load_shards(args.results, args.shards, args.main)
    meta = load_manifest_meta(args.results, args.shards)

    pairs_dir = os.path.join(args.out, "pairs")
    os.makedirs(pairs_dir, exist_ok=True)

    manifest = []
    key = {}
    sample_rows = []
    ballot_rows = []

    for other in args.vs:
        od = load_shards(args.results, args.shards, other)
        cand = [k for k in main
                if od.get(k)
                and main[k].get("num_subjects") in args.entities
                and main[k].get("image_path") and os.path.exists(main[k]["image_path"])
                and od[k].get("image_path") and os.path.exists(od[k]["image_path"])]
        rng.shuffle(cand)
        cand = cand[: args.per]
        for k in cand:
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
            rpaths = meta.get(k, {}).get("refs", [])
            strip = ref_strip(rpaths)
            if strip is not None:
                strip.save(os.path.join(pairs_dir, pid + "_refs.png"))
            ours_side = "LEFT" if ours_on_left else "RIGHT"
            key[pid] = {"task_id": k, "main": args.main, "other": other,
                        "ours_side": ours_side}
            manifest.append({
                "pair_id": pid,
                "comparison": other,
                "prompt": meta.get(k, {}).get("prompt", ""),
                "num_subjects": main[k].get("num_subjects"),
                "image": f"pairs/{pid}.png",
                "refs": f"pairs/{pid}_refs.png" if strip is not None else None,
            })
            sample_rows.append({"task_id": k, "comparison": other, "pair_id": pid})
            ballot_rows.append({"pair_id": pid, "annotator_choice(LEFT/RIGHT)": ""})

    # manifest.js (loaded by index.html via <script src>; works offline from file://)
    with open(os.path.join(args.out, "manifest.js"), "w") as f:
        f.write("window.MANIFEST = ")
        json.dump(manifest, f, indent=2)
        f.write(";\n")
    # key.json (hidden; used by aggregate_human_eval.py to unblind)
    with open(os.path.join(args.out, "key.json"), "w") as f:
        json.dump(key, f, indent=2)
    # sample.jsonl (fixed sampled task_ids; attach to supplementary for anti-p-hacking)
    with open(os.path.join(args.out, "sample.jsonl"), "w") as f:
        for r in sample_rows:
            f.write(json.dumps(r) + "\n")
    # legacy blank ballot (for CSV-based labeling if no browser)
    with open(os.path.join(args.out, "ballot.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "annotator_choice(LEFT/RIGHT)"])
        w.writeheader()
        w.writerows(ballot_rows)

    n = len(manifest)
    print(f"exported {n} blind pairs -> {pairs_dir}")
    print(f"  manifest.js  : {os.path.join(args.out, 'manifest.js')}")
    print(f"  key.json     : {os.path.join(args.out, 'key.json')}  (keep hidden from labelers)")
    print(f"  sample.jsonl : {os.path.join(args.out, 'sample.jsonl')}")
    print(f"  ballot.csv   : {os.path.join(args.out, 'ballot.csv')}  (legacy)")
    print(f"comparisons: {args.vs}, per={args.per}, entities={args.entities}, seed={args.seed}")
    print("Next: zip <out> + index.html + aggregate_human_eval.py, send to 3 labelers.")


if __name__ == "__main__":
    main()
