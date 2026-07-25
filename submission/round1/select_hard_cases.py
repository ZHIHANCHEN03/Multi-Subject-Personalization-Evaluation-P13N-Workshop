"""Select the Round-1 hard-case subset from the MIBE data_v2 prompt pool.

Hardest cases for identity collapse = strong interaction + occlusion + many
entities. In data_v2 that is:
    class_tag == "occlusion_interaction"   AND   total_entities >= 6   (i.e. n>4)

Each selected task resolves its entity names to reference image paths under
`refs/<name>.jpg`, and is written to a JSONL our pipelines consume.

Usage:
    python select_hard_cases.py \
        --src   /abs/.../data_v2/prompt/train_60k_v13_2.jsonl \
        --refs  /abs/.../data_v2/refs \
        --n 60 --out hard_cases.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random


def resolve_ref(refs_dir: str, name: str) -> str | None:
    for ext in (".jpg", ".png", ".jpeg", ".webp"):
        p = os.path.join(refs_dir, name + ext)
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="train_60k_v13_2.jsonl")
    ap.add_argument("--refs", required=True, help="refs/ directory")
    ap.add_argument("--class_tag", default="occlusion_interaction",
                    help="hardest interaction class")
    ap.add_argument("--min_entities", type=int, default=6, help="n>4 -> >=6")
    ap.add_argument("--max_entities", type=int, default=None)
    ap.add_argument("--exact_entities", type=int, default=None,
                    help="override min/max and keep exactly this entity count")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="JSONL manifest(s) whose task_ids must not be selected",
    )
    ap.add_argument("--out", default="hard_cases.jsonl")
    args = ap.parse_args()

    excluded_ids = set()
    for exclude_path in args.exclude:
        if not os.path.exists(exclude_path):
            continue
        with open(exclude_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    excluded_ids.add(str(json.loads(line)["task_id"]))

    pool = []
    with open(args.src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("class_tag") != args.class_tag:
                continue
            if f"hard_{r['id']:06d}" in excluded_ids:
                continue
            n_entities = r.get("total_entities", 0)
            if args.exact_entities is not None and n_entities != args.exact_entities:
                continue
            if args.exact_entities is None and n_entities < args.min_entities:
                continue
            if args.exact_entities is None and args.max_entities is not None and n_entities > args.max_entities:
                continue
            pool.append(r)

    entity_filter = (
        f"entities={args.exact_entities}"
        if args.exact_entities is not None
        else f"entities={args.min_entities}..{args.max_entities or 'inf'}"
    )
    print(f"[select] {len(pool)} candidates match "
          f"class_tag={args.class_tag} {entity_filter}")

    random.Random(args.seed).shuffle(pool)

    written, skipped = 0, 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for r in pool:
            names = list(r.get("people_names", [])) + list(r.get("object_names", []))
            ref_paths, ok = [], True
            for name in names:
                p = resolve_ref(args.refs, name)
                if p is None:
                    ok = False
                    break
                ref_paths.append(p)
            if not ok:
                skipped += 1
                continue

            task = {
                "task_id": f"hard_{r['id']:06d}",
                "prompt": r["prompt_en"],
                "subjects": [
                    {"name": name, "ref_images": [ref_paths[i]]}
                    for i, name in enumerate(names)
                ],
                "meta": {
                    "class_tag": r.get("class_tag"),
                    "n_humans": r.get("n_humans"),
                    "n_objects": r.get("n_objects"),
                    "total_entities": r.get("total_entities"),
                    "level": r.get("level"),
                    "hard": True,
                },
            }
            fout.write(json.dumps(task, ensure_ascii=False) + "\n")
            written += 1
            if written >= args.n:
                break

    print(f"[select] wrote {written} tasks -> {args.out} (skipped {skipped} missing-ref)")


if __name__ == "__main__":
    main()
