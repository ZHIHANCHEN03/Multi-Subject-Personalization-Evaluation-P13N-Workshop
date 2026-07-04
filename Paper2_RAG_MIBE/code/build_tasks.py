"""Build MISC task JSONL from the MIB prompt bank + refs/ image folder.

The prompt bank (prompt/train_60k_v13_2.jsonl) lists, per prompt, the entities
used (`people_names` + `object_names`). Each entity name maps 1:1 to a single
reference image refs/<name>.jpg. This script joins the two into the schema that
data.py expects:

    {"task_id","prompt","subjects":[{"name","ref_images":[<abs path>]}], "meta":{...}}

Stratified sampling across (level x class_tag) gives a balanced eval set that
stresses Existence (many entities / occlusion) and Interaction (interaction tag).

Usage (from code/):
    python build_tasks.py --limit 200                 # 200 stratified tasks
    python build_tasks.py --limit 60 --levels 4 6 8   # only multi-subject
    python build_tasks.py --class_tags occlusion_interaction --limit 100
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS = REPO_ROOT / "prompt" / "train_60k_v13_2.jsonl"
DEFAULT_REFS = REPO_ROOT / "refs"
DEFAULT_OUT = REPO_ROOT / "Paper2_RAG_MIBE" / "data" / "tasks.jsonl"


def _ref_index(refs_dir: Path) -> dict:
    idx = {}
    for f in os.listdir(refs_dir):
        stem, ext = os.path.splitext(f)
        if ext.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            idx[stem] = str((refs_dir / f).resolve())
    return idx


def _human(name: str) -> str:
    """Entity id -> the surface form used in the prompt text (underscores->spaces)."""
    return name.replace("_", " ")


def build(args) -> None:
    refs = _ref_index(Path(args.refs))
    prompts_path = Path(args.prompts)
    levels = set(args.levels) if args.levels else None
    class_tags = set(args.class_tags) if args.class_tags else None

    # bucket rows by (level, class_tag) for stratified sampling
    buckets: dict = collections.defaultdict(list)
    kept = skipped_missing = 0
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if levels and r.get("level") not in levels:
                continue
            if class_tags and r.get("class_tag") not in class_tags:
                continue
            entities = list(r.get("people_names", [])) + list(r.get("object_names", []))
            if not entities:
                continue
            if any(e not in refs for e in entities):
                skipped_missing += 1
                continue
            buckets[(r.get("level"), r.get("class_tag"))].append(r)
            kept += 1

    rng = random.Random(args.seed)
    for k in buckets:
        rng.shuffle(buckets[k])

    # round-robin across buckets so the sample is balanced
    selected = []
    if args.limit and buckets:
        order = sorted(buckets.keys())
        i = 0
        while len(selected) < args.limit and any(buckets[k] for k in order):
            k = order[i % len(order)]
            if buckets[k]:
                selected.append(buckets[k].pop())
            i += 1
    else:
        for k in buckets:
            selected.extend(buckets[k])
        rng.shuffle(selected)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fo:
        for r in selected:
            entities = list(r.get("people_names", [])) + list(r.get("object_names", []))
            subjects = [{"name": _human(e), "ref_images": [refs[e]]} for e in entities]
            task = {
                "task_id": f"mib_{r.get('id')}",
                "prompt": r.get("prompt_en", ""),
                "subjects": subjects,
                "meta": {
                    "level": r.get("level"),
                    "class_tag": r.get("class_tag"),
                    "n_humans": r.get("n_humans"),
                    "n_objects": r.get("n_objects"),
                },
            }
            fo.write(json.dumps(task, ensure_ascii=False) + "\n")

    dist = collections.Counter((r.get("level"), r.get("class_tag")) for r in selected)
    print(f"eligible prompts: {kept}  (skipped missing-ref: {skipped_missing})")
    print(f"wrote {len(selected)} tasks -> {out_path}")
    print("stratum distribution (level, class_tag):")
    for k in sorted(dist):
        print(f"  {k}: {dist[k]}")


def parse_args():
    ap = argparse.ArgumentParser(description="Build MISC tasks.jsonl from MIB prompts + refs")
    ap.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    ap.add_argument("--refs", default=str(DEFAULT_REFS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=200, help="total tasks (stratified); 0 = all")
    ap.add_argument("--levels", type=int, nargs="*", default=None, help="filter by level(s), e.g. 4 6 8")
    ap.add_argument("--class_tags", nargs="*", default=None, help="filter by class_tag(s)")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


if __name__ == "__main__":
    build(parse_args())
