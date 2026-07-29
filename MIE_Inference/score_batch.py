"""Score a manifest of (references, candidate, prompt) triples with MIE.

Loads the model once and streams results, so a long run costs one model load.
Output is written incrementally and already-scored rows are skipped on restart,
which makes the script safe to interrupt and resume.

Manifest is JSONL, one task per line, in the format the MIDC pipelines already
use (see round2/results_r2/manifests/*.jsonl):

    {"task_id": "hard_050399",
     "prompt": "White studio. ...",
     "subjects": [{"name": "...", "ref_images": ["/abs/path.jpg"]}, ...]}

The candidate image for a task is looked up as <images>/<task_id>.png (override
the suffix with --image_suffix).

    MIE_CKPT=/path/to/ckpt python score_batch.py \
        --manifest /workspace/misc/round2/results_r2/manifests/round2_full.jsonl \
        --images   /workspace/misc/round2/.../images \
        --out      scores.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from mie_loader import DIMS, load_runtime, score


def read_manifest(path: Path) -> list[dict]:
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("{"):
            tasks.append(json.loads(line))
    return tasks


def already_done(out_path: Path) -> set[str]:
    """Task ids already present in a partial output file."""
    if not out_path.exists():
        return set()
    done = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            done.add(json.loads(line)["task_id"])
        except (json.JSONDecodeError, KeyError):
            continue  # truncated final line from an interrupted run
    return done


def ref_paths_for(task: dict, refs_root: str | None) -> list[str]:
    paths = []
    for subject in task.get("subjects", []):
        for ref in subject.get("ref_images", []):
            p = Path(ref)
            # Manifests carry absolute paths from the box that built them; allow
            # re-rooting when scoring on a different machine.
            if refs_root and not p.exists():
                p = Path(refs_root) / p.name
            paths.append(str(p))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--images", required=True, help="dir of <task_id><suffix> candidates")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=None, help="defaults to $MIE_CKPT")
    ap.add_argument("--refs_root", default=None,
                    help="re-root manifest ref paths onto this dir if missing")
    ap.add_argument("--image_suffix", default=".png")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    manifest = Path(args.manifest).expanduser().resolve()
    images = Path(args.images).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    tasks = read_manifest(manifest)
    if args.limit:
        tasks = tasks[:args.limit]
    done = already_done(out_path)
    todo = [t for t in tasks if t["task_id"] not in done]
    print(f"manifest {len(tasks)} tasks | already scored {len(done)} | to score {len(todo)}",
          file=sys.stderr)
    if not todo:
        print("nothing to do", file=sys.stderr)
        return

    runtime = load_runtime(args.checkpoint)
    print(json.dumps(runtime["report"]), file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = failed = 0
    started = time.time()
    with out_path.open("a", encoding="utf-8") as sink:
        for i, task in enumerate(todo, 1):
            tid = task["task_id"]
            image = images / f"{tid}{args.image_suffix}"
            record: dict = {"task_id": tid}
            try:
                if not image.exists():
                    raise FileNotFoundError(f"no candidate image at {image}")
                refs = ref_paths_for(task, args.refs_root)
                if not refs:
                    raise ValueError("task has no reference images")
                record.update(score(runtime, str(image), refs, task["prompt"]))
                record["n_refs"] = len(refs)
                ok += 1
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                failed += 1
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()  # so an interrupted run stays resumable
            if i % 25 == 0 or i == len(todo):
                rate = i / max(time.time() - started, 1e-6)
                print(f"  {i}/{len(todo)}  ok={ok} failed={failed}  {rate:.2f} task/s",
                      file=sys.stderr)

    print(f"done: {ok} scored, {failed} failed -> {out_path}", file=sys.stderr)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
