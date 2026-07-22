"""Score a folder of pre-generated images with MIE (E/A/I + total + standardized deficits).

This complements round2/score_precomputed.py (which only does SCR/DINO).
MIE scores are needed for the paper's narrative: baselines like one_shot / umo
do not have MIE scores in their records (only ours_v2 stores them in-loop), so
we recompute MIE on their final images to fill the table.

This does NOT train anything and does NOT use MIE to optimize — it only scores
final images for reporting. The independent judge remains SCR/DINO.

Usage (in the omni venv, which has the MIE subprocess client):
  MIE_CKPT=<ckpt> python score_mie_precomputed.py \
      --data <manifest.jsonl> --images <dir of task_id.png> --name one_shot \
      --calibration <mie_baselines.json> --out_dir <results_dir>

Output: writes records.jsonl with MIE total/existence/appearance/interaction
and standardized deficits (median_d - score_d) / scale_d per entity count, so
the paper can report "baselines have X% worse deficit on dimension D".
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

# reuse round1 core (Task loader + MIE critic)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "round1"))
import common  # noqa: E402
from PIL import Image  # noqa: E402


def find_image(images_dir, task_id):
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = os.path.join(images_dir, task_id + ext)
        if os.path.exists(p):
            return p
    return None


def load_calibration(path: str) -> dict:
    """Load frozen mie_baselines.json: {<n_entities>: {dim: {median, scale, ...}}}."""
    with open(path, "r") as f:
        return json.load(f)


def standardized_deficits(scores: dict, n: int, calibration: dict) -> dict:
    """(median_d - score_d) / scale_d for each dimension. Higher = worse deficit."""
    group = calibration.get(str(n)) or calibration.get(n) or {}
    out = {}
    for dim in common.DIMS:
        stats = group.get(dim, {})
        median = stats.get("median", 0.5)
        scale = stats.get("scale", 0.1)
        out[dim] = (median - scores[dim]) / scale if scale > 0 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="manifest jsonl (same as the run)")
    ap.add_argument("--images", required=True, help="dir of <task_id>.png from the baseline")
    ap.add_argument("--name", required=True, help="method name, e.g. one_shot / umo")
    ap.add_argument("--calibration", required=True, help="frozen mie_baselines.json")
    ap.add_argument("--out_dir", default=None, help="results dir (default: round1/results)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.out_dir:
        os.environ["ROUND1_WORK"] = args.out_dir
    import importlib
    importlib.reload(common)

    tasks = common.load_tasks(args.data, limit=args.limit)
    critic = common.build_critic()
    calibration = load_calibration(args.calibration)

    out_dir = common.WORK_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / "records_mie.jsonl"

    started_count = 0
    with open(rec_path, "w", encoding="utf-8") as fout:
        for i, task in enumerate(tasks):
            image_path = find_image(args.images, task.task_id)
            if image_path is None:
                print(f"[MIE-score] SKIP {task.task_id}: no image found in {args.images}")
                continue
            with Image.open(image_path) as loaded:
                scores = critic.score(loaded.convert("RGB"), task)
            deficits = standardized_deficits(scores, task.num_subjects, calibration)
            rec = {
                "task_id": task.task_id,
                "method": args.name,
                "num_subjects": task.num_subjects,
                "meta": task.meta,
                "mie_total": scores["total"],
                "mie_existence": scores["existence"],
                "mie_appearance": scores["appearance"],
                "mie_interaction": scores["interaction"],
                "deficits": deficits,
                "worst_dim": max(deficits, key=deficits.get),
                "image_path": image_path,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            started_count += 1
            print(
                f"[MIE-score] {started_count:03d}/{len(tasks):03d} task={task.task_id} "
                f"n={task.num_subjects} total={scores['total']:.3f} "
                f"E={scores['existence']:.3f} A={scores['appearance']:.3f} "
                f"I={scores['interaction']:.3f} worst={rec['worst_dim']}",
                flush=True,
            )

    print(f"[MIE-score] done -> {rec_path} ({started_count} records)")
    print(f"[MIE-score] merge with SCR records via task_id for the full table.")


if __name__ == "__main__":
    main()
