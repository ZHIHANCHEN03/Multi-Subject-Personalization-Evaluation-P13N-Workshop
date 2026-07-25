"""Score a folder of pre-generated images with the SAME SCR(DINOv2) metric.

For cross-system baselines (MOSAIC, MultiCrafter, XVerse, ... — each on its own
base, run in its own repo) we do NOT reimplement them. Run their official repo
on our manifest, save one image per task as <images_dir>/<task_id>.png, then this
scores them into round1/round2 results format for apples-to-apples SCR/DINO
comparison. Nothing is trained here.

Usage (in the omni venv, which has torch + our common.py):
  python score_precomputed.py --data <hard_cases.jsonl> \
      --images /path/to/mosaic_images --name mosaic --out_dir <results_dir>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# reuse round1 core (Task loader + detection-aware SCR)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "round1"))
import common  # noqa: E402
from PIL import Image  # noqa: E402


def find_image(images_dir, task_id):
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = os.path.join(images_dir, task_id + ext)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="manifest jsonl (same as the run)")
    ap.add_argument("--images", required=True, help="dir of <task_id>.png from the baseline's repo")
    ap.add_argument("--name", required=True, help="method name, e.g. mosaic")
    ap.add_argument("--out_dir", default=None, help="results dir (default: round1/results)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.out_dir:
        os.environ["ROUND1_WORK"] = args.out_dir
    # re-import WORK_DIR after env set
    import importlib
    importlib.reload(common)

    tasks = common.load_tasks(args.data, limit=args.limit)
    scorer = common.DinoScorer()

    def method(task: common.Task):
        p = find_image(args.images, task.task_id)
        if p is None:
            raise FileNotFoundError(
                f"[{args.name}] no image for {task.task_id} in {args.images}; "
                "run the baseline's official repo on this manifest first."
            )
        return Image.open(p).convert("RGB"), {"budget": 0, "source": f"{args.name}_precomputed"}

    common.run_over_dataset(args.name, method, tasks, scorer,
                            save_images=False, continue_on_error=True)
    print(f"[{args.name}] scored -> {common.WORK_DIR / args.name / 'records.jsonl'}")


if __name__ == "__main__":
    main()
