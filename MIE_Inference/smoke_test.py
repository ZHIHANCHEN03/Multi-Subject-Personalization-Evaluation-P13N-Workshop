"""End-to-end check that a MIE checkpoint loads and produces sane scores.

Run this first on any new box or checkpoint. It fails loudly on the two things
that otherwise fail silently: weights that did not actually apply, and a model
that returns the same score no matter what it is shown.

    MIE_CKPT=/path/to/ckpt python smoke_test.py --refs /path/to/refs

Checks
------
1.  the checkpoint loads and every weight group verifiably applied
2.  scoring a real (references, candidate) pair returns finite numbers in range
3.  the three facet probabilities lie in [0, 1]
4.  the model *discriminates*: a matched candidate should not score identically
    to a deliberately mismatched one. A tie means the heads are not reading the
    images, which is the failure mode a plain "it loaded fine" check misses.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from mie_loader import DIMS, load_runtime, score


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="defaults to $MIE_CKPT")
    ap.add_argument("--refs", required=True,
                    help="directory of reference images (uses the first few)")
    ap.add_argument("--n_refs", type=int, default=2)
    args = ap.parse_args()

    refs_dir = Path(args.refs).expanduser().resolve()
    images = sorted(
        p for p in refs_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if len(images) < args.n_refs + 2:
        fail(f"need at least {args.n_refs + 2} images in {refs_dir}, found {len(images)}")

    print("[1/4] loading checkpoint ...")
    runtime = load_runtime(args.checkpoint)
    report = runtime["report"]
    print(json.dumps(report, indent=2))

    for key, label in (("backbone_tensors_applied", "backbone tensors"),
                       ("lora_params", "LoRA parameters")):
        if key in report and not report[key]:
            fail(f"{label} reported zero -- weights did not apply")
    if report.get("mode") in {"lora", "lora_layer"} and "lora_params" not in report:
        fail("mode claims LoRA but no adapter was attached")
    print("[1/4] OK -- all weight groups applied")

    refs = [str(p) for p in images[:args.n_refs]]
    matched = str(images[0])            # a reference itself: should look consistent
    mismatched = str(images[-1])        # an unrelated subject
    prompt = "a photo containing the referenced subjects together"

    print("[2/4] scoring a matched candidate ...")
    a = score(runtime, image_path=matched, ref_paths=refs, prompt=prompt)
    print("   ", {k: round(v, 4) for k, v in a.items()})
    for k, v in a.items():
        if not math.isfinite(v):
            fail(f"{k} is not finite ({v})")
    print("[2/4] OK -- finite scores")

    print("[3/4] checking facet probabilities are in [0, 1] ...")
    for dim in DIMS:
        if not 0.0 <= a[dim] <= 1.0:
            fail(f"{dim}={a[dim]} outside [0,1]; heads may be misaligned")
    print("[3/4] OK")

    print("[4/4] checking the model discriminates ...")
    b = score(runtime, image_path=mismatched, ref_paths=refs, prompt=prompt)
    print("   ", {k: round(v, 4) for k, v in b.items()})
    deltas = {k: abs(a[k] - b[k]) for k in a}
    if max(deltas.values()) < 1e-6:
        fail("matched and mismatched candidates scored identically -- "
             "the model is not conditioning on the image")
    print("    deltas:", {k: round(v, 5) for k, v in deltas.items()})
    print("[4/4] OK -- scores differ between candidates")

    print("\nSMOKE TEST PASSED")
    print(f"  checkpoint : {report['checkpoint']}")
    print(f"  base model : {report['base_model']}  (mode={report['mode']})")
    print(f"  device     : {report['device']}")


if __name__ == "__main__":
    main()
