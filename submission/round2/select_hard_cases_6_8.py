"""Select 6/8-entity hard cases for the FLUX.2 scaling experiment (P1-3).

This is a thin wrapper around round1/select_hard_cases.py that builds TWO
manifests (6-entity + 8-entity) for the FLUX.2 scaling story:
  "the more subjects, the worse the collapse, the larger our gain."

The manifests are disjoint from:
  - round1 calibration tasks (seed 0/1)
  - round2 test tasks (seed 200/201)
  - FLUX.2 calibration tasks (seed 777/778)
We use seed 300/301 here.

Usage:
  python select_hard_cases_6_8.py --src <train_60k_v13_2.jsonl> --refs <refs/> \
      --n6 100 --n8 100 --out_dir <manifests_dir>
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROUND1 = Path(__file__).resolve().parent.parent / "round1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="train_60k_v13_2.jsonl")
    ap.add_argument("--refs", required=True, help="refs/ directory")
    ap.add_argument("--n6", type=int, default=100, help="number of 6-entity tasks")
    ap.add_argument("--n8", type=int, default=100, help="number of 8-entity tasks")
    ap.add_argument("--out_dir", required=True, help="output manifests dir")
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="JSONL manifest(s) whose task_ids must not be selected",
    )
    ap.add_argument("--seed6", type=int, default=300)
    ap.add_argument("--seed8", type=int, default=301)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    exclude_args = []
    for ex in args.exclude:
        exclude_args += ["--exclude", ex]

    out6 = os.path.join(args.out_dir, "scaling_6.jsonl")
    out8 = os.path.join(args.out_dir, "scaling_8.jsonl")
    combined = os.path.join(args.out_dir, "scaling_6_8.jsonl")

    # 6-entity
    cmd6 = [
        sys.executable, str(ROUND1 / "select_hard_cases.py"),
        "--src", args.src, "--refs", args.refs,
        "--exact_entities", "6", "--n", str(args.n6),
        "--seed", str(args.seed6), "--out", out6,
    ] + exclude_args
    print("[select-6-8] " + " ".join(cmd6))
    subprocess.run(cmd6, check=True)

    # 8-entity
    cmd8 = [
        sys.executable, str(ROUND1 / "select_hard_cases.py"),
        "--src", args.src, "--refs", args.refs,
        "--exact_entities", "8", "--n", str(args.n8),
        "--seed", str(args.seed8), "--out", out8,
    ] + exclude_args
    print("[select-6-8] " + " ".join(cmd8))
    subprocess.run(cmd8, check=True)

    # combined
    with open(combined, "w") as fout:
        for p in (out6, out8):
            with open(p) as fin:
                fout.writelines(fin.readlines())
    n_total = sum(1 for _ in open(combined) if _.strip())
    print(f"[select-6-8] combined -> {combined} ({n_total} tasks)")


if __name__ == "__main__":
    main()
