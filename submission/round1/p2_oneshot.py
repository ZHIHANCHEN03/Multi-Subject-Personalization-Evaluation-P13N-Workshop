"""Pipeline 2: one-shot baseline. Single generation, no verifier, no correction.

Run:
  python p2_oneshot.py --data hard_cases.jsonl --generator omnigen2
"""
from __future__ import annotations

import argparse

import common
from generators import build_generator


def make_method(generator, seed_offset: int = 0):
    def method(task: common.Task):
        img = generator.generate(common.omnigen_prompt(task), task.load_refs(), seed=seed_offset)
        return img, {"budget": 1, "gen_calls": 1, "seed_offset": seed_offset}

    return method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", default="one_shot")
    ap.add_argument("--generator", default="omnigen2")
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_scr", action="store_true")
    args = ap.parse_args()

    tasks = common.load_tasks(args.data, limit=args.limit)
    generator = build_generator(args.generator)
    scorer = None if args.no_scr else common.DinoScorer()
    common.run_over_dataset(args.name, make_method(generator, args.seed_offset), tasks, scorer)


if __name__ == "__main__":
    main()
