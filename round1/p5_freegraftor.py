"""Pipeline 5: official FreeGraftor training-free OPEN-LOOP baseline.

FreeGraftor is training-free but open-loop / single-pass (cross-image feature
grafting), with NO external verifier and NO iterative correction. It is the key
"training-free rival". This file directly imports the official repository-local
pipeline and released dependencies; it does not require precomputed images.
"""
from __future__ import annotations

import argparse
import common
from external_generators import FreeGraftorGenerator


def make_method(generator: FreeGraftorGenerator):
    def method(task: common.Task):
        image = generator.generate(task, seed=0)
        return image, {
            "budget": 1,
            "gen_calls": 1,
            "source": "official_freegraftor",
        }

    return method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", default="freegraftor")
    ap.add_argument("--no_offload", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_scr", action="store_true")
    args = ap.parse_args()

    tasks = common.load_tasks(args.data, limit=args.limit)
    generator = FreeGraftorGenerator(cpu_offload=not args.no_offload)
    scorer = None if args.no_scr else common.DinoScorer()
    common.run_over_dataset(
        args.name,
        make_method(generator),
        tasks,
        scorer,
        save_images=True,
        continue_on_error=True,
    )


if __name__ == "__main__":
    main()
