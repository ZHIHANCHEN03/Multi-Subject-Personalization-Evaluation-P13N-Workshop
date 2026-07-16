"""Pipeline 4: released UMO-OmniGen2 trained baseline (direct inference).

Loads the official UMO source and released LoRA from repository-local paths,
runs the same JSONL tasks, and applies the same independent SCR scorer.
Nothing is trained here.
"""
from __future__ import annotations

import argparse
import common
from external_generators import UMOGenerator


def make_method(generator: UMOGenerator):
    def method(task: common.Task):
        image = generator.generate(task, seed=0)
        return image, {
            "budget": 1,
            "gen_calls": 1,
            "source": "official_umo_omnigen2",
        }

    return method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", default="umo")
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--lora_path", default=None)
    ap.add_argument("--no_offload", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_scr", action="store_true")
    args = ap.parse_args()

    tasks = common.load_tasks(args.data, limit=args.limit)
    generator = UMOGenerator(
        model_path=args.model_path,
        lora_path=args.lora_path,
        cpu_offload=not args.no_offload,
    )
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
