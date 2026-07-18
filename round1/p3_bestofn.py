"""Pipeline 3: best-of-N. Generate N candidates, keep the one with the highest
MIE preference (total) score. This is the strong "selection" baseline; the axis
vs OURS is selection-vs-correction at equal budget B=N.

Run:
  python p3_bestofn.py --data hard_cases.jsonl --generator omnigen2 --budget 8
"""
from __future__ import annotations

import argparse

import common
from generators import build_generator


def make_method(critic, generator, budget: int, seed_offset: int = 0):
    def method(task: common.Task):
        refs = task.load_refs()
        prompt = common.omnigen_prompt(task)
        best_img, best_score = None, None
        for s in range(budget):
            print(
                f"[BEST-OF-N][{task.task_id}] candidate={s+1}/{budget} "
                "action=generate_and_score",
                flush=True,
            )
            img = generator.generate(prompt, refs, seed=seed_offset + s)
            sc = critic.score(img, task)
            print(
                f"[BEST-OF-N][{task.task_id}] candidate={s+1}/{budget} "
                f"MIE_total={sc['total']:.4f}",
                flush=True,
            )
            if best_score is None or sc["total"] > best_score["total"]:
                best_img, best_score = img, sc
        info = {
            "budget": budget,
            "gen_calls": budget,
            "seed_offset": seed_offset,
            "final_total": best_score["total"],
            "final_dims": {d: best_score[d] for d in common.DIMS},
        }
        return best_img, info

    return method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", default="best_of_n")
    ap.add_argument("--generator", default="omnigen2")
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_scr", action="store_true")
    args = ap.parse_args()

    tasks = common.load_tasks(args.data, limit=args.limit)
    critic = common.build_critic()
    generator = build_generator(args.generator)
    scorer = None if args.no_scr else common.DinoScorer()
    common.run_over_dataset(
        args.name,
        make_method(critic, generator, args.budget, args.seed_offset),
        tasks,
        scorer,
    )


if __name__ == "__main__":
    main()
