"""Run ONE configuration over the dataset and dump per-task records.

Example (CPU smoke, no GPU/weights/data/MIE/LLM needed):
    python run.py --name smoke --method misc \
        --generator mock --critic mock --limit 4

Example (real, A100/H100):
    FLUX2_MODEL_ID=black-forest-labs/FLUX.2-klein-4B \
    MISC_CRITIC=mie_checkpoint MIE_ADAPTER=my_mie.adapter \
    MISC_DATA=/path/mib_gold.jsonl \
    python run.py --name misc_main --method misc --generator flux2 --critic mie_checkpoint
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import config
import data
import metrics
from critic import MIECritic
from generator import build_generator
from pipeline import MISCPipeline


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run name (output subdir)")
    ap.add_argument("--method", default="misc",
                    choices=["misc", "best_of_n", "one_shot", "caption_upsample"])
    ap.add_argument("--generator", default="flux2", choices=["flux2", "mock"])
    ap.add_argument("--critic", default=config.CRITIC_BACKEND,
                    choices=["mie_checkpoint", "vlm_judge", "mock"])
    ap.add_argument("--routing", default=config.ROUTING_DEFAULT, choices=config.ROUTING_MODES)
    ap.add_argument("--action", default=config.ACTION_MODE_DEFAULT, choices=config.ACTION_MODES)
    ap.add_argument("--seed_mode", default=config.SEED_MODE_DEFAULT, choices=config.SEED_MODES)
    ap.add_argument("--n_init", type=int, default=config.N_INIT)
    ap.add_argument("--k_steps", type=int, default=config.K_STEPS)
    ap.add_argument("--budget", type=int, default=None, help="aligned generation budget B")
    ap.add_argument("--data", default=None, help="path to MIB-Gold jsonl (else mock/env)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_metrics", action="store_true", help="skip independent metrics")
    ap.add_argument("--save_images", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = config.WORK_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    if args.save_images:
        img_dir.mkdir(exist_ok=True)

    tasks = data.load_dataset(args.data, limit=args.limit)
    critic = MIECritic(backend=args.critic)
    generator = build_generator(args.generator)
    pipe = MISCPipeline(
        critic, generator, method=args.method, routing=args.routing,
        action_mode=args.action, seed_mode=args.seed_mode,
        n_init=args.n_init, k_steps=args.k_steps, budget=args.budget,
    )

    records_path = out_dir / "records.jsonl"
    cfg = {k: getattr(args, k) for k in vars(args)}
    (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2))

    with open(records_path, "w", encoding="utf-8") as fout:
        for i, task in enumerate(tasks):
            image, tr = pipe.run(task)
            rec = {
                "task_id": tr.task_id,
                "method": tr.method,
                "routing": tr.routing,
                "action_mode": tr.action_mode,
                "seed_mode": tr.seed_mode,
                "num_subjects": task.num_subjects,
                "budget": tr.budget,
                "gen_calls": tr.gen_calls,
                "accepted_steps": tr.accepted_steps,
                "rejected_steps": tr.rejected_steps,
                "init_total": tr.init_total,
                "final_total": tr.final_total,
                "final_dims": tr.final_dims,
                "collateral_damage_rate": tr.collateral_damage_rate,
                "final_prompt": tr.final_prompt,
                "step_log": tr.step_log,
            }
            if not args.no_metrics:
                rec["independent"] = metrics.independent_metrics(
                    image, task.load_refs(), task.prompt
                )
            if args.save_images:
                p = img_dir / f"{tr.task_id}.png"
                image.save(p)
                rec["image_path"] = str(p)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 20 == 0:
                print(f"[{args.name}] {i+1}/{len(tasks)} done")

    print(f"[{args.name}] wrote {records_path}")


if __name__ == "__main__":
    main()
