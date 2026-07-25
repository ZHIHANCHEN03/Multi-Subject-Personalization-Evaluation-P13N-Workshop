"""Pipeline 1 (OURS): training-free closed-loop MIE-guided correction.

Loop (nothing trained):
  init : generate N_INIT candidates (different seeds), MIE-score, keep best-total
  step : standardize 3 dims against held-out median/MAD -> largest deficit
         -> apply action
         (prompt rewrite + reference-set manipulation) -> regenerate -> MIE-score
         -> accept only if total+target improve without collateral damage
  out  : best image over the whole trajectory

Budget B = N_INIT + K_STEPS generator calls (kept equal to best_of_n for fairness).

Run:
  python p1_ours.py --data hard_cases.jsonl --generator omnigen2 --n_init 4 --k 4
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import common
from actions import CalibratedRouter, RawRouter, apply_action
from generators import build_generator


def make_method(critic, generator, router, n_init: int, k_steps: int, action_mode: str = "both", seed_offset: int = 0):
    total_eps = float(os.environ.get("MIE_TOTAL_EPS", "0.0"))
    dim_eps = float(os.environ.get("MIE_DIM_EPS", "0.0"))
    collateral_tol = float(os.environ.get("MIE_COLLATERAL_TOL", "0.05"))
    # Only correct when the worst standardized deficit is meaningfully large;
    # small deficits mean the image is essentially fine -> leave it (fixes
    # easy-case regression where tinkering hurt already-good outputs).
    deficit_min = float(os.environ.get("OURS_DEFICIT_MIN", "0.75"))
    # Per correction step, propose several candidates (seed diversity) and let
    # MIE pick the best -> directed mini-search instead of a single blind edit.
    proposals = int(os.environ.get("OURS_PROPOSALS", "2"))

    def method(task: common.Task):
        base_refs = task.load_refs()
        base_prompt = common.omnigen_prompt(task)

        # ---- init: N_INIT candidates, keep best by MIE total ----
        best_img, best_score = None, None
        for s in range(n_init):
            print(
                f"[OURS][{task.task_id}] phase=initialization "
                f"candidate={s+1}/{n_init} action=generate",
                flush=True,
            )
            img = generator.generate(base_prompt, base_refs, seed=seed_offset + s)
            sc = critic.score(img, task)
            print(
                f"[OURS][{task.task_id}] phase=initialization "
                f"candidate={s+1}/{n_init} MIE_total={sc['total']:.4f} "
                f"E={sc['existence']:.3f} A={sc['appearance']:.3f} "
                f"I={sc['interaction']:.3f}",
                flush=True,
            )
            if best_score is None or sc["total"] > best_score["total"]:
                best_img, best_score = img, sc
        init_total = best_score["total"]

        gen_calls = n_init
        accepted, rejected, log = 0, 0, []
        cur_prompt = base_prompt

        # ---- correction steps ----
        for step in range(k_steps):
            dim, before_deficits = router.route(best_score, task.num_subjects)
            # Trigger gate: only correct when the worst deficit is large enough.
            if max(before_deficits.values()) <= deficit_min:
                print(
                    f"[OURS][{task.task_id}] phase=correction step={step+1}/{k_steps} "
                    f"action=early_stop reason=worst_deficit<={deficit_min}",
                    flush=True,
                )
                log.append(
                    {"step": step, "stopped": f"worst_deficit<={deficit_min}",
                     "deficits": before_deficits}
                )
                break
            new_prompt, new_refs, action = apply_action(
                task, cur_prompt, base_refs, dim, step, action_mode=action_mode
            )
            print(
                f"[OURS][{task.task_id}] phase=correction step={step+1}/{k_steps} "
                f"route={dim} deficits={before_deficits} action={action} "
                f"proposals={proposals}",
                flush=True,
            )
            # Multi-proposal: MIE picks the best candidate for this diagnosed edit.
            cand_img, cand_sc = None, None
            for p in range(proposals):
                img_p = generator.generate(new_prompt, new_refs, seed=seed_offset + 1000 + step * 10 + p)
                gen_calls += 1
                sc_p = critic.score(img_p, task)
                if cand_sc is None or sc_p["total"] > cand_sc["total"]:
                    cand_img, cand_sc = img_p, sc_p
            sc = cand_sc
            _, after_deficits = router.route(sc, task.num_subjects)
            total_improved = sc["total"] > best_score["total"] + total_eps
            target_improved = sc[dim] > best_score[dim] + dim_eps
            no_collateral = all(
                sc[other] >= best_score[other] - collateral_tol
                for other in common.DIMS
                if other != dim
            )
            improved = total_improved and target_improved and no_collateral
            print(
                f"[OURS][{task.task_id}] phase=verification step={step+1}/{k_steps} "
                f"MIE_total={sc['total']:.4f} target={sc[dim]:.3f} "
                f"accepted={improved} checks="
                f"total:{total_improved},target:{target_improved},"
                f"collateral:{no_collateral}",
                flush=True,
            )
            log.append(
                {
                    "step": step,
                    "routed_dim": dim,
                    "deficits_before": before_deficits,
                    "deficits_after": after_deficits,
                    "action": action,
                    "candidate_total": sc["total"],
                    "candidate_dims": {d: sc[d] for d in common.DIMS},
                    "accept_checks": {
                        "total_improved": total_improved,
                        "target_improved": target_improved,
                        "no_collateral": no_collateral,
                    },
                    "accepted": improved,
                }
            )
            if improved:
                best_img, best_score = cand_img, sc
                cur_prompt = new_prompt
                accepted += 1
            else:
                rejected += 1  # rollback: keep previous best

        info = {
            "budget": n_init + k_steps,
            "gen_calls": gen_calls,
            "init_total": init_total,
            "final_total": best_score["total"],
            "final_dims": {d: best_score[d] for d in common.DIMS},
            "accepted_steps": accepted,
            "rejected_steps": rejected,
            "step_log": log,
        }
        return best_img, info

    return method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", default="ours")
    ap.add_argument("--generator", default="omnigen2")
    ap.add_argument("--n_init", type=int, default=4)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument(
        "--calibration",
        default=os.environ.get(
            "MIE_CALIBRATION",
            str(Path(__file__).resolve().parent / "results" / "calibration" / "mie_baselines.json"),
        ),
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_scr", action="store_true")
    ap.add_argument("--routing", default="calibrated", choices=["calibrated", "raw"],
                    help="ablation: calibrated deficit routing vs raw-argmin (lowest score)")
    ap.add_argument("--action", default="both", choices=["both", "prompt_only"],
                    help="ablation: prompt+reference-set vs prompt-only")
    ap.add_argument("--seed_offset", type=int, default=0,
                    help="shift all generation seeds (for multi-seed Round-2 runs)")
    args = ap.parse_args()

    tasks = common.load_tasks(args.data, limit=args.limit)
    critic = common.build_critic()
    generator = build_generator(args.generator)
    if args.routing == "raw":
        router = RawRouter()
    elif Path(args.calibration).exists():
        router = CalibratedRouter(args.calibration)
    elif args.generator == "mock":
        print("[ours] calibration missing in mock mode; using raw router")
        router = RawRouter()
    else:
        raise FileNotFoundError(
            f"frozen MIE calibration missing: {args.calibration}"
        )
    scorer = None if args.no_scr else common.DinoScorer()
    method = make_method(critic, generator, router, args.n_init, args.k,
                         action_mode=args.action, seed_offset=args.seed_offset)
    common.run_over_dataset(args.name, method, tasks, scorer)


if __name__ == "__main__":
    main()
