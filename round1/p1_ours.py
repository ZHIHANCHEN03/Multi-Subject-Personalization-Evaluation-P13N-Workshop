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


def make_method(critic, generator, router, n_init: int, k_steps: int):
    total_eps = float(os.environ.get("MIE_TOTAL_EPS", "0.0"))
    dim_eps = float(os.environ.get("MIE_DIM_EPS", "0.0"))
    collateral_tol = float(os.environ.get("MIE_COLLATERAL_TOL", "0.05"))

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
            img = generator.generate(base_prompt, base_refs, seed=s)
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
            if max(before_deficits.values()) <= 0:
                print(
                    f"[OURS][{task.task_id}] phase=correction step={step+1}/{k_steps} "
                    "action=early_stop reason=all_dims_above_norm",
                    flush=True,
                )
                log.append(
                    {
                        "step": step,
                        "stopped": "all dimensions at/above calibrated norm",
                        "deficits": before_deficits,
                    }
                )
                break
            new_prompt, new_refs, action = apply_action(
                task, cur_prompt, base_refs, dim, step
            )
            print(
                f"[OURS][{task.task_id}] phase=correction step={step+1}/{k_steps} "
                f"route={dim} deficits={before_deficits} action={action}",
                flush=True,
            )
            img = generator.generate(new_prompt, new_refs, seed=1000 + step)
            gen_calls += 1
            sc = critic.score(img, task)
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
                best_img, best_score = img, sc
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
    args = ap.parse_args()

    tasks = common.load_tasks(args.data, limit=args.limit)
    critic = common.build_critic()
    generator = build_generator(args.generator)
    if Path(args.calibration).exists():
        router = CalibratedRouter(args.calibration)
    elif args.generator == "mock":
        print("[ours] calibration missing in mock mode; using raw router")
        router = RawRouter()
    else:
        raise FileNotFoundError(
            f"frozen MIE calibration missing: {args.calibration}"
        )
    scorer = None if args.no_scr else common.DinoScorer()
    method = make_method(critic, generator, router, args.n_init, args.k)
    common.run_over_dataset(args.name, method, tasks, scorer)


if __name__ == "__main__":
    main()
