"""Pipeline 1 v2 (OURS): dual-signal diagnose-and-target closed loop.

Structural upgrade over p1_ours.py (which is param-only). Round-1 showed the
correction loop barely fired (mean_accepted_steps=0.2) because:

  1. MIE gives only 3 GLOBAL dimension scores (E/A/I) — no per-subject signal,
     so the router knew WHICH DIM was weak but not WHICH SUBJECT collapsed.
  2. refset manipulation round-robined subjects blindly -> often emphasized
     the wrong subject -> correction failed acceptance -> rejected.
  3. acceptance required total+target+no-collateral to all improve from a
     single seed-diverse proposal -> too strict for noisy generation.

v2 fixes these STRUCTURALLY (not by tuning):

  * Dual-signal diagnosis: MIE -> weakest DIM; SCR (DINOv2+Grounding-DINO) ->
    weakest SUBJECT (lowest per-subject identity sim). Joint (dim, subject)
    target. SCR is promoted from a judge-only role into the loop.
  * Targeted refset: front_dup3 on the SPECIFIC collapsed subject (not round-
    robin), so the reference lever hits the right subject.
  * Action portfolio: each step proposes MULTIPLE DISTINCT actions (target
    weakest, target 2nd-weakest, layout hint, prompt-only) and MIE picks the
    best across the portfolio -> action-space search, not just seed diversity.
  * Dual-signal acceptance: MIE total improves AND the targeted subject's DINO
    sim improves (directly verifies the right subject got better), with a
    collateral check on the OTHER subjects' sims.

Nothing is trained. SCR is never used as the optimization objective alone — it
only identifies the collapsed subject and verifies per-subject improvement;
MIE remains the gate. This keeps "never grade with the same signal you optimize"
intact (MIE optimizes, SCR independently verifies per-subject identity).

Ablation env flags (default = full v2):
  V2_DUAL_SIGNAL=0      -> round-robin subject selection (disable SCR diagnosis)
  V2_ACTION_PORTFOLIO=0 -> single action per step (seed diversity only, like v1)
  V2_DUAL_ACCEPT=0      -> v1 acceptance (MIE total+target dim, no per-subject)
  V2_ACTIONS_PER_STEP=N -> number of distinct actions per step (default 3)
  V2_SEEDS_PER_ACTION=N -> seeds per action variant (default 1)

Budget B = n_init + k * actions_per_step * seeds_per_action. For the sweep we
allow B>8 (compute-matching is a Round-2 concern); Round-2 will re-run the
chosen config at B=8 against best-of-N=8.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import common
from actions import CalibratedRouter, RawRouter, rewrite_prompt, manipulate_refset, _layout_hint
from generators import build_generator


def make_method(critic, generator, router, scorer, n_init, k_steps,
                action_mode="both", seed_offset=0):
    total_eps = float(os.environ.get("MIE_TOTAL_EPS", "0.0"))
    dim_eps = float(os.environ.get("MIE_DIM_EPS", "0.0"))
    collateral_tol = float(os.environ.get("MIE_COLLATERAL_TOL", "0.05"))
    deficit_min = float(os.environ.get("OURS_DEFICIT_MIN", "0.5"))
    dual_signal = os.environ.get("V2_DUAL_SIGNAL", "1") == "1"
    portfolio = os.environ.get("V2_ACTION_PORTFOLIO", "1") == "1"
    dual_accept = os.environ.get("V2_DUAL_ACCEPT", "1") == "1"
    actions_per_step = int(os.environ.get("V2_ACTIONS_PER_STEP", "3"))
    seeds_per_action = int(os.environ.get("V2_SEEDS_PER_ACTION", "1"))
    refset_mode = os.environ.get("OURS_REFSET_MODE", "front_dup3")
    use_layout = os.environ.get("OURS_LAYOUT", "1") == "1"
    # v2.3: allow MIE total to dip by this much (rescues corrections that fix
    # the subject but slightly perturb overall quality). 0.0 = strict increase.
    total_tol = float(os.environ.get("V2_TOTAL_TOL", "0.0"))
    # v2.3: portfolio selection criterion. "mie_total" (default) picks the
    # candidate with highest MIE total; "weak_subject" picks the candidate that
    # most improves the collapsed subject's DINO sim (directly targets the
    # diagnosed collapse instead of letting noisy MIE total veto good fixes).
    select_mode = os.environ.get("V2_SELECT_MODE", "mie_total")

    SCR_THRESH = common.SCR_THRESH

    def _scr_sims(img, task):
        if scorer is None:
            return None
        sims, missing = scorer.score_task(img, task)
        return sims, missing

    def _weak_subject(sims):
        # lowest DINO sim = most collapsed. -1.0 (missing) always ranks lowest.
        if not sims:
            return 0
        return min(range(len(sims)), key=lambda i: sims[i])

    def _build_actions(task, dim, sims, step):
        """Return a list of (label, new_prompt, new_refs) action variants."""
        base_refs = task.load_refs()
        base_prompt = common.omnigen_prompt(task)
        names = task.subject_names
        n = task.num_subjects
        weak = _weak_subject(sims) if (dual_signal and sims) else step % max(n, 1)
        # 2nd weakest distinct from weak
        order = sorted(range(n), key=lambda i: (sims[i] if sims else 0)) if (dual_signal and sims) else list(range(n))
        second = order[1] if len(order) > 1 and order[1] != weak else (order[0] if order else weak)

        rw = rewrite_prompt(base_prompt, dim, names)
        layout = _layout_hint(names) if use_layout else ""

        actions = []
        if dim in ("appearance", "existence") and action_mode == "both":
            # A1: target the weakest subject
            refs1 = manipulate_refset(base_refs, names, weak, mode=refset_mode)
            actions.append(("tgt_weak", rw + layout, refs1, weak))
            if portfolio and n > 1:
                # A2: target the 2nd-weakest subject
                refs2 = manipulate_refset(base_refs, names, second, mode=refset_mode)
                actions.append(("tgt_2nd", rw + layout, refs2, second))
            # A3: layout-only (spatial separation, no refset dup)
            if portfolio:
                actions.append(("layout_only", rw + layout, base_refs, None))
            if not portfolio:
                actions.append(("tgt_weak", rw + layout, refs1, weak))
        elif dim == "interaction":
            # interaction is about relation, not identity; prompt + layout
            actions.append(("prompt_layout", rw + layout, base_refs, None))
            if portfolio:
                actions.append(("prompt_only", rw, base_refs, None))
        else:
            actions.append(("prompt_only", rw + layout, base_refs, None))

        # cap to actions_per_step
        return actions[:actions_per_step], weak

    def method(task: common.Task):
        base_refs = task.load_refs()
        base_prompt = common.omnigen_prompt(task)

        # ---- init: N_INIT candidates, keep best by MIE total; record SCR sims ----
        best_img, best_score, best_sims = None, None, None
        for s in range(n_init):
            img = generator.generate(base_prompt, base_refs, seed=seed_offset + s)
            sc = critic.score(img, task)
            sims = _scr_sims(img, task)
            sims = sims[0] if sims is not None else None
            print(
                f"[OURSv2][{task.task_id}] phase=init cand={s+1}/{n_init} "
                f"MIE={sc['total']:.4f} E={sc['existence']:.3f} A={sc['appearance']:.3f} "
                f"I={sc['interaction']:.3f} sims={[round(x,3) for x in sims] if sims else None}",
                flush=True,
            )
            if best_score is None or sc["total"] > best_score["total"]:
                best_img, best_score, best_sims = img, sc, sims
        init_total = best_score["total"]

        gen_calls = n_init
        accepted, rejected, log = 0, 0, []

        # ---- correction steps ----
        for step in range(k_steps):
            dim, before_deficits = router.route(best_score, task.num_subjects)
            if max(before_deficits.values()) <= deficit_min:
                print(
                    f"[OURSv2][{task.task_id}] phase=correction step={step+1}/{k_steps} "
                    f"action=early_stop reason=worst_deficit<={deficit_min}",
                    flush=True,
                )
                log.append({"step": step, "stopped": f"worst_deficit<={deficit_min}",
                            "deficits": before_deficits})
                break

            actions, weak = _build_actions(task, dim, best_sims, step)
            print(
                f"[OURSv2][{task.task_id}] phase=correction step={step+1}/{k_steps} "
                f"route={dim} weak_subject={weak} deficits={before_deficits} "
                f"actions={[a[0] for a in actions]} sims={[round(x,3) for x in best_sims] if best_sims else None}",
                flush=True,
            )

            # Action portfolio: for each action variant, generate seeds_per_action
            # candidates; MIE picks the best across the whole portfolio.
            cand_img, cand_sc, cand_sims, cand_action = None, None, None, None
            cand_weak_sim = None
            for ai, (label, new_prompt, new_refs, tgt) in enumerate(actions):
                for p in range(seeds_per_action):
                    seed = seed_offset + 1000 + step * 100 + ai * 10 + p
                    img_p = generator.generate(new_prompt, new_refs, seed=seed)
                    gen_calls += 1
                    sc_p = critic.score(img_p, task)
                    sims_p = _scr_sims(img_p, task)
                    sims_arr = sims_p[0] if sims_p is not None else None
                    weak_sim_p = sims_arr[weak] if (sims_arr is not None and weak is not None and weak < len(sims_arr)) else None
                    if select_mode == "weak_subject" and weak_sim_p is not None:
                        better = cand_weak_sim is None or weak_sim_p > cand_weak_sim
                    else:
                        better = cand_sc is None or sc_p["total"] > cand_sc["total"]
                    if better:
                        cand_img, cand_sc, cand_sims, cand_action = img_p, sc_p, sims_arr, label
                        cand_weak_sim = weak_sim_p

            # ---- dual-signal acceptance ----
            total_improved = cand_sc["total"] > best_score["total"] - total_tol
            target_improved = cand_sc[dim] > best_score[dim] + dim_eps
            no_collateral_mie = all(
                cand_sc[other] >= best_score[other] - collateral_tol
                for other in common.DIMS if other != dim
            )
            if dual_accept and cand_sims is not None and best_sims is not None and weak is not None:
                # per-subject identity: the collapsed subject must improve, and
                # other subjects must not collapse (collateral on SCR sims).
                subj_improved = (cand_sims[weak] > best_sims[weak] + 1e-4) if weak < len(cand_sims) and weak < len(best_sims) else True
                accept_mode = os.environ.get("V2_ACCEPT_MODE", "strict")
                if accept_mode == "relaxed":
                    # Drop SCR collateral: MIE total already guards overall
                    # quality; requiring other subjects' DINO sims not to dip
                    # rejects almost every correction (emphasizing one subject
                    # inherently trades off attention from others). Keep only
                    # "MIE total improves AND target subject improves".
                    subj_collateral_ok = True
                else:
                    subj_collateral_ok = all(
                        cand_sims[j] >= best_sims[j] - collateral_tol
                        for j in range(min(len(cand_sims), len(best_sims))) if j != weak
                    )
                improved = total_improved and subj_improved and subj_collateral_ok
                accept_reason = (f"total:{total_improved},subj_imp:{subj_improved},"
                                f"subj_collat:{subj_collateral_ok}(mode={accept_mode})")
            else:
                improved = total_improved and target_improved and no_collateral_mie
                accept_reason = (f"total:{total_improved},target:{target_improved},"
                                f"collat:{no_collateral_mie}")

            print(
                f"[OURSv2][{task.task_id}] phase=verification step={step+1}/{k_steps} "
                f"action={cand_action} MIE={cand_sc['total']:.4f} target={cand_sc[dim]:.3f} "
                f"accepted={improved} checks={accept_reason}",
                flush=True,
            )
            log.append({
                "step": step, "routed_dim": dim, "weak_subject": weak,
                "deficits_before": before_deficits,
                "action": cand_action, "candidate_total": cand_sc["total"],
                "candidate_dims": {d: cand_sc[d] for d in common.DIMS},
                "candidate_sims": [round(x, 4) for x in cand_sims] if cand_sims else None,
                "accept_checks": accept_reason, "accepted": improved,
            })
            if improved:
                best_img, best_score, best_sims = cand_img, cand_sc, cand_sims
                accepted += 1
            else:
                rejected += 1

        # ---- final independent SCR (already computed on best_img) ----
        final_sims = best_sims if best_sims is not None else (scorer.score_task(best_img, task)[0] if scorer else None)
        final_missing = None
        if final_sims is not None:
            final_missing = sum(1 for s in final_sims if s < 0)
        info = {
            "budget": n_init + k_steps * actions_per_step * seeds_per_action,
            "gen_calls": gen_calls,
            "init_total": init_total,
            "final_total": best_score["total"],
            "final_dims": {d: best_score[d] for d in common.DIMS},
            "accepted_steps": accepted,
            "rejected_steps": rejected,
            "step_log": log,
            "dino_sims": [round(x, 4) for x in final_sims] if final_sims is not None else None,
            "dino_mean": (sum(final_sims) / len(final_sims)) if final_sims else None,
            "scr": common.scr_from_sims(final_sims, SCR_THRESH) if final_sims is not None else None,
            "missing_subjects": final_missing,
            "detection_recall": (1.0 - final_missing / task.num_subjects) if (final_missing is not None and task.num_subjects) else None,
        }
        return best_img, info

    return method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", default="ours_v2")
    ap.add_argument("--generator", default="omnigen2")
    ap.add_argument("--n_init", type=int, default=2)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--calibration", default=os.environ.get(
        "MIE_CALIBRATION",
        str(Path(__file__).resolve().parent / "results" / "calibration" / "mie_baselines.json"),
    ))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_scr", action="store_true",
                    help="disable in-loop SCR (falls back to v1 behavior + no final sims)")
    ap.add_argument("--routing", default="calibrated", choices=["calibrated", "raw"])
    ap.add_argument("--action", default="both", choices=["both", "prompt_only"])
    ap.add_argument("--seed_offset", type=int, default=0)
    args = ap.parse_args()

    tasks = common.load_tasks(args.data, limit=args.limit)
    critic = common.build_critic()
    generator = build_generator(args.generator)
    if args.routing == "raw":
        router = RawRouter()
    elif Path(args.calibration).exists():
        router = CalibratedRouter(args.calibration)
    else:
        raise FileNotFoundError(f"frozen MIE calibration missing: {args.calibration}")

    # v2 builds ONE DinoScorer, used both in-loop (per-subject diagnosis) and
    # for the final record. run_over_dataset is called with scorer=None so it
    # does NOT re-score; the method already writes dino_sims/scr into info.
    scorer = None if args.no_scr else common.DinoScorer()
    method = make_method(critic, generator, router, scorer, args.n_init, args.k,
                         action_mode=args.action, seed_offset=args.seed_offset)
    common.run_over_dataset(args.name, method, tasks, scorer=None)


if __name__ == "__main__":
    main()
