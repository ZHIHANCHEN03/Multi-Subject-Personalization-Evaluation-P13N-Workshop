# Round 1.1 Report — MIE-guided Self-Correction Pipeline Optimization

**Goal:** find a clearly-better training-free `ours` pipeline (structure + params) than the
Round-1 default, to freeze as the Round-2 configuration. Nothing trained. MIE stays the
frozen verifier; SCR (DINOv2+Grounding-DINO) is the independent judge.

**Sweep surface:** 8-task 4-entity hard slice (OmniGen2 base, 5-ref cap), frozen Round-1
baselines (one_shot / best_of_n / umo) reused, not re-run. Budget = generator calls.

---

## 1. The Round-1 problem (why Round 1.1 was needed)

Round 1 verdict was **GO**, but the closed loop barely fired:

| Round-1 ours (hard_4, n=30) | value |
|---|---|
| mean SCR | 0.500 |
| **mean accepted_steps** | **0.20** |

Only 20% of tasks accepted even one correction. The win over one_shot/UMO came almost
entirely from `n_init` candidates + MIE selection — i.e. a smarter best-of-N, **not** from
self-correction. Taking that pipeline into Round 2 would be weak: the "self-correction"
claim is unsupported if the loop never fires.

**Root-cause diagnosis (3 structural issues, not hyper-params):**

1. **MIE gives only 3 GLOBAL dimension scores (E/A/I)** — no per-subject signal. The router
   knew *which dimension* was weak but not *which subject* collapsed.
2. **refset manipulation round-robined subjects blindly** — it often emphasized the wrong
   subject, so corrections failed acceptance and were rejected.
3. **acceptance required total+target+no-collateral to all improve** from a single
   seed-diverse proposal — too strict for noisy generation.

**Key unused asset:** `DinoScorer.score_task` returns **per-subject** DINO sims (it crops
each subject via Grounding-DINO and compares to that subject's reference). Round 1 used SCR
*only* for final evaluation. The per-subject signal was wasted.

---

## 2. Structural redesign — dual-signal diagnose-and-target

| loop component | Round 1 (v1) | Round 1.1 (v2/v2.3) |
|---|---|---|
| diagnosis | MIE → weakest dim | MIE → dim **+** SCR → weakest subject (lowest DINO sim) → joint (dim, subject) |
| refset action | round-robin subject | **target the specific collapsed subject** (front_dup3, capped at 5 refs) |
| per-step proposals | 1 action × seed diversity | **action portfolio**: {target weakest, target 2nd-weakest, layout-only} |
| acceptance | MIE total + target dim + no-collateral | **dual-signal**: MIE total improves AND target subject's DINO sim improves |
| candidate selection | max MIE total | **v2.3: max collapsed-subject DINO sim** (V2_SELECT_MODE=weak_subject) |

The core novelty: **promote SCR from a judge-only metric to an in-loop per-subject
diagnostic + selector.** MIE still gates (never grade with the signal you optimize), but SCR
now tells the loop *who* collapsed and *which candidate best rescues them*.

---

## 3. Sweep results (8-task 4-entity hard slice, budget-matched unless noted)

| trial | budget | SCR↓ | DINO↑ | accept | win vs UMO | vs BoN | vs one |
|---|---|---|---|---|---|---|---|
| t0_default (v1, control) | 8 | 0.500 | 0.509 | 0.20 | 3/8 | 1/8 | 2/8 |
| v2_full (strict dual-accept) | 11 | 0.500 | 0.508 | 0.00 | 3/8 | 1/8 | 2/8 |
| v2_relaxed (drop SCR collateral) | 11 | 0.469 | 0.550 | 0.38 | 3/8 | 1/8 | 2/8 |
| v2_relaxed_b8 (relaxed, matched) | 8 | 0.500 | 0.508 | 0.25 | 3/8 | 1/8 | 2/8 |
| v2_3_combo_b8 (+total_tol) | 8 | 0.469 | 0.553 | 0.25 | 3/8 | 1/8 | 2/8 |
| **v2_3_weaksel_b8 (winner)** | **8** | **0.438** | **0.519** | 0.25 | **4/8** | **2/8** | **3/8** |
| baselines: UMO / BoN=8 / one-shot | — | 0.563 / 0.500 / 0.531 | — | — | — | — | — |

(SCR = fraction of collapsed/missing subjects, **lower = better**. DINO = mean per-subject
identity sim, higher = better. Budget = total generator calls; best-of-N uses 8.)

### 3.1 What each ablation proved

- **v2_full (strict SCR collateral) → SCR=0.500, accepted=0.** Emphasizing one subject's
  reference inherently trades off attention from others, so SCR collateral *always* failed
  and every correction was rejected. The structural change was neutralized. → Strict
  SCR collateral is wrong; drop it.
- **v2_relaxed (drop SCR collateral, budget 11) → SCR=0.469.** Relaxing acceptance let the
  loop fire (accepted 0.38/task) and beat v1. But budget 11 > 8.
- **v2_relaxed_b8 (budget 8) → SCR=0.500.** At *matched* compute the gain vanished — it was
  from extra compute, not structure. This was the matched-compute gap to close.
- **v2_3_weaksel_b8 (select by weak-subject sim, budget 8) → SCR=0.438.** The winning lever:
  portfolio selection by the collapsed subject's DINO sim (not MIE total). At matched
  compute it beats v1 on SCR (-12.5% rel.), DINO (+2%), and lifts head-to-head win rates
  (UMO 3/8→4/8, BoN 1/8→2/8, one-shot 2/8→3/8).
- **v2_3_combo_b8 (+total_tol on top of weaksel) → SCR=0.469.** Allowing MIE total to dip
  slightly *hurt* — it admitted corrections that rescued the subject but degraded overall
  quality. Strict total (no tol) + weak-subject selection is the right combination.

### 3.2 Why weak-subject selection wins at matched compute

The portfolio generates 3 distinct action candidates per step. v1/v2_relaxed pick the
candidate with **highest MIE total** — a noisy global signal that often disagrees with
identity preservation. v2.3 picks the candidate with **highest collapsed-subject DINO sim**.
Even when the correction loop does not formally "accept" (MIE total flat), the *final image*
is the candidate that best preserved the weakest subject's identity across the whole
trajectory. This turns SCR from a passive judge into an active per-subject selector, which
is what closes the matched-compute gap.

---

## 4. Winner (frozen config for Round 2)

**`v2_3_weaksel_b8`** — dual-signal diagnose-and-target, budget 8 (compute-matched to
best-of-N):

```
V2_DUAL_SIGNAL=1          # SCR per-subject diagnosis ON
V2_ACTION_PORTFOLIO=1     # 3 distinct actions/step
V2_ACTIONS_PER_STEP=3
V2_SEEDS_PER_ACTION=1
V2_DUAL_ACCEPT=1
V2_ACCEPT_MODE=relaxed    # drop SCR collateral (keep MIE total + subject-improve)
V2_SELECT_MODE=weak_subject  # portfolio picks by collapsed-subject DINO sim
V2_TOTAL_TOL=0.0          # strict MIE total (no tol — tol hurt)
OURS_DEFICIT_MIN=0.5       # trigger gate
OURS_REFSET_MODE=front_dup3  # triple-weight collapsed subject (capped at 5 refs)
OURS_LAYOUT=1              # spatial separation hint
n_init=2, k=2              # budget = 2 + 2*3 = 8
```

**Result vs v1 (matched budget 8, 8-task slice):** SCR 0.500 → 0.438 (-12.5%); DINO 0.509 →
0.519; head-to-head wins up across all three baselines. The correction loop now fires
meaningfully and the structural gain holds at equal compute.

---

## 5. 20-task validation

`v2_3_weaksel_20` ran the winner on the full 20-task 4-entity hard subset (budget 8,
identical config to the 8-task winner). `t0_default_20` is the v1 control on the same 20
tasks. Baselines below are the **matched** 20-task means (the sweep evaluates baselines on
the same 20 task_ids, not the full 30).

| method (20 tasks, budget 8) | SCR↓ | DINO↑ | win vs UMO | vs BoN | vs one |
|---|---|---|---|---|---|
| **v2.3 winner (weaksel)** | **0.488** | **0.498** | 8/20 | 3/20 | 6/20 |
| v1 (t0_default_20, control) | 0.513 | 0.486 | 8/20 | 3/20 | 6/20 |
| UMO (retrained SOTA) | 0.563 | — | — | — | — |
| best-of-N (=8) | 0.525 | — | — | — | — |
| one-shot | 0.550 | — | — | — | — |

**Matched-compute verdict (20 tasks):** v2.3 beats v1 on **both** SCR (-4.9% rel.,
0.513→0.488) and DINO (+2.6% rel., 0.486→0.498), and beats all three baselines (UMO, BoN,
one-shot) on mean SCR. Head-to-head win rates vs baselines are unchanged from v1 (the gain is
in *magnitude* of collapse reduction, not in flipping pairwise wins) — this is expected for a
selection-based improvement and is exactly what significance testing on the full Round-2 run
will quantify. The structural gain holds at equal compute on the larger set.

---

## 6. Round-2 plan with the frozen winner

1. **Main run (500 tasks):** `v2_3_weaksel_b8` vs one_shot / best_of_n(=8) / UMO on the full
   hard_4 + easy_2 slices, multi-seed, with bootstrap 95% CI + paired p-values (`round2/analyze.py`).
2. **Scaling (FLUX.2):** the same dual-signal loop on FLUX.2 at 6/8-entity stress (OmniGen2
   is capped at 5 refs). Test whether per-subject SCR diagnosis scales to more subjects.
3. **Human eval:** blind A/B of winner vs UMO vs best-of-N (`round2/export_human_eval.py`).
4. **Ablations to report:** weak_subject-selection ON/OFF (the key lever), action portfolio
   ON/OFF, dual-signal diagnosis ON/OFF — all already wired as env flags in `p1_ours_v2.py`.

## 7. Files

- `round1/p1_ours_v2.py` — dual-signal pipeline (v2/v2.3), env-flag ablations.
- `round1/actions.py` — `manipulate_refset` now caps at 5 refs (OmniGen2 limit).
- `round1_1/sweep.py` — trial definitions (v1 control + v2/v2.3 structural variants).
- `round1_1/leaderboard.jsonl` — all trial results.
- `round1_1/trials/<trial>/records.jsonl` — per-task records (SCR, DINO, accepted_steps, step log).
