# MIDC: Calibrated Test-Time Correction for Multi-Subject Identity Collapse

> AAAI-27 (2027) abstract submission. The abstract below is the **final**
> version, synchronized with `paper/main.tex` (2026-07-25). It carries the
> 500-task, 3-seed Round 2 final numbers (0.531 / 0.536 / 0.470 + human-eval
> 84.6% + calibrated-routing ablation +10.3%). The 7/21 OpenReview submission
> used an earlier title/abstract ("Training-Free ... Dual-Signal Decomposed
> Verifier", preliminary 0.563/0.525/0.488) — **the OpenReview title + abstract
> fields MUST be updated to the version below before the 7/28 full-paper
> deadline** (see ⚠️ action item at the bottom).

## Abstract

Personalized image generation with multiple subjects suffers from *interaction-induced identity collapse*: as the number of subjects grows, models increasingly merge, swap, or drop identities, even when each subject is provided as a reference image. We introduce **MIDC**, a training-free, inference-time correction paradigm that treats a *decomposed* verifier as a structured diagnostic signal and routes correction to the most deficient facet. Given a candidate image, a decomposed evaluator scores three facets—existence, appearance, and interaction—and we convert each facet score into a standardized deficit against a calibrated per-facet baseline. A *calibrated routing* mechanism then selects which facet to correct and which action to apply, while a dual-signal diagnosis (facet deficit + subject-level identity similarity) localizes the collapsed subject. Correction proceeds as a propose–verify loop with a guarded acceptance criterion that accepts a revision only if the verifier's overall score does not regress and the collapsed subject's identity similarity strictly improves. On OmniGen2, MIDC reduces Subject Collapse Rate (SCR) to **0.470** on 4-entity hard cases versus **0.531** for the retrained state-of-the-art (UMO) and **0.536** for one-shot generation. Notably, the retrained SOTA is itself a near-tie with one-shot (SCR +0.005, DINO identity similarity Δ=−0.0002 over 500×3 pairs): retraining the *identity-consistency* axis yields near-zero gain on hard interaction cases, indicating collapse's cause is not on the optimized axis. MIDC's decomposed diagnosis localizes that cause and cheaply repairs it at inference, with 15/16 paired bootstrap comparisons significant at *p*<0.05. On FLUX.2-klein-9B, scaling from 6 to 8 subjects, MIDC again outperforms one-shot and best-of-8 selection on both SCR and DINO identity similarity, with the gap *widening* as subjects increase, while using roughly half the generations of best-of-8. A blind human study independently corroborates the existence result: labelers prefer MIDC over UMO 84.6% of the time (*p*=4×10⁻⁷). Ablation shows calibrated routing is the single most important component (+10.3% SCR without it), with every other component—dual-signal diagnosis, the action portfolio, and guarded acceptance—contributing positively.

## Contributions

- **A training-free, closed-loop test-time repair** for multi-subject identity
  collapse that uses a frozen decomposed verifier as a controller (diagnose →
  semantic edit → verified accept/rollback), touching no model weights.
- **Dual-signal diagnosis**: the decomposed evaluator identifies the degraded
  *facet* (calibrated against a frozen per-difficulty baseline), while an
  independent detection-based identity scorer identifies the collapsed
  *subject*—promoting the identity scorer from a passive judge into an active
  in-loop diagnostic and selector.
- **Reference-set manipulation** as a correction lever unique to
  multi-reference generators, complementary to prompt rewriting.
- **Evidence that retraining is not necessary** to recover most of the
  identity-preservation gap: on the same base model, the training-free loop
  reaches parity with—and on the hardest cases slightly exceeds—a retrained
  SOTA, evaluated by an independent detection-based metric and human preference.

## AAAI-27 Submission Checklist (notes, not for submission)

**Key deadlines (all 11:59 PM UTC-12 / "anywhere on Earth")**:
- **2026-07-21**: Abstracts due — full title + complete abstract, no placeholders (vacuous submissions deleted, no full paper allowed)
- **2026-07-28**: Full papers due
- **2026-07-31**: Supplementary material + code due
- **2026-09-24**: Phase 1 rejection notification
- **2026-10-19 to 10-25**: Author feedback window
- **2026-11-30**: Final acceptance/rejection notification
- **2026-12-14**: Camera-ready files due
- **2027-02-16 to 02-23**: Conference (Montréal, Canada)

**Format**:
- AAAI-27 official LaTeX author kit, two-column, US Letter, Type 1/TrueType fonts
- Main paper: up to **7 pages** of content; pages 8–9 reserved **exclusively for references** (max 9 total)
- Anonymized for double-blind review (no author/affiliation; acknowledgements omitted)
- PDF only at submission; source files required only if accepted

**OpenReview portal**: register + submit at the AAAI-27 Conference group
(`https://openreview.net/group?id=AAAI.org/2027/Conference`).

**What's locked after 2026-07-21** (cannot change until review ends):
- paper topics (primary + secondaries)
- nominated reciprocal reviewer

**Editable 2026-07-21 → 2026-07-28** (but should not be substantively changed):
- title, TL;DR, abstract
- author list / order
- submitted paper PDF

**After 2026-07-31**: nothing can be changed until notification.

**⚠️ Action items for this paper**:
1. **DONE (2026-07-21)**: abstract submitted to OpenReview with the *old* title/numbers.
2. **⚠️ MUST DO before 2026-07-28**: log in to OpenReview and **update the title + abstract fields** to the final version above (title `MIDC: Calibrated Test-Time Correction for Multi-Subject Identity Collapse`; abstract body in the `## Abstract` section). AAAI permits editing title/abstract/TL;DR/PDF between 7/21 and 7/28, but warns it "may reject papers that change abstracts substantially" — our change is a reframing (dual-signal → calibrated routing) + number update, which is within bounds, but **must be done before 7/28**; after that nothing is editable until notification.
3. **By 2026-07-28**: submit the full 7-page PDF (built from `paper/main.tex` with the AAAI-27 `aaai2027.sty` kit, anonymized) + reproducibility checklist.
4. **By 2026-07-31**: supplementary material + code (commit `round2/results_r2/` raw records + MIE checkpoint to the repo or a Code and Data Supplement ZIP).

## Notes (not for submission)

- Numbers `[0.563 / 0.525 / 0.488]` are Round 1.1 preliminary (20-task 4-subject
  hard slice, compute-matched budget 8). The 500-task multi-seed run is in
  progress; final paper replaces these with bootstrap-CI-bearing 500-task means
  + paired p-values + human-eval win rates.
- Base model = OmniGen2 (UMO's base → same-base fair comparison). FLUX.2 is the
  "newer base, plug-and-play, scaling to 6/8 subjects" showcase (no same-base
  retrained SOTA exists there → only the ours-vs-baselines signal is claimed).
- FreeGraftor (FLUX.1) is a cross-system open-loop reference, not a same-base
  causal comparison; state this explicitly in the paper.
- Do NOT claim "decomposed > scalar" (PromptEnhancer) or "correction > selection
  in general" (Ma et al. 2025) as novelty; the novelty is the training-free
  semantic-level repair of identity collapse, the dual-signal controller, and
  the calibrated facet routing.
