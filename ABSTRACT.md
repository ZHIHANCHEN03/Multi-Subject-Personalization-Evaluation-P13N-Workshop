# (Working Title) Training-Free Test-Time Repair of Multi-Subject Identity Collapse via a Decomposed Verifier

> AAAI abstract draft. Bracketed `[…]` are placeholders to fill with Round-2 numbers.
> Framing follows the locked claim; collision boundaries (Ma et al., PromptEnhancer,
> FreeCus/FreeGraftor, UMO/MultiCrafter) are respected.

## Abstract

Multi-subject personalized generation still suffers from *identity collapse*:
as the number of referenced subjects grows and their interactions become
physically entangled, generators drop, merge, or swap identities. Prior work
either *measures* this failure (multi-subject benchmarks and identity metrics)
or *repairs* it by **retraining** the generator (e.g., reinforcement- or
LoRA-based alignment), which is costly and must be redone for every new base
model. We ask a different question: **can multi-subject identity collapse be
repaired at inference time, without training any weights?**

We present a **training-free, test-time correction loop** that repurposes a
frozen *decomposed* multi-subject evaluator as a **controller** rather than a
scorer. At each step the evaluator diagnoses which identity facet
(existence / appearance / interaction) has degraded **relative to a frozen,
per-difficulty calibrated baseline**—so that a facet whose score is merely
*low in absolute terms* is not blindly targeted—and drives a **semantic-level**
edit: a facet-specific prompt rewrite together with **reference-set
manipulation** (reordering/duplicating reference images), a lever unique to
multi-reference generators. A candidate is accepted only when the preference
score improves, the targeted facet improves, and no other facet regresses;
otherwise it is rolled back. Crucially, the evaluator is used only *inside* the
loop; final quality is judged by an **independent** detection-based identity
metric (Subject Collapse Rate) and human preference, avoiding self-evaluation.

Because the method touches no weights, it plugs into any multi-reference
generator on day one. On [OmniGen2] over strong-interaction, occlusion-heavy
multi-subject prompts, our correction reduces Subject Collapse Rate by [X%]
over one-shot generation and by [Y%] over compute-matched best-of-N selection,
and—**without any training**—reaches parity with the retrained state-of-the-art
[UMO] (within [Δ] SCR) on the same base model, while [humans prefer our outputs
in Z% of pairs]. Ablations show that **calibrated routing** is essential
(uncalibrated argmin degenerates to always targeting the same facet) and that
**reference-set manipulation** contributes beyond prompt rewriting alone.

We position this as distinct from (i) noise-level, scalar-verifier inference-time
scaling, (ii) open-loop, single-pass training-free personalization, and
(iii) retraining-based identity repair: to our knowledge this is the first
demonstration that interaction-induced multi-subject identity collapse is
**diagnosable and repairable at test time, training-free, at the semantic
level**, closing much of the gap to retraining at zero training cost.

## Contributions
- **A training-free, closed-loop test-time repair** for multi-subject identity
  collapse that uses a frozen decomposed verifier as a controller (diagnose →
  semantic edit → verified accept/rollback).
- **Calibrated dimension routing**: correcting the facet with the largest
  deficit *relative to a frozen per-difficulty baseline*, not the lowest raw
  score—necessary because facet scores are heterogeneously scaled.
- **Reference-set manipulation** as a correction lever unique to
  multi-reference generators, complementary to prompt rewriting.
- **Evidence that retraining is not necessary** to recover most of the
  identity-preservation gap: on the same base model, the training-free loop
  reaches near-parity with a retrained SOTA, evaluated by an independent
  detection-based metric and human preference.

## Notes for filling in (not for submission)
- Numbers `[X/Y/Δ/Z]` come from Round 2 (500 tasks, 2/4/6/8 subjects, human eval).
- Base model on this draft = OmniGen2 (UMO's base → same-base fair comparison).
  FLUX.2 to be added as a "newest base, plug-and-play" generality showcase.
- FreeGraftor is a cross-system (FLUX.1) open-loop reference, not a same-base
  causal comparison; state this explicitly in the paper.
- Do NOT claim "decomposed > scalar" (PromptEnhancer) or "correction > selection
  in general" (Ma et al. 2025) as novelty; the novelty is the training-free
  semantic-level repair of identity collapse and the calibrated controller.
