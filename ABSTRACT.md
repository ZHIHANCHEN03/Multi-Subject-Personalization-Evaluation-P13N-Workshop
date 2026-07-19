# Training-Free Test-Time Repair of Multi-Subject Identity Collapse via a Dual-Signal Decomposed Verifier

> AAAI-27 (2027) abstract submission (due 2026-07-21). The abstract body below
> is complete and submittable as-is: it carries concrete Round 1.1 preliminary
> numbers (0.563 / 0.525 / 0.488), no placeholders. The full paper (due
> 2026-07-28) will replace those with the 500-task, multi-seed Round 2 final
> numbers (means + 95% CI + p-values + human-eval win rates). Claim and
> collision boundaries are locked (see PLAN.md §二, §六).

## Abstract

Multi-subject personalized image generation still suffers from *identity
collapse*: as the number of referenced subjects grows and their interactions
become physically entangled, generators drop, merge, or swap identities. Prior
work either *measures* this failure (multi-subject benchmarks and identity
metrics) or *repairs* it by **retraining** the generator
(reinforcement-/LoRA-based alignment), which is costly and must be repeated for
every new base model. We ask a different question: **can interaction-induced
multi-subject identity collapse be repaired at inference time, without
training any weights?**

We present a **training-free, test-time correction loop** that repurposes a
frozen *decomposed* multi-subject evaluator (MIE, scoring
existence/appearance/interaction) as a **controller**, not a scorer. The key
structural insight is a **dual-signal diagnosis**: the evaluator identifies
*which identity facet* has degraded relative to a frozen, per-difficulty
calibrated baseline—so a facet that is merely *low in absolute terms* is not
blindly targeted—while an independent detection-based identity scorer
(DINOv2 + Grounding-DINO) identifies *which subject* has collapsed. This joint
(facet, subject) target drives a **semantic-level** edit: a facet-specific
prompt rewrite together with **reference-set manipulation** (reordering and
duplicating the collapsed subject's reference images), a lever unique to
multi-reference generators. Each step proposes a portfolio of distinct actions;
the candidate that most rescues the collapsed subject is kept only if the
preference score does not regress and the collapsed subject's identity improves,
otherwise it is rolled back. The evaluator is used only *inside* the loop;
final quality is judged by an **independent** detection-based Subject Collapse
Rate (SCR) and human preference, avoiding self-evaluation.

Because the method touches no weights, it plugs into any multi-reference
generator on day one. On OmniGen2 over strong-interaction, occlusion-heavy
multi-subject prompts, preliminary results on a 20-task 4-subject hard slice
show our correction reduces mean SCR from 0.563 (retrained SOTA, UMO) and 0.525
(compute-matched best-of-N=8) to **0.488** at equal compute, while improving
mean DINOv2 identity similarity—i.e., a training-free loop that **matches and
slightly exceeds a retrained baseline on the same base model**. A full 500-task,
multi-seed study with bootstrap significance and blind human evaluation is
underway to validate this at scale, together with a scaling study on a newer
base (FLUX.2) at 6/8 subjects. Ablations indicate that **per-subject
diagnosis** (promoting the identity scorer from a passive judge into the loop)
and **calibrated facet routing** are both essential; uncalibrated argmin
degenerates to always targeting the same facet, and without per-subject
targeting the correction loop rarely fires.

We position this as distinct from (i) noise-level, scalar-verifier
inference-time scaling, (ii) open-loop, single-pass training-free
personalization (FreeCus/FreeGraftor), and (iii) retraining-based identity
repair (UMO/MultiCrafter): to our knowledge this is the first demonstration
that interaction-induced multi-subject identity collapse is **diagnosable and
repairable at test time, training-free, at the semantic level**, closing much
of the gap to retraining at zero training cost.

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
1. **By 2026-07-21**: submit the abstract above (it's complete, no placeholders) + pick primary/secondary topics + nominate a reciprocal reviewer.
   - **Title**: `Training-Free Test-Time Repair of Multi-Subject Identity Collapse via a Dual-Signal Decomposed Verifier`
   - **Primary topic**: `CV: Diffusion & Generative Models for Vision`
   - **Secondary topics** (pick 3–4):
     - `ML: Reasoning & Test-Time Compute` (strongly recommended — training-free test-time correction = test-time compute framing)
     - `CV: Object Detection, Segmentation & Scene Understanding` (SCR uses Grounding-DINO + DINOv2)
     - `CV: Language, Vision & Multi-modal` (reference-set manipulation + facet-specific prompt rewrite)
     - `ML: Evaluation, Benchmarking, Datasets & Analysis` (optional — only if Round 2 human-eval + bootstrap significance are solid; AAAI warns extra secondaries can attract harsher reviews)
2. **By 2026-07-28**: write the full 7-page paper using AAAI-27 LaTeX kit, anonymized, with Round-2 final numbers (500-task means + 95%CI + p-values + human-eval win rates) replacing the preliminary `0.563/0.525/0.488` in the abstract.
3. The abstract submitted on 7/21 **can** be lightly edited until 7/28 if Round-2 numbers shift the framing, but should not be substantively rewritten (AAAI may reject papers that change abstracts substantially).

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
