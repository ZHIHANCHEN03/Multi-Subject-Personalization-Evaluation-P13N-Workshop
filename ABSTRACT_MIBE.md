# MIBE: Multi-subject Interaction Benchmark and Evaluator for Personalized Image Generation

> AAAI-27 (2027) abstract registration (due 2026-07-21). Backup slot for the
> MIBE benchmark+evaluator paper in case NeurIPS Datasets & Benchmarks rejects.
>
> **⚠️ DUAL-SUBMISSION WARNING**: If MIBE is currently under review at NeurIPS
> D&B, you CANNOT submit this abstract to AAAI — AAAI requires you to attest
> that "neither this manuscript nor a substantially similar version of it is
> currently under review at another archival venue." You must either (a)
> withdraw from NeurIPS before submitting to AAAI, or (b) wait for the NeurIPS
> decision and only submit to AAAI if rejected. Registering the abstract locks
> title + topics + reciprocal reviewer on 7/21 AoE; the full paper is due 7/28.
> If NeurIPS accepts, simply abandon this AAAI slot (no penalty for withdrawing
> an abstract before the full-paper deadline).
>
> The abstract body below is complete and submittable as-is: concrete numbers
> (60K / 4,020 / 95.1% / 0.922 / 0.982 / 0.884), no placeholders. Claim and
> collision boundaries are locked (see PLAN.md §二, §六).

## TL;DR (for OpenReview TL;DR field)

MIBE: a 60K-pair VLM-labeled + 4K-pair human-labeled multi-subject interaction benchmark and a dual-head reference-conditioned evaluator (MIE) reaching 0.922 human-alignment, exposing that standard metrics collapse as subject count grows.

## Abstract

Multi-subject personalized image generation requires rendering all requested
reference identities *and* their specified interactions. State-of-the-art
generators still frequently omit subjects, leak appearance across identities,
or misattribute interactions—and existing metrics, designed for single-subject
fidelity or text-only composition, degrade toward random agreement with human
preference as the subject count grows (on our gold set, HPS v2.1 reaches only
0.520 pairwise agreement and PickScore falls below random at 0.486). We argue
this is a *binding* problem with three coupled dimensions—Existence,
Appearance, Interaction—and that measuring and repairing it requires
factorized, reference-grounded supervision rather than holistic aesthetics.

We introduce **MIBE**, a unified framework pairing a controlled benchmark
(**MIB**) with a learned evaluator (**MIE**). MIB decouples benchmark
construction from evaluator learning through a hierarchical prompt regime:
15K Level-8 seeds are reduced into strict Level-6/4/2 subsets by removing
entities together with their relations, so scene semantics stay approximately
invariant while entity-relation density varies. The prompt space is factorized
into 36 buckets across subject count (2/4/6/8), human/non-human ratio, and
relation type (non-contact / occlusion / physical interaction). MIB comprises
a **60K-pair Silver Set** labeled by a dual-VLM consensus (Gemini-2.5-Flash +
Gemini-3.1-Flash-Lite) under a structured SOP, reaching 95.1% cross-judge
preference agreement, and a **4,020-pair Gold Set** double-blind human-labeled
across six generators (GPT-Image-1.5, FLUX.2, Seedream 4.5, GLM, Nano Banana,
MOSAIC). To demonstrate the benchmark's utility we train **MIE**, a
reference-conditioned evaluator with a dual head: a ranking head for pairwise
preference and a diagnostic head over Existence/Appearance/Interaction, jointly
optimized so the scalar score grounds in concrete binding failures rather than
superficial aesthetics. Trained only on Silver and evaluated on Gold, MIE
achieves 0.922 overall pairwise accuracy against human preference—0.982 on
seen generators and 0.884 on unseen generators—outperforming all baselines
including CLIP, DINO, SigLIP, HPS, PickScore, ImageReward, and PSNR. Through
cross-generator meta-evaluation we surface systematic failure patterns:
existing metrics collapse as subject count increases, Existence and Appearance
failures are coupled (a missing subject redistributes visual features onto
survivors), and Interaction is the hardest dimension and the primary driver of
identity deformation. MIBE provides the data, labels, and evaluator needed to
benchmark and guide future alignment of multi-subject personalized generators.

## Contributions

- **MIB**, a controlled reference-conditioned benchmark that factorizes
  subject count, human/object ratio, and relation type, with a 60K-pair
  dual-VLM-consensus Silver Set (95.1% agreement) and a 4,020-pair
  double-blind human-labeled Gold Set across six state-of-the-art generators.
- **MIE**, a reference-conditioned dual-head evaluator that jointly learns
  pairwise ranking and structured Existence/Appearance/Interaction
  diagnostics, achieving 0.922 overall human-alignment (0.982 seen / 0.884
  unseen) while remaining interpretable, beating a broad spectrum of baseline
  metrics that collapse under multi-subject binding.
- **Cross-generator meta-evaluation** revealing that standard metrics degrade
  toward random agreement as subject count grows, that Existence–Appearance
  failures are coupled, and that Interaction is the dominant driver of identity
  deformation—providing actionable targets for future model alignment.

## AAAI-27 Submission Checklist (notes, not for submission)

**Key deadlines (all 11:59 PM UTC-12 / "anywhere on Earth")**:
- **2026-07-21**: Abstracts due — full title + complete abstract, no placeholders
- **2026-07-28**: Full papers due
- **2026-07-31**: Supplementary material + code due
- **2026-09-24**: Phase 1 rejection notification
- **2026-10-19 to 10-25**: Author feedback window
- **2026-11-30**: Final acceptance/rejection notification
- **2026-12-14**: Camera-ready files due
- **2027-02-16 to 02-23**: Conference (Montréal, Canada)

**Format**:
- AAAI-27 official LaTeX author kit, two-column, US Letter, Type 1/TrueType fonts
- Main paper: up to **7 pages** of content; pages 8–9 reserved **exclusively for references**
- Anonymized for double-blind review
- PDF only at submission; source files required only if accepted

**OpenReview portal**: `https://openreview.net/group?id=AAAI.org/2027/Conference`

**⚠️ Action items for this paper**:
1. **By 2026-07-21** (only if NeurIPS has rejected or been withdrawn):
   submit the abstract above + pick primary/secondary topics + nominate reciprocal reviewer.
   - **Title**: `MIBE: Multi-subject Interaction Benchmark and Evaluator for Personalized Image Generation`
   - **Primary topic**: `CV: Diffusion & Generative Models for Vision`
   - **Secondary topics** (pick 3):
     - `ML: Evaluation, Benchmarking, Datasets & Analysis` (strongly recommended — benchmark+evaluator paper)
     - `CV: Object Detection, Segmentation & Scene Understanding` (Existence/Appearance grounding)
     - `CV: Language, Vision & Multi-modal` (reference-conditioned VLM evaluator)
2. **By 2026-07-28**: write the full 7-page paper using AAAI-27 LaTeX kit, anonymized.
   The NeurIPS D&B paper can be adapted to AAAI format; trim the dataset-hosting/Croisson
   appendix material (AAAI is not a D&B track) and emphasize the methodological contribution
   (dual-head MIE + diagnostic supervision) and the meta-evaluation findings.
3. **Reciprocal reviewer**: nominate a different author than the MISC paper's nominee
   (same author can be nominated on both papers, but they'd carry up to 12 reviews — spread the load).

## Notes (not for submission)

- Numbers `[60K / 4,020 / 95.1% / 0.922 / 0.982 / 0.884]` are from the current MIBE
  manuscript and `paper_data` summaries. Verify against the latest
  `section_4_2_2_mie_alignment/mie_overall_metrics.csv` before submission.
- Best MIE variant: `qwen35_4b_lora_layer` (overall 0.922, seen 0.982, unseen 0.884, macro-F1 0.818).
- Strongest baseline: `MIE-4B-LoRA` at 0.884; worst: `PSNR` at 0.399.
- Gold retained-pair rate after preference-consistency filter: 94.1% (v10) / 90.4% (v13).
- Silver preference agreement rises with subject count: 91.1% (2 subjects) → 98.1% (8 subjects).
- Collision boundary with the MISC (Paper 2) submission: MIBE is the *asset* (benchmark +
  evaluator); MISC is the *application* (training-free test-time repair using MIE as controller).
  They are distinct papers with distinct claims — MIBE measures the problem, MISC repairs it.
  State this relationship explicitly if both are submitted to AAAI-27.


