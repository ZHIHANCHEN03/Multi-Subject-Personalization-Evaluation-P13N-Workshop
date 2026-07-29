# Response to Reviewer yNv5

*(Draft. W2 and W3 are answered with new data; W1 is not yet — see below.)*

We thank the reviewer for identifying the evaluator's weakest evidential point. Since
submission we ran a **reference-extension study** designed around exactly W2 and W3:
the same 216 structured prompts re-rendered under **3 reference sources × 4 generators**
(2586 of 2592 cells scored), using **22 subjects with zero overlap against MIB's 80**,
with **two fresh human-annotation batches (2592 votes, 8 annotators, every item rated
by exactly two)**.

**A note on how we report agreement.** Both the Gold Set and the new batches are
heavily one-sided head-to-heads. On the Gold Set a predictor that always names the
majority winner scores 0.961 overall — *above* MIE's 0.921. Raw pairwise accuracy is
therefore dominated by the class prior and we do not rely on it below. We report AUC of
MIE's score margin against the human preference, which pins any constant predictor at
exactly 0.500, and per-facet correlation against human marks, which a constant
predictor cannot produce at all. **We will re-state the paper's headline numbers in
this form in the revision.**

---

## W3 — "A single training matchup … evidence for unseen generalization is limited"

> *All 60K training pairs are Nano Banana vs. MOSAIC. MIE scores 0.982 on seen but
> drops ~0.10 to 0.884 on unseen. With training this narrow, the evidence that unseen
> generalization extends to a broader range of generator families is limited.*

We now have three independent lines of evidence, and they converge.

### 1. The 0.982 → 0.884 drop is largely an imbalance artifact, not a loss of ranking skill

The two splits differ in how one-sided they are: the seen split is 98.8 % one-sided,
the unseen split 94.5 %. Accuracy therefore falls even if ranking skill is unchanged.
Measured per matchup with AUC, on the Gold Set (n = 3599 usable pairs):

| matchup | split | n | AUC | 95 % CI |
|---|---|---|---|---|
| MOSAIC vs nano_banana | **seen** | 1365 | 0.865 | [0.765, 0.937] |
| GLM vs Seedream 4.5 | **unseen** | 1186 | **0.891** | [0.832, 0.943] |
| Flux.2-klein-9B vs GPT-Image-1.5 | **unseen** | 1048 | 0.764 | [0.717, 0.808] |

**The two unseen matchups bracket the seen one.** One is above it, one below, and the
confidence intervals overlap. There is no systematic seen → unseen degradation in
ranking skill; what the accuracy gap mostly measures is that the two splits have
different class priors. (We report per-matchup rather than pooled AUC: pooling
matchups of different difficulty inflates the statistic.)

We would also note that the evaluation's unseen coverage is wider than the review
assumes: v13 contains **four unseen generators in two matchups** — Flux.2-klein-9B,
GPT-Image-1.5, GLM and Seedream 4.5 — spanning a step-distilled flow model, two closed
API models and an open-weight model. The *training* matchup is indeed single; the
*evaluation* is not.

### 2. The same matchup replicates on entirely new subjects and new references

The new study re-runs the `Flux.2-klein-9B vs GPT-Image-1.5` matchup on 22 subjects
disjoint from MIB's 80, with references from three independent sources, scored by
eight fresh annotators:

| | Gold Set v13 | reference-extension study |
|---|---|---|
| matchup | Flux.2-klein-9B vs GPT-Image-1.5 | identical |
| subjects | MIB's 80 | **22, zero overlap** |
| references | MIB's | 3 sources, incl. **real photographs** |
| annotators | Gold protocol | 8 fresh, 1296 votes |
| **AUC** | 0.764 [0.717, 0.808] | **0.815 [0.755, 0.868]** |

Changing the subjects and the references does not degrade alignment — the point
estimate rises and the intervals overlap. This is a replication of the unseen result
under a near-total change of inputs, not a fresh claim.

Two further properties from the new batch that memorisation of a training matchup
cannot produce:

- **Alignment does not decay with subject count.** On the Gold Set unseen split, AUC by
  subject count is 0.744 / 0.862 / 0.870 / **0.911** for 2 / 4 / 6 / 8 — *rising*, and
  best at 8, where MIE's training data is thinnest and the benchmark matters most. The
  new batch agrees (agreement 88.4 / 85.0 / 89.8 / 92.4 %).
- **The score margin is calibrated.** Agreement runs 82.5 % → 98.5 % → 100 % across
  |Δtotal| < 0.5 / 0.5–1.5 / > 1.5, monotonically, on two independent batches. When MIE
  is confident it is right; when it hedges, so do the annotators.

### 3. The generalization gap shrinks with evaluator capacity

If the gap were a structural consequence of the narrow training matchup, scaling the
evaluator should not systematically close it. It does:

| evaluator | seen (v10) | unseen (v13) | gap |
|---|---|---|---|
| Qwen3.5-0.8B + LoRA | 0.987 | 0.780 | −0.206 |
| Qwen3.5-2B + LoRA | 0.983 | 0.854 | −0.129 |
| **Qwen3.5-4B + LoRA** | 0.982 | **0.884** | **−0.098** |

Seen accuracy is saturated across all three sizes, so the entire narrowing comes from
the unseen side rising (0.780 → 0.854 → 0.884). A substantial part of the gap is
evaluator capacity, not a ceiling imposed by the training distribution.

---

## W2 — "The 'ground truth' is itself VLM-generated imagery"

> *Reference subjects are GPT-Image-1 generations … the reference identity is synthetic
> and may not be self-consistent, injecting hard-to-bound noise into the Appearance
> dimension.*

We agree this needed testing rather than argument. The extension study re-runs the
identical 216 prompts against **three reference sources**, one of which is **licensed
real photographs** (22 subjects, downsampled to 512). On the unseen-generator batch:

| reference source | n | AUC | 95 % CI | existence r (GPT-Image-1.5) |
|---|---|---|---|---|
| **A — real photographs** | 173 | **0.827** | [0.706, 0.921] | +0.936 |
| B — GPT-Image | 176 | 0.851 | [0.760, 0.929] | +0.935 |
| C — Qwen-Image | 181 | 0.775 | [0.681, 0.864] | +0.968 |

**Real references perform as well as synthetic ones** — all three intervals clear 0.5,
A and B are statistically indistinguishable, and the highest per-facet correlations are
split across A and C. MIE's human alignment does not depend on the references being
synthetic and self-consistent, so the noise channel the review identifies is not what
is carrying the result. The generator ordering is also identical under all three
reference sources.

**Where the review is right, and we will say so.** The reviewer's concern is
specifically about the **Appearance** dimension, and appearance is the one facet these
annotations cannot validate: inter-annotator κ on appearance is 0.19–0.34, i.e. the
humans do not agree with each other, which caps any attainable correlation. We will
report this as a limitation of the annotation protocol rather than claim the appearance
head is validated. Existence, by contrast, is the facet annotators agree on most
strongly (κ 0.80–0.92) and there MIE correlates at **r = +0.94** with human marks on
subjects it has never seen.

---

## W1 — "The evaluator and its teacher share a source"

> *The Silver Set is annotated by Gemini-2.5/3.1. MIE is trained on those labels. The
> paper repeatedly claims MIE far exceeds a "VLM-based scorer (Qwen-7B)" baseline, but
> that comparison is unfair.*

**We accept this criticism as stated.** A baseline derived from the same teacher that
produced MIE's training labels cannot establish what we used it to establish, and we
should not have leaned on it.

The right correction is not an argument but a change of anchor: **human preference, not
the teacher, should be the reference against which both are measured.** We commit to
reporting MIE and the Gemini teacher side by side as predictors of the *human* labels —
on the Gold Set and on the new batches — using the same imbalance-robust metric (AUC)
applied to both. The human labels are already collected and independent of both
systems, so this comparison is fair by construction and requires no new annotation. We
will include it in the revision, and we will report the result whichever way it falls.
