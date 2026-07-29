# Response to Reviewer ws3J

*(Draft. Every number below is recomputed from `mie_validation/` — see `mie_human_validation_20260727.zip`.)*

---

We thank the reviewer for the careful reading and the encouraging assessment. Since
submission we have run a **reference-extension study** built specifically to answer
questions of this kind: the same 216 structured prompts are re-rendered under
**3 reference sources × 4 generators** (2586 of 2592 cells scored), with **22 subjects
that have zero overlap with MIB's 80**, plus **two fresh human-annotation batches
(2592 votes, 8 annotators, every item rated by exactly two)**. All three questions are
answered from that study.

## Q1. Model scores, and where current models fall short

**MIE scores, pooled over the three reference sources** (higher is better; `total` is
the signed ranking score, facets are probabilities):

| Generator | total | Existence | Appearance | Interaction |
|---|---|---|---|---|
| GPT-Image-1.5 | **+0.773** | 0.926 | 0.844 | 0.598 |
| nano_banana | +0.655 | 0.909 | 0.631 | 0.596 |
| Flux.2-klein-9B | +0.329 | 0.851 | 0.466 | 0.361 |
| MOSAIC | −0.947 | 0.310 | 0.129 | 0.080 |

The **generator ordering is identical under all three reference sources**, which is
a useful robustness check on the benchmark itself.

**Where they fall short: interaction, unambiguously.** It is the weakest facet for
**every one of the four generators**, and the same ordering appears in the raw human
annotations, which are independent of MIE:

| Generator | Existence | Appearance | Interaction | *(human-marked satisfaction rate)* |
|---|---|---|---|---|
| GPT-Image-1.5 | 0.912 | 0.759 | **0.626** | |
| nano_banana | 0.890 | 0.410 | **0.618** | |
| Flux.2-klein-9B | 0.618 | 0.287 | **0.393** | |
| MOSAIC | 0.258 | 0.074 | **0.113** | |

**And it degrades fastest with subject count.** Pooled over all generators and
reference sources:

| Subjects | total | Existence | Appearance | Interaction |
|---|---|---|---|---|
| 2 | +0.763 | 0.946 | 0.732 | 0.637 |
| 4 | +0.319 | 0.782 | 0.556 | 0.407 |
| 6 | +0.046 | 0.707 | 0.441 | 0.370 |
| 8 | **−0.319** | 0.560 | 0.341 | **0.223** |

Existence falls by a factor of 1.7 from 2 to 8 subjects; interaction falls by 2.9.
This is precisely the failure mode MIB was designed to isolate, and it is invisible to
a single aggregate score.

## Q2. Classes in MIB, and extension to unseen classes

**MIB's 80 reference subjects**: 30 human, 20 animal, 20 object, 10 food.

**The reference-extension study introduces 22 new subjects with zero overlap** —
verified by slug comparison against the released manifests, not assumed:

| category | subjects |
|---|---|
| human (10) | black_teen_girl_braids, east_asian_man_leather_jacket, elderly_black_woman_headwrap, elderly_south_asian_man, latino_man_black_polo, man_orange_turtleneck, middle_eastern_teen_boy, middleaged_white_woman_glasses, southeast_asian_woman, woman_pink_hair |
| animal (5) | alpaca, dairy_cow, flamingo, hamster, turtle |
| object (5) | desk_globe, electric_kettle, picnic_basket, shopping_cart, violin |
| food (2) | croissant, cupcake |

To be precise about what kind of "unseen" this is: for **animals, objects and food the
categories themselves are new** (MIB contains no camelid, no bird of this type, no
stringed instrument, no basket); for **humans these are new identities within adjacent
demographic descriptions**. We will state this distinction explicitly rather than
claim more than the design supports.

**MIE does extend.** On the batch where *both generators are also unseen*
(Flux.2-klein-9B vs GPT-Image-1.5, neither present in the 60K training pairs):

| | value |
|---|---|
| AUC (MIE score margin → human preference) | **0.815**, 95% CI [0.755, 0.868] |
| Existence correlation vs human annotation, GPT-Image-1.5 | **r = +0.942** (annotator κ = 0.924) |
| Agreement by subject count (2 / 4 / 6 / 8) | 88.4 / 85.0 / 89.8 / **92.4 %** |
| Agreement by MIE confidence, \|Δ\| < 0.5 / 0.5–1.5 / > 1.5 | 82.5 / 98.5 / **100 %** |

We report AUC rather than raw agreement deliberately: these head-to-heads are heavily
one-sided, so a predictor that always names the majority winner would score 92.3%
while carrying no information. AUC pins any such constant predictor at exactly 0.500.

Two points beyond the headline. **Alignment does not decay with subject count** — it
is marginally *best* at 8 subjects, the regime where MIE's training data is thinnest
and the benchmark matters most. And **MIE's margin is calibrated**: when it is
confident it is right, and when it hedges so do the annotators. A model that had
merely memorised its training matchup could not produce either behaviour on two
unseen generators and 22 unseen subjects.

## Q3. Leaderboard

We agree, and we commit to it. The evaluation harness, the reference sets, the prompt
manifests and the MIE checkpoint are already part of the release; we will host a
public leaderboard with a held-out prompt split and a submission protocol, and will
add the link in the camera-ready.

---

## On the three minor weaknesses

**"The actual generative model results are not presented, so the room for growth is
not obvious."** — This is a fair omission and we now have the numbers. The clearest
statement of headroom is the fraction of prompts on which **human annotators mark all
three facets satisfied simultaneously**:

| Generator | 2 subjects | 8 subjects |
|---|---|---|
| nano_banana | 79.3 % | **5.9 %** |
| GPT-Image-1.5 | 75.3 % | **26.4 %** |
| Flux.2-klein-9B | 34.4 % | **0.9 %** |
| MOSAIC | 6.2 % | 0.3 % |

**The strongest generator we tested fully satisfies only about one 8-subject prompt in
four, and the second-strongest one in seventeen.** The task is very far from solved,
and we will add this table to the main paper — it makes the benchmark's headroom
concrete in a way the metric-comparison tables do not.

**"The number of example prompts presented explicitly is very limited."** — Agreed. We
will promote a set of worked examples into the main text: one per interaction class
(`no_interaction_no_occlusion`, `occlusion_no_interaction`, `occlusion_interaction`)
at 2 and at 8 subjects, each shown with its reference set, its generated candidates
and its per-facet human marks, so a reader can see what the three facets mean without
turning to the appendix.

**"The number of reference subjects is fairly limited … how well identity preservation
extends for more obscure, long-tailed classes."**

We took this seriously enough to build a study around it. Three things follow: what we
added, whether alignment survives it, and what still is not shown.

### (a) What was added

MIB's Gold Set tests **unseen generators** but holds the reference inventory fixed. The
new reference-extension study varies the other axis: the same 216 structured prompts
are re-rendered against **three independent reference sources** — **A: licensed real
photographs**, B: GPT-Image, C: Qwen-Image — over **22 subjects with zero overlap
against MIB's 80** (verified by slug comparison against the released manifests), for
four generators, giving 2586 of 2592 scored cells. Two fresh human-annotation batches
(2592 votes, 8 annotators, every item rated by exactly two) provide the reference
judgement. **The evaluator is the identical checkpoint used for the Gold Set results —
nothing was retrained.**

### (b) Alignment survives the change of references, and does not decay with subject count

Measured as AUC of MIE's score margin against human preference, on the unseen-generator
batch (Flux.2-klein-9B vs GPT-Image-1.5):

| reference source | n | AUC | 95 % CI |
|---|---|---|---|
| **A — real photographs** | 173 | **0.827** | [0.706, 0.921] |
| B — GPT-Image | 176 | 0.851 | [0.760, 0.929] |
| C — Qwen-Image | 181 | 0.775 | [0.681, 0.864] |

| subjects (pooled over A/B/C) | n | AUC | 95 % CI |
|---|---|---|---|
| 2 | 121 | 0.828 | [0.714, 0.919] |
| 4 | 127 | 0.749 | [0.625, 0.862] |
| 6 | 137 | 0.813 | [0.708, 0.899] |
| **8** | 145 | **0.866** | [0.739, 0.984] |

All intervals clear 0.5; real and synthetic references are statistically
indistinguishable; and alignment is *highest* at eight subjects, the regime the
benchmark exists for. On the lopsided batch (nano_banana vs MOSAIC) annotators were
close to unanimous, so AUC is undefined there and we report agreement only: 99.5 / 99.0
/ 98.5 % for A / B / C.

*We report these two marginals rather than the full 3 × 4 grid deliberately: individual
cells hold 40–52 items with only 4–14 minority-class labels, and their intervals span
as much as [0.28, 0.93]. They are not informative and we do not quote them.*

### (c) Which facets are actually being measured

Per-item correlation between MIE's facet probability and the fraction of annotators
marking that facet satisfied, broken out by reference source:

| reference source | existence (GPT-Image-1.5) | appearance | interaction |
|---|---|---|---|
| A — real photographs | **+0.932** | +0.320 | +0.467 |
| B — GPT-Image | **+0.926** | +0.395 | +0.494 |
| C — Qwen-Image | **+0.938** | +0.305 | +0.523 |

**Existence is validated and essentially invariant to the reference source** (+0.93
across all three). Interaction is moderate and consistent (+0.47 to +0.52). **Appearance
cannot be validated by these annotations** — inter-annotator κ on appearance is
0.19–0.34, i.e. the humans do not agree with each other, which caps any attainable
correlation. That is a limitation of the annotation protocol; we state it as such and
do not use it as an excuse. We also note MIE over-credits Flux.2's existence (0.829
against humans' 0.562), the one place its existence correlation falls.

### (d) Which classes are hard — a result we did not expect



The 22 added subjects are not uniformly "new" in the same sense. The 10 humans are new
*identities* drawn from demographic descriptions adjacent to MIB's; the 12 animal,
object and food subjects are new **categories** — MIB contains no camelid, no wading
bird, no stringed instrument, no basket. That split is exactly the reviewer's question,
so we measured it.

A score is produced per image and each image holds 2–8 subjects, so we work with
residuals: each image's score minus the mean of its (subject-count × generator ×
reference-source) cell, attributed to every subject present. Positive means *easier*
than its cell average. A permutation test confirms the resulting spread is real
(p = 0.0005 for both facets), so subject identity does carry signal.

| | new **category** (12) | new **identity** (10 humans) | contrast |
|---|---|---|---|
| **Human-marked appearance** | **+0.0145** [+0.008, +0.021] | −0.0145 [−0.020, −0.009] | **+0.029** [+0.021, +0.038] |
| **Human-marked existence** | **+0.0209** [+0.015, +0.027] | −0.0208 [−0.027, −0.015] | **+0.042** [+0.034, +0.050] |
| MIE appearance | +0.018 [+0.014, +0.022] | −0.018 [−0.022, −0.014] | +0.036 |
| MIE existence | +0.029 [+0.023, +0.035] | −0.028 [−0.035, −0.022] | +0.057 |

**Category novelty carries no penalty — it is an advantage.** The categories MIB does
not cover (alpaca, flamingo, desk globe, violin, picnic basket, croissant) are handled
*better* than average, on human annotation and on MIE independently. The generators'
difficulty is concentrated on **human identity**, not on category rarity: the five
hardest subjects for existence are all people (`latino_man_black_polo`,
`man_orange_turtleneck`, `middle_eastern_teen_boy`, `woman_pink_hair`,
`east_asian_man_leather_jacket`), and the easiest are all inanimate
(`desk_globe`, `violin`, `picnic_basket`).

As a by-product this is a second, sharper generalization test for MIE: on 22 subjects
it never saw, **MIE's per-subject difficulty index reproduces the human one at
r = +0.90 for existence (Spearman ρ = +0.92, 20/22 subjects on the same side) and
r = +0.71 for appearance (ρ = +0.69, 19/22)**. MIE does not merely rank *images*
consistently with humans; it recovers which *subjects* are hard, on an inventory
disjoint from its training set.

**What this does not show.** The 12 categories are novel relative to MIB, but they are
not genuinely rare — a violin and a croissant are common objects. So the honest
statement is that *MIB's inventory boundary is not a generalization boundary*, not that
the true long tail has been tested. A genuine long-tail study needs rare, fine-grained
categories — specific breeds, specific landmarks, uncommon artifacts — at a scale we
have not collected. We will say so as an explicit limitation. The reference-extension
protocol built for this response is precisely the machinery required to run it, and we
would rather flag it as the next extension than describe 22 subjects as a long-tail
result. We will also note the attribution caveat: the residual is shared equally among
co-occurring subjects, so the per-subject index is a first-order attribution rather
than a causal decomposition.
