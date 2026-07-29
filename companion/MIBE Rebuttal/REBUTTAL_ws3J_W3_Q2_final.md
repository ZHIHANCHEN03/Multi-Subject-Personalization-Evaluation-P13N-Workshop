# ws3J — W3 / Q2 (final rebuttal text)

**Reviewer point.** *The number of reference subjects is also fairly limited, given that
identity-preservation is a key element the benchmark. It might be useful to see how well
identity preserving extends for more obscure, long-tailed classes; What are the classes
present in MIB and does MIE extend to unseen classes?*

---

We thank the reviewer for these two linked points: they ask what MIB actually contains,
and whether the evaluator's behaviour survives leaving that inventory. We answer them
together, because since submission we built a study specifically for this, and it changes
what we are able to claim.

## 1. The classes present in MIB

MIB uses a fixed pool of **80 reference subjects — 30 human and 50 non-human**. Human
references span age groups, gender presentations, ethnicities, clothing styles and
hairstyles; non-human references cover animals, furniture, tools, wearable items and
common manipulable objects, which is what lets MIB test person–person, person–object and
object–object binding rather than identity preservation alone. The full inventory:

| | n | subjects |
|---|---|---|
| **human** | 30 | adult_man_beard, adult_woman_curly_hair, elderly_black_man, elderly_black_woman, elderly_east_asian_man, elderly_east_asian_woman, elderly_white_man, elderly_white_woman, man_black_suit, man_bomber_jacket, man_denim_jacket, man_flannel_shirt, man_sportswear, middle_eastern_man, middle_eastern_woman_hijab, middleaged_black_man, middleaged_black_woman_short_hair, south_asian_man, south_asian_woman, teen_boy, teen_girl, woman_green_cardigan, woman_hijab_coat, woman_red_hoodie, woman_white_blazer, woman_yellow_dress, young_east_asian_man, young_east_asian_woman, young_white_man, young_white_woman |
| **non-human** | 50 | acoustic_guitar, apple, black_cat, blue_bicycle, bread_loaf, brown_horse, burger, deer, donut, dslr_camera, duck, elephant, folding_umbrella, fried_chicken, giraffe, goat, golden_retriever, green_salad, headphones, helmet, ice_cream, koala, lion, open_silver_laptop, orange_basketball, owl, panda, parrot, penguin, pizza_slice, rabbit, red_backpack, red_fox, sheep, skateboard, soccer_ball, spaghetti, succulent_plant, sushi_set, table_lamp, tennis_racket, tiger, toolbox, travel_suitcase, tripod, watering_can, white_mug, wolf, wooden_dining_chair, zebra |

We agree with the reviewer that 80 subjects is a limited inventory for a benchmark whose
central axis is identity preservation. It is also worth being precise about which axis
the Gold Set varies and which it does not. It tests **unseen generators** — v13 contains
**four generators absent from training, in two matchups** (Flux.2-klein-9B,
GPT-Image-1.5, GLM, Seedream 4.5) — but it holds the reference inventory fixed. The
reviewer is asking about the other axis, and the submitted version did not test it.

## 2. The reference-extension study

We therefore varied that axis. The same 216 structured prompts were re-rendered against
**three independent reference sources** — **A: licensed real photographs**, B: GPT-Image,
C: Qwen-Image — for four generators, yielding **2586 of 2592 scored cells**, with two
fresh human-annotation batches providing the reference judgement: **2592 votes, 8
annotators, every item rated by exactly two**. The evaluator is the **identical
checkpoint** reported in the paper; nothing was retrained, so this measures
generalization and not adaptation.

The study adds **22 subjects with zero overlap against MIB's 80** — verified by slug
comparison against the released manifests rather than assumed:

| | n | subjects |
|---|---|---|
| **human** | 10 | black_teen_girl_braids, east_asian_man_leather_jacket, elderly_black_woman_headwrap, elderly_south_asian_man, latino_man_black_polo, man_orange_turtleneck, middle_eastern_teen_boy, middleaged_white_woman_glasses, southeast_asian_woman, woman_pink_hair |
| **non-human** | 12 | alpaca, croissant, cupcake, dairy_cow, desk_globe, electric_kettle, flamingo, hamster, picnic_basket, shopping_cart, turtle, violin |

This takes the combined inventory to **102 subjects (40 human / 62 non-human)**.

One distinction matters for the reviewer's question, and we would rather draw it
ourselves than have it read into the numbers. The two halves are not "unseen" in the
same sense. For the **12 non-human subjects the categories themselves are new** — MIB
contains no camelid, no wading bird, no stringed instrument, no basket, no kettle. For
the **10 human subjects these are new identities within demographic descriptions
adjacent to MIB's** (for instance `middle_eastern_teen_boy` against MIB's
`middle_eastern_man` and `teen_boy`). We do not describe all 22 as unseen categories.

## 3. Does MIE extend to unseen classes? Yes, and changing the references does not degrade it

We report AUC of MIE's score margin against human preference rather than raw pairwise
agreement. Both the Gold Set and these batches are heavily one-sided head-to-heads, so
accuracy is dominated by the class prior — on the Gold Set a predictor that always names
the majority winner scores 0.961 — whereas AUC pins any constant predictor at exactly
0.500. On the unseen-generator batch (Flux.2-klein-9B vs GPT-Image-1.5), every item
involving only unseen subjects:

| reference source | n | AUC [95 % CI] | existence correlation with human marks |
|---|---|---|---|
| **A — real photographs** | 173 | **0.827** [0.706, 0.921] | **+0.932** |
| B — GPT-Image | 176 | 0.851 [0.760, 0.929] | +0.926 |
| C — Qwen-Image | 181 | 0.775 [0.681, 0.864] | +0.938 |

| subjects per prompt (pooled over A/B/C) | n | AUC [95 % CI] |
|---|---|---|
| 2 | 121 | 0.828 [0.714, 0.919] |
| 4 | 127 | 0.749 [0.625, 0.862] |
| 6 | 137 | 0.813 [0.708, 0.899] |
| **8** | 145 | **0.866** [0.739, 0.984] |

Three things follow. Every interval clears 0.5, so ranking skill survives on subjects the
evaluator has never seen. **Real and synthetic references are statistically
indistinguishable**, and the per-facet existence correlation is essentially invariant to
the reference source (+0.93 under all three). And **alignment does not decay with subject
count — it is highest at eight subjects**, the regime the benchmark exists for and where
MIE's training data is thinnest.

On the lopsided batch (nano_banana vs MOSAIC) annotators were near-unanimous, leaving no
negative class for AUC; we report agreement only there: 99.5 / 99.0 / 98.5 % for
A / B / C.

*We give these two marginals rather than the full 3 × 4 grid deliberately: individual
cells hold 40–52 items with only 4–14 minority-class labels and their intervals reach
[0.28, 0.93]. They are not informative and we do not quote them.*

## 4. On "more obscure" classes specifically — the result was not the one we expected

A score is produced per image and each image holds 2–8 subjects, so no score belongs to
one subject alone. We therefore residualise: each image's score minus the mean of its
(subject-count × generator × reference-source) cell, attributed equally to every subject
present. A permutation test confirms the resulting spread across subjects exceeds chance
(**p = 0.0005** on both facets), so subject identity does carry signal. Positive means
*easier* than the cell average:

| | **new non-human** (12) | **new human identity** (10) | contrast [95 % CI] |
|---|---|---|---|
| Human-marked appearance | **+0.0145** [+0.008, +0.021] | −0.0145 [−0.020, −0.009] | **+0.029** [+0.021, +0.038] |
| Human-marked existence | **+0.0209** [+0.015, +0.027] | −0.0208 [−0.027, −0.015] | **+0.042** [+0.034, +0.050] |
| MIE appearance | +0.018 [+0.014, +0.022] | −0.018 [−0.022, −0.014] | +0.036 |
| MIE existence | +0.029 [+0.023, +0.035] | −0.028 [−0.035, −0.022] | +0.057 |

**Category novelty carries no penalty — it is an advantage.** The non-human categories
MIB does not cover are handled *better* than the cell average, on human annotation and on
MIE independently. Difficulty is concentrated on **human identity**: the five hardest
subjects for existence are all people (`latino_man_black_polo`, `man_orange_turtleneck`,
`middle_eastern_teen_boy`, `woman_pink_hair`, `east_asian_man_leather_jacket`) and the
three easiest are all inanimate (`desk_globe`, `violin`, `picnic_basket`). For a
benchmark built around identity preservation this is the useful finding: the binding
constraint is not category rarity but human identity.

As a by-product this is a sharper generalization test than the pairwise one. On 22
subjects it has never seen, **MIE's per-subject difficulty index reproduces the human one
at r = +0.90 for existence** (Spearman ρ = +0.92; 20 of 22 subjects on the same side of
the mean) and r = +0.71 for appearance (ρ = +0.69; 19 of 22). MIE does not merely rank
*images* consistently with humans — it recovers *which subjects are hard*, on an
inventory disjoint from its training set.

## 5. What this does not show

We do not want to convert this into a claim we have not earned. The 12 non-human
categories are novel relative to MIB, but they are not genuinely rare — a violin and a
croissant are common objects. The defensible statement is that **MIB's inventory boundary
is not a generalization boundary**, not that the long tail has been tested. A genuine
long-tail study needs rare, fine-grained categories — specific breeds, specific
landmarks, uncommon artifacts — at a scale we have not collected. We will state this as an
explicit limitation rather than describe 22 subjects as a long-tail result; the
reference-extension protocol built for this response is precisely the machinery required
to run it.

Two further caveats we state ourselves. The per-subject residual is shared equally among
co-occurring subjects, so the index is a first-order attribution rather than a causal
decomposition. And appearance is the one facet these annotations cannot adjudicate:
inter-annotator κ on appearance is 0.19–0.34, meaning the annotators do not agree with
each other, which caps any attainable correlation — a limitation of the annotation
protocol, which we report as such and do not offer as an excuse.

## What we will add to the paper

The 80-subject inventory table of §1; the reference-extension results of §3–4; and the
limitation of §5. We will also **release the real-photograph reference split (A)**, so
that future work can evaluate under the stricter setting. This study exists because of
this review point, and we are grateful for it.
