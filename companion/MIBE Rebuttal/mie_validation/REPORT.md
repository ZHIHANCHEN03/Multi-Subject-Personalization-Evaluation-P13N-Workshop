# MIE vs. human annotators — generalization & alignment

Validation of the trained MIE evaluator against two fresh human-annotation batches
on the PP1 reference-extension study (216 prompts × 3 reference sources, none of
which MIE was trained on).

- **MIE checkpoint**: `unsloth/Qwen3.5-4B`, `lora_layer`, `20260503_045230/…-best`
- **Batches**: `pp1` = nano_banana vs MOSAIC (1274 votes, 8 annotators);
  `pp2` = Flux.2 vs GPT-Image-1.5 (1040 votes after quality filtering, 7 annotators)
- **Reproduce**: `python MIE_Inference/analyze_mie_vs_human.py --scores … --pp1 … --pp2 …`

Annotators marked existence / appearance / interaction as 0/1 **for each model
separately**, not just a preference. That is what makes real validation possible:
absolute per-facet ratings on the same three axes MIE emits.

---

## Headline: AUC 0.765 on the informative pair

| batch | pair | AUC (Δtotal → human preference) | 95% CI |
|---|---|---|---|
| pp2 | Flux.2 vs GPT-Image-1.5 | **0.765** | [0.703, 0.822] |
| pp1 | nano_banana vs MOSAIC | 0.579 | [0.537, 0.616] |

AUC, not agreement rate, because **both head-to-heads are lopsided** (humans pick
the winner 89.8% / 99.8% of the time). Under that imbalance a predictor that always
names the majority winner scores near-perfectly while carrying zero information;
AUC pins any constant predictor at exactly 0.500.

pp1's AUC is near chance only because there is almost nothing to predict — humans
were unanimous on 601 of 623 items and preferred MOSAIC on exactly **one**. That
batch cannot discriminate a good evaluator from a constant one, in either direction.

### The trap we are explicitly not walking into

| batch | MIE agreement | always-pick-majority | MIE − baseline |
|---|---|---|---|
| pp1 | 99.0% | 99.8% | **−0.8 pp** |
| pp2 | 86.3% | 89.8% | **−3.5 pp** |

**Raw preference agreement must not be quoted as evidence for MIE** — a constant
predictor beats it on both batches. We report it only to be complete. MIE does carry
minority-class signal (it recovers 21.4% of the 56 items where humans preferred
Flux.2, where the constant baseline is 0% by construction), but not enough to win on
accuracy. The evidence for MIE is in the two sections below, neither of which a
constant predictor can produce at all.

---

## Evidence 1 — the existence head is strongly aligned

Per-item Pearson r between the fraction of annotators marking a facet satisfied and
MIE's probability for that facet, with annotator reliability alongside:

| model | facet | r | annotator κ | MIE−human bias |
|---|---|---|---|---|
| GPT-Image-1.5 | existence | **+0.921** | +0.868 | +0.014 |
| MOSAIC | existence | **+0.900** | +0.820 | +0.050 |
| nano_banana | existence | **+0.855** | +0.814 | +0.019 |
| Flux.2 | existence | +0.498 | +0.825 | **+0.234** |
| nano_banana | appearance | +0.675 | +0.564 | +0.216 |
| MOSAIC | interaction | +0.538 | +0.321 | −0.033 |
| MOSAIC | appearance | +0.517 | +0.540 | +0.054 |
| GPT-Image-1.5 | interaction | +0.502 | +0.537 | +0.012 |
| Flux.2 | appearance | +0.492 | +0.221 | +0.175 |
| Flux.2 | interaction | +0.463 | +0.501 | −0.005 |
| nano_banana | interaction | +0.458 | +0.463 | −0.023 |
| GPT-Image-1.5 | appearance | +0.283 | +0.180 | +0.098 |

Read this table by facet, not by row order:

- **existence — validated.** r 0.86–0.92 on three of four models, against the facet
  annotators agree on most strongly (κ 0.81–0.87). This is the claim the data
  supports cleanly.
- **appearance — cannot be validated here.** r is low, but so is annotator κ
  (0.18–0.22 in pp2): the humans do not agree with each other, which caps any
  achievable correlation. This is a limit of the annotation task, not a measured
  failure of MIE — and it must be stated that way rather than spun either
  direction.
- **interaction — moderate** (r ≈ 0.46–0.54, κ ≈ 0.32–0.54), consistent with a
  genuinely harder judgement.

**The one real miss: Flux.2 existence.** MIE scores it 0.852 where humans give
0.617 — a +0.234 gap, and the only existence correlation that drops (0.498). MIE
systematically over-credits Flux.2 for keeping subjects present. Worth naming as a
limitation rather than leaving for a reviewer to find.

More broadly MIE is **lenient**: 10 of 12 biases are positive, largest on appearance
(+0.10 to +0.22). Rankings survive this (it is close to a monotone shift) but
absolute probabilities should not be read as calibrated rates.

---

## Evidence 2 — the score margin is calibrated

Agreement with the human majority, bucketed by how confident MIE was:

| \|Δtotal\| | pp1 | pp2 |
|---|---|---|
| < 0.5 | 90.5% (n=42) | 79.3% (n=338) |
| 0.5–1.5 | 99.1% (n=224) | 97.4% (n=192) |
| > 1.5 | 100.0% (n=357) | 100.0% (n=17) |

**Monotonic in both batches, independently.** When MIE is confident it is right;
when it is unsure, so are the humans. A constant predictor has no margin and cannot
produce this pattern, which makes it the strongest available evidence that the
score magnitude — not just its sign — carries information.

---

## Evidence 3 — no degradation with subject count

Preference agreement by number of subjects:

| level | pp1 | pp2 |
|---|---|---|
| 2 | 100.0% | 86.3% |
| 4 | 98.7% | 80.5% |
| 6 | 98.1% | 87.1% |
| 8 | 99.4% | 90.9% |

Flat, or slightly *better* at 8 subjects. Since MIE's training set is dominated by
lower subject counts, this is a real generalization result: alignment does not decay
as scenes get crowded.

---

## What this does and does not establish

**Supports**

1. MIE's **existence** head generalizes to unseen generators and unseen reference
   sources, correlating r ≈ 0.86–0.92 with reliable human labels.
2. MIE's **score margin is calibrated** — agreement rises monotonically with
   confidence, on two independent batches.
3. Alignment **does not degrade with subject count** (2 → 8).

**Does not support**

1. Any claim built on raw preference agreement — the majority-class baseline beats
   MIE on both batches.
2. Validation of the **appearance** head — annotator κ of 0.18–0.22 leaves no
   headroom to measure against.
3. Calibration of absolute probabilities — MIE is systematically lenient, and
   markedly so for Flux.2 existence (+0.234).

**Would settle the open questions**: a pair where humans split closer to 50/50
(both batches here are ≥ 90/10, so most items are foregone conclusions), and an
appearance protocol with tighter guidelines or more annotators per item to lift κ
into a usable range.

---

## Files

| file | contents |
|---|---|
| `mie_vs_human.json` | every number above, machine-readable |
| `anno_pp1.csv` / `anno_pp2.csv` | the annotation batches as analyzed |
| `../../../MIE_Inference/results/mie_scores.json` | MIE scores, 2586 of 2592 cells |
| `../../../MIE_Inference/analyze_mie_vs_human.py` | the analysis |
