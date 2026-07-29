# MIE validation against human annotation — final report

**Question**: does the trained MIE evaluator generalize to unseen generators and
unseen reference sources, and does it agree with human judgement?

**Short answer**: yes on **existence** and on **score calibration**, both cleanly.
No on **appearance** (the annotations cannot test it). And one metric that looks
like strong support must not be used — details in §2.

> Run on the **complete** annotation set (2026-07-27): both batches are now a full
> 648 items × 2 annotators, 1296 votes each. The earlier run used a partial export
> (pp2 was missing one annotator entirely and 208 of its items had a single vote).
> Every number below is the complete-data value. §7 records what moved.

---

## 1. What was run

| | |
|---|---|
| Evaluator | MIE, `unsloth/Qwen3.5-4B`, `lora_layer`, checkpoint `20260503_045230/…-lora_layer-best` |
| Weight-load check | 688 LoRA parameter tensors, 53/53 trainable-backbone tensors applied, 0 unexpected keys, both heads (`score_head`, `classification_head`) loaded — verbatim from `mie_scores.json → meta.weight_load_report` |
| Scored | PP1 study: 3 reference sources × 4 generators × 216 prompts = **2586 of 2592** (6 gaps are missing `A/flux2` images, ids 211–216) |
| Human batches | `pp1` nano_banana vs MOSAIC (1296 votes) · `pp2` Flux.2 vs GPT-Image-1.5 (1296 votes, 1284 after quality flags) — **8 annotators, every item rated by exactly 2** |
| Annotation form | existence / appearance / interaction marked 0/1 **per model**, plus a preference |

Nothing in the PP1 study is in MIE's training data — different generators,
different reference sources, different prompt pool. So this is a generalization
test, not a fit check.

**Filtering applied**: rows flagged `exclude` or `prompt_ilogical` (12 rows in pp2,
all the missing `A/flux2` images); and items where the two annotators disagree on
preference are dropped — **25 in pp1, 112 in pp2**. With exactly two annotators a
disagreement *is* a tie, and ties were already excluded from every preference
statistic, so the rule leaves those numbers untouched; it changes only the facet
correlations, and by at most 0.04. Both runs are in the repo.

Retained for analysis: **pp1 623 items, pp2 530 items**.

---

## 2. The number we must NOT report

MIE agrees with the human majority on **89.1%** of pp2 items and **99.0%** of pp1
items. That reads well and is worthless, because both head-to-heads are lopsided:

| batch | MIE agreement | "always pick the majority winner" | MIE − baseline |
|---|---|---|---|
| pp1 | 99.0% | 99.8% | **−0.8 pp** |
| pp2 | 89.1% | 92.3% | **−3.2 pp** |

**A constant predictor beats MIE on both batches.** Humans chose the same side
92.3% / 99.8% of the time, so accuracy is dominated by the class prior. If we quote
89.1% as evidence, the first reviewer to compute the baseline sinks it.

MIE is not information-free here — it recovers **24.4% of the 41 pp2 items where
humans preferred the minority model (Flux.2)**, where the constant baseline scores
0% by construction — but that is not enough to win on accuracy, and we should not
pretend otherwise. The real evidence is below.

---

## 3. Evidence that holds

### 3a. Ranking skill, measured imbalance-robustly

| batch | pair | AUC (MIE Δtotal → human preference) | 95% CI |
|---|---|---|---|
| **pp2** | Flux.2 vs GPT-Image-1.5 | **0.815** | [0.755, 0.868] |
| pp1 | nano_banana vs MOSAIC | 0.579 | [0.537, 0.616] |

AUC pins any constant predictor at exactly 0.500, so it is the honest metric under
imbalance. **0.815 with the entire CI above 0.75 is the headline result.**

pp1's near-chance AUC is not evidence against MIE: humans preferred MOSAIC on
**exactly one of 623 items**. There is essentially no signal in that batch for any
evaluator to find. pp1 is a sanity check (MIE ranks the obvious pair the obvious
way); pp2 is the actual test.

### 3b. The existence head tracks humans closely

Per-item Pearson r between MIE's facet probability and the fraction of annotators
marking that facet satisfied:

| model | facet | r | annotator κ | human | MIE | MIE − human |
|---|---|---|---|---|---|---|
| GPT-Image-1.5 | **existence** | **+0.942** | +0.924 | 0.919 | 0.929 | +0.010 |
| MOSAIC | **existence** | **+0.894** | +0.804 | 0.243 | 0.297 | +0.053 |
| nano_banana | **existence** | **+0.852** | +0.821 | 0.889 | 0.908 | +0.019 |
| nano_banana | appearance | +0.667 | +0.537 | 0.404 | 0.626 | +0.222 |
| MOSAIC | interaction | +0.534 | +0.307 | 0.102 | 0.074 | −0.028 |
| Flux.2 | existence | +0.528 | +0.808 | 0.562 | 0.829 | **+0.267** |
| GPT-Image-1.5 | interaction | +0.513 | +0.505 | 0.632 | 0.587 | −0.045 |
| Flux.2 | interaction | +0.497 | +0.456 | 0.355 | 0.336 | −0.018 |
| Flux.2 | appearance | +0.472 | +0.344 | 0.249 | 0.437 | +0.188 |
| MOSAIC | appearance | +0.472 | +0.535 | 0.063 | 0.122 | +0.059 |
| nano_banana | interaction | +0.471 | +0.482 | 0.627 | 0.595 | −0.032 |
| GPT-Image-1.5 | appearance | +0.307 | +0.186 | 0.772 | 0.841 | +0.069 |

- **existence: validated.** r 0.85–0.94 on three of four generators, on the facet
  annotators agree about most strongly (κ 0.80–0.92). A constant predictor cannot
  produce an absolute per-item correlation at all.
- **appearance: not testable with these annotations.** r is low, but annotator κ is
  0.19–0.34 in pp2 — the humans do not agree with each other, which caps any
  attainable correlation. State this as a limit of the annotation protocol, not as a
  measured MIE failure, and not as an excuse either.
- **interaction: moderate** (r 0.47–0.53, κ 0.31–0.51), consistent with a harder
  judgement for both sides.

### 3c. The score margin is calibrated

Agreement with the human majority, bucketed by MIE's confidence:

| \|Δtotal\| | pp1 | pp2 |
|---|---|---|
| < 0.5 | 90.5% (n=42) | 82.5% (n=314) |
| 0.5 – 1.5 | 99.1% (n=224) | 98.5% (n=197) |
| > 1.5 | 100.0% (n=357) | 100.0% (n=19) |

**Monotonic on two independent batches.** When MIE is confident it is right; when it
hedges, so do the annotators. A constant predictor has no margin and cannot exhibit
this, which makes it — with 3b — the strongest support in the study: the *magnitude*
of the score is meaningful, not just its sign.

### 3d. No decay with subject count

| level | pp1 | pp2 |
|---|---|---|
| 2 | 100.0% (n=147) | 88.4% (n=121) |
| 4 | 98.7% (n=158) | 85.0% (n=127) |
| 6 | 98.1% (n=157) | 89.8% (n=137) |
| 8 | 99.4% (n=161) | **92.4%** (n=145) |

Flat, and *best* at 8 subjects on the informative batch. MIE's training set skews to
lower subject counts, so this is a genuine generalization result: alignment does not
degrade as scenes get crowded — the regime the benchmark is about.

### 3e. Real photographic references work as well as synthetic ones

Same 216 prompts, three reference sources, on the unseen-generator batch (pp2):

| reference source | n | AUC | 95% CI | agreement | existence r (GPT-Image-1.5) |
|---|---|---|---|---|---|
| **A — real photographs** | 173 | **0.827** | [0.706, 0.921] | 91.9% | +0.936 |
| B — GPT-Image | 176 | 0.851 | [0.760, 0.929] | 91.5% | +0.935 |
| C — Qwen-Image | 181 | 0.775 | [0.681, 0.864] | 84.0% | +0.968 |

All three CIs clear 0.5, and **A ≈ B**. MIE's human alignment does not depend on the
references being synthetic and self-consistent.

(On pp1 the A and C AUCs are undefined — annotators were unanimous, so there is no
negative class. Agreement there is 99.5% / 99.0% / 98.5% across A/B/C.)

---

## 4. Limitations to state ourselves

1. **MIE over-credits Flux.2 existence**: 0.829 vs humans' 0.562, a **+0.267** gap,
   and the only existence correlation that falls (0.528 vs 0.85–0.94 elsewhere).
2. **MIE is lenient overall** — 8 of 12 facet biases are positive, largest on
   appearance (+0.07 to +0.22). Rankings survive (roughly a monotone shift) but
   absolute probabilities should not be read as calibrated rates.
3. **Neither batch is a close call.** pp2 is 92/8 and pp1 is 99.8/0.2, so most items
   are foregone conclusions and the AUC CI is wider than it needs to be.
4. **Appearance is unresolved**, per 3b.
5. **17% of pp2 items are dropped** as split-preference (112 of 648). With only two
   annotators per item there is no tiebreaker, so every disagreement costs an item.

---

## 5. What would strengthen this

- **A pair humans split near 50/50.** The single highest-value addition: it would
  tighten the AUC CI and test MIE where it actually matters. Judging by MIE's own
  scores, `gpt15 vs nano_banana` is the closest candidate (MIE splits it 62/38,
  consistently across all three reference groups).
- **A third annotator per item** — it would both break the 112 pp2 ties and lift
  appearance κ out of the range where nothing is measurable.
- **Appearance protocol**: tighter guidelines, to lift κ out of 0.19–0.34.

---

## 6. Files

All under `companion/MIBE Rebuttal/mie_validation/`:

| file | contents |
|---|---|
| `FINAL_REPORT.md` | this document |
| `REPORT.md` | longer version, earlier partial-data run |
| `mie_vs_human_dropsplit.json` | every number above, machine-readable (drop rule applied) |
| `mie_vs_human.json` | same, without the drop rule, for comparison |
| `anno_pp1.csv`, `anno_pp2.csv` | the complete annotation batches |
| `archive_partial/` | the earlier partial export and its results, for provenance |
| `mie_scores.json` | MIE scores for all 2586 scored cells |
| `analyze_mie_vs_human.py` | the analysis; `--drop_split` toggles the rule |

Reproduce:

```bash
python analyze_mie_vs_human.py --scores mie_scores.json \
    --pp1 anno_pp1.csv --pp2 anno_pp2.csv --drop_split \
    --out mie_vs_human_dropsplit.json
```

Runs on CPU in seconds; no GPU and no model load required.

---

## 7. What changed from the partial run

pp1 is unchanged to three decimals — the 22 added votes all confirmed an existing
single vote, so no item's majority flipped. pp2 is where the complete data lands,
and **every change is in our favour**:

| | partial | complete |
|---|---|---|
| pp2 votes / annotators | 1048 / 7 | **1296 / 8** |
| pp2 items with a single vote | 208 | **0** |
| **pp2 AUC** | 0.765 [0.703, 0.822] | **0.815 [0.755, 0.868]** |
| pp2 class imbalance | 89.8 / 10.2 | 92.3 / 7.7 (**harder**) |
| GPT-Image-1.5 existence r | +0.931 (κ 0.901) | **+0.942** (κ 0.924) |
| Flux.2 appearance κ | 0.259 | 0.344 |
| GPT-Image-1.5 appearance κ | 0.153 | 0.186 |
| pp2 margin buckets | 79.3 / 97.4 / 100 | **82.5 / 98.5 / 100** |
| pp2 real-reference AUC (group A) | 0.790 | **0.827** |

The imbalance line is the one to notice: the complete batch is *more* one-sided than
the partial one (the constant baseline rises 89.8% → 92.3%), and MIE's ranking AUC
went **up** anyway. Adding data made the task harder and the result stronger, which
is the opposite of what a fitting artifact does.
