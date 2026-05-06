# Paper Section 4 & 5 Analysis

This note summarizes the currently available paper-facing data under `paper_data` and converts it into section-level conclusions and figure recommendations for Sections 4 and 5.

## Overall Takeaways

- `MIB-Silver` is reliable enough to serve as scalable supervision: the two LLM judges reach `95.1%` preference agreement on `59852` matched tasks.
- `MIB-Gold` is genuinely challenging: after filtering human preference inconsistencies, the retained-pair rate is `94.1%` on `v10` and `90.4%` on `v13`, indicating more disagreement on the unseen-generator benchmark.
- Existing metrics remain far from a full human proxy: the best third-party baseline is `ArcFace Distance` at `0.873` pairwise accuracy, while the worst baseline (`PSNR`) falls to `0.399`.
- The strongest current MIE variant is `qwen35_2b_lora_layer` with overall pairwise accuracy `0.903`, seen accuracy `0.983`, unseen accuracy `0.854`, and macro-F1 `0.801`.
- The main current evaluator story is not full 6-model scaling; it is that `2B + LoRA-layer` is the strongest available operating point, especially on unseen generators.

## Section 4.1

### MIB-Silver: Label Reliability and Scalability

**Core findings**
- The two silver annotators overlap on `59,852` tasks and achieve `95.1%` preference agreement.
- Full diagnostic agreement is lower (`46.2%`), which is expected: pairwise preference is easier to align on than fine-grained error localization.
- Preference agreement increases sharply with subject count: `91.1%` at `2` subjects vs `98.1%` at `8` subjects.
- Agreement is highest for `occlusion_interaction` scenarios, suggesting that explicit interactions provide stronger grounding cues for structured judging than weaker occlusion-only cases.

**Interpretation**
- These numbers support the claim that the SOP is not merely producing noisy pseudo-labels. Instead, it yields a stable pairwise supervision signal that becomes even more decisive in dense multi-subject scenes where failures are more catastrophic.
- The fact that full diagnostic agreement is substantially lower than preference agreement is also useful: it shows the three diagnostic dimensions capture genuinely difficult distinctions rather than trivial labels.

**Recommended figure**
- `paper_data/images/section_4_1_silver_agreement.png`

**Primary files to cite**
- `paper_data/section_4_1_mib_silver/silver_agreement_summary.json`
- `paper_data/section_4_1_mib_silver/silver_agreement_by_subject_count.csv`
- `paper_data/section_4_1_mib_silver/silver_agreement_by_class_tag.csv`

## Section 4.2

### MIB-Gold: Human Annotation Findings and Benchmark Difficulty

**Core findings**
- The benchmark contains `4,020` raw pair groups in total: `1,500` from `v10` and `2,520` from `v13`.
- After requiring preference consistency, the retained-pair rate is `94.1%` on `v10` and `90.4%` on `v13`.
- Human preference consistency rises from `87.0%` at level `2` to `94.9%` at level `6`, then remains high at level `8`.
- Among the three diagnostic dimensions, `interaction` is the most ambiguous (`2027` ambiguous votes), followed by `appearance` (`1595`) and `existence` (`601`).

**Interpretation**
- The higher disagreement rate on `v13` supports the paper's intended story: unseen-generator evaluation is harder and more informative than seen-generator evaluation.
- The ambiguity ordering is also sensible. Existence is comparatively concrete, while interaction is inherently more interpretive in crowded multi-subject scenes.
- This section should be written as benchmark findings, not as annotation bookkeeping. The key message is that human labels reveal meaningful structure in benchmark difficulty.

**Recommended figure**
- `paper_data/images/section_4_2_gold_benchmark.png`

**Primary files to cite**
- `paper_data/section_4_2_mib_gold/gold_human_annotation_summary.json`
- `paper_data/section_4_2_mib_gold/gold_summary_by_level.csv`
- `paper_data/section_4_2_mib_gold/gold_summary_by_class_tag.csv`

## Section 5.1

### Existing Metrics Fail on MIB-Gold

**Core findings**
- The strongest third-party baseline is `ArcFace Distance` at `0.873` pairwise accuracy.
- The next tier is a small group of identity or representation-heavy metrics such as `DINO` and `SigLIP-I`, which remain far below a human-level substitute for integrated binding judgment.
- General preference metrics are weak in this setting: `HPS v2.1` is near random, `PickScore` is below `0.5`, and `PSNR` performs worst at `0.399`.
- Baseline sensitivity to subject count is substantial: several metrics degrade sharply as the number of requested subjects increases, confirming that multi-subject binding breaks the assumptions behind standard image-quality surrogates.

**Interpretation**
- This section should not just say that baselines are bad. It should say that MIB-Gold exposes a benchmark gap: existing metrics each capture only fragments of the human decision rule.
- Identity-specialized metrics can be competitive on some subsets, but they do not generalize into a unified evaluator of existence, appearance, and interaction.

**Recommended figure**
- `paper_data/images/section_5_1_baseline_alignment.png`

**Primary files to cite**
- `paper_data/section_5_1_existing_metrics/baseline_human_alignment_overall.csv`
- `paper_data/section_5_1_existing_metrics/baseline_human_alignment_v10_v13.csv`
- `paper_data/section_5_1_existing_metrics/baseline_metric_means_by_subject_count.csv`

## Section 5.2

### MIE Aligns Better with Human Preference

**Core findings**
- The best currently exported MIE variant is `qwen35_2b_lora_layer` with overall pairwise accuracy `0.903`.
- This best variant reaches `0.983` on `v10` and `0.854` on `v13`, meaning it remains substantially above the strongest third-party baseline even on the unseen benchmark.
- The same model also achieves macro-F1 `0.801`, supporting the claim that MIE is not just a ranker but a diagnostically meaningful evaluator.
- Among the currently available four checkpoints, `2B lora_layer` is the only variant that is simultaneously strongest in overall alignment, unseen-generator alignment, and macro-F1.

**Interpretation**
- This is the section where the paper should cash in on the benchmark story. MIB does not merely reveal that existing metrics fail; it also enables training a better evaluator.
- The main result here is not that every MIE variant is strong. It is that the best MIE variant clearly dominates the available baselines on human alignment while remaining interpretable.

**Recommended figure**
- `paper_data/images/section_5_2_mie_alignment.png`

**Primary files to cite**
- `paper_data/section_5_2_mie_alignment/mie_overall_metrics.csv`
- `paper_data/section_5_2_mie_alignment/mie_category_metrics.csv`
- `paper_data/section_5_2_mie_alignment/mie_vs_human_summary.json`

## Section 5.3

### Breakdown Analysis

**Core findings**
- All current MIE variants perform better on seen generators than unseen generators, but the generalization gap is smallest for `2b lora_layer` (`-0.129` unseen-minus-seen) and largest for `2b layer_only` (`-0.221`).
- At `2B`, adding LoRA-layer tuning improves pairwise accuracy by `0.061` and macro-F1 by `0.116` relative to `2B layer_only`.
- At `0.8B`, LoRA-layer barely changes pairwise accuracy (`-0.001`) but still improves macro-F1 by `0.128`.
- The category-level view shows that `Existence` is the easiest dimension, while `Appearance` and especially `Interaction` remain the harder diagnostics, particularly on `v13`.

**Interpretation**
- The most important ablation message is not simply `bigger is better`. The more defensible claim is that extra capacity only pays off when paired with the right fine-tuning regime.
- `2B lora_layer` is the current sweet spot because it simultaneously improves unseen generalization and the fine-grained diagnostic signal.
- The category breakdown also supports a strong narrative: interaction-heavy reasoning remains the hardest part of binding, which is consistent with both the human benchmark and the observed generation failures.

**Recommended figure**
- `paper_data/images/section_5_3_mie_breakdown.png`

**Primary files to cite**
- `paper_data/section_5_3_breakdown/mie_seen_unseen_table.csv`
- `paper_data/section_5_3_breakdown/mie_lora_vs_layer_table.csv`
- `paper_data/section_5_3_breakdown/mie_scaling_table.csv`
- `paper_data/section_5_3_breakdown/mie_category_by_dataset.csv`

## Writing Guidance

- For now, write Section 5 around the currently available `0.8B` and `2B` evaluator family rather than promising a full 6-model sweep.
- The strongest storyline is:
  1. silver labels are reliable enough to train on,
  2. gold human labels expose a real benchmark gap,
  3. existing metrics fail to close that gap,
  4. MIE, especially `2B lora_layer`, closes a meaningful part of it while remaining interpretable.
