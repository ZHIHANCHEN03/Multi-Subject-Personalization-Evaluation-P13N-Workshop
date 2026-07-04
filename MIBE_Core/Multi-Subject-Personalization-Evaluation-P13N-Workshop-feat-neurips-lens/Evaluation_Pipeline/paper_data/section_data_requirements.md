# Section Data Requirements

This note maps each planned results section to:

- what data it needs
- where that data lives
- why the data is needed
- whether the currently available files are already sufficient

It is written to support the current paper restructuring:

- `4 Benchmark Validation`
- `4.1 MIB-Silver: Label Reliability and Scalability`
- `4.2 MIB-Gold: Human Annotation Findings and Benchmark Difficulty`
- `5 Evaluator Results on MIB-Gold`
- `5.1 Existing Metrics Fail on MIB-Gold`
- `5.2 MIE Aligns Better with Human Preference`
- `5.3 Breakdown Analysis`

## High-Level Recommendation

Yes, it is reasonable to **temporarily ignore the 4B evaluator results** and write the paper around the currently available `0.8B` and `2B` evaluator checkpoints.

Why this is acceptable:

- the benchmark story does not depend on 4B
- `4.1`, `4.2`, and `5.1` are fully supported by existing data
- `5.2` and `5.3` already have enough material to support a strong story using the currently exported 4 evaluator variants:
  - `qwen35_08b_layer_only`
  - `qwen35_08b_lora_layer`
  - `qwen35_2b_layer_only`
  - `qwen35_2b_lora_layer`
- this is already sufficient to support:
  - evaluator-vs-baseline comparison
  - scale trend from `0.8B -> 2B`
  - `layer_only` vs `lora_layer`
  - category-level and seen/unseen analysis

What this means for writing:

- do **not** claim a complete 6-model evaluator sweep in the paper unless the 4B jsonl files are also available
- it is safer to explicitly frame the evaluator ablation as the currently available `0.8B/2B` family

## Section 4.1.1

### 4.1 MIB-Silver: Label Reliability and Scalability

### Required data

- raw LLM annotation outputs from the two silver annotators
- fields needed:
  - `task_id`
  - `winner`
  - `a_existence`, `a_appearance`, `a_interaction`
  - `b_existence`, `b_appearance`, `b_interaction`
  - `subject_count`
  - `metadata.level`
  - `metadata.class_tag`
  - `metadata.ratio_type`

### Current files

- `/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Model_Training/data_v2/60k_LLM_Result/2_5_merged_sorted.jsonl`
- `/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Model_Training/data_v2/60k_LLM_Result/3_1_merged_sorted.jsonl`

### Why these files are needed

This section is meant to validate that `MIB-Silver` is a credible supervision source.
The main claims here are:

- the SOP makes two frontier LLM judges agree at a high rate
- preference agreement is stronger than naive prompt-based judging would suggest
- agreement can be analyzed by:
  - dimension
  - subject count
  - relation type
- the silver set is therefore a scalable but reasonably reliable training resource

### Why the current files are sufficient

These two jsonl files are exactly the raw per-model silver annotations.
They are sufficient for:

- preference agreement
- per-dimension agreement
- by-subject-count agreement
- by-class-tag / relation-type agreement
- qualitative disagreement or agreement examples

### What this section can already support

- a full `4.1` subsection is already supportable
- no 4B evaluator output is needed

### Caveat

If you want to report the exact size of the final consensus-filtered silver subset used for training, you may also want the post-intersection training split statistics.
But for the core two-LLM agreement analysis, these two files are enough.

## Section 4.1.2

### 4.2 MIB-Gold: Human Annotation Findings and Benchmark Difficulty

### Required data

- human annotation records for the gold benchmark
- fields needed:
  - `combo_id` or `base_id`
  - `level`
  - `class_tag`
  - `ratio_type`
  - `model_a`, `model_b`
  - `a_existence`, `a_appearance`, `a_interaction`
  - `b_existence`, `b_appearance`, `b_interaction`
  - `preference`
  - `prompt_ilogical`

### Current files

- `/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Evaluation_Pipeline/data/human_annotations/round10_annotations_latest.csv`
- `/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Evaluation_Pipeline/data/human_annotations/export_round1260_20260503.csv`

### Why these files are needed

This section is not just about "we collected human labels".
It should show what the human benchmark reveals:

- which failure modes are common
- how difficulty scales with subject count
- where disagreement happens
- how existence and appearance failures co-occur
- why this benchmark is challenging and informative

### Why the current files are sufficient

These two CSV files are the actual human annotation sources for:

- `v10` / seen-generator human benchmark
- `v13` / unseen-generator human benchmark

They contain the exact fields needed for:

- human preference aggregation
- disagreement filtering
- category aggregation
- preference-consistency filtering
- descriptive analysis of benchmark difficulty

### What this section can already support

- human annotation protocol summary
- valid-vs-dropped sample counts
- level-wise and dataset-wise benchmark difficulty behavior
- qualitative and quantitative benchmark findings
- a clean `4.2` subsection

### Caveat

This section should be written as `human annotation findings`, not just `annotation statistics`.
The strongest use of these files is to explain what the benchmark reveals about multi-subject failure.

## Section 4.2.1

### 5.1 Existing Metrics Fail on MIB-Gold

### Required data

- baseline metric outputs on the gold benchmark
- aggregated human-vs-metric alignment summaries
- if possible:
  - metric means overall
  - metric means by subject count
  - metric means by relation type or scenario type

### Current files

Primary source directories:

- `/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/multisubject_generation_eval/data`
- `/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/multisubject_generation_eval/data/v13_2_1.26k_evl/eval_results`

Especially useful analysis files:

- `/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/multisubject_generation_eval/data/v13_2_1.26k_evl/eval_results/analysis/human_vs_metric_pairwise_accuracy_v10_v13.csv`
- `/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/multisubject_generation_eval/data/v13_2_1.26k_evl/eval_results/analysis/metric_means_by_subject_count.csv`
- `/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/multisubject_generation_eval/data/v13_2_1.26k_evl/eval_results/analysis/metric_means_overall.csv`
- `/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/multisubject_generation_eval/data/v13_2_1.26k_evl/eval_results/analysis/metric_means_by_occlusion_interaction.csv`

Raw metric result files also exist, for example:

- `*_dino_clip_metrics.json`
- `*_pref_metrics.json`
- `*_fast_metrics.json`
- `*_lpips_metrics.json`
- `*_arcface_humans.json`

### Why these files are needed

This section should establish the benchmark gap:

- existing metrics are inconsistent with human preference
- some metrics only capture one facet, such as identity or coarse semantics
- no third-party baseline faithfully reproduces human ranking in multi-subject binding scenarios

### Why the current files are sufficient

The current files already include:

- per-metric human alignment summaries
- per-model metric means
- by-subject-count metric behavior
- baseline result visualizations and tables

This is enough to write a strong `5.1` subsection without needing any new evaluator output.

### What this section can already support

- the main baseline table
- pairwise human-alignment comparison
- by-subject-count failure trend
- optional by-scenario breakdown

## Section 4.2.2

### 5.2 MIE Aligns Better with Human Preference

### Required data

- exported evaluator jsonl outputs for each evaluator checkpoint
- human annotation data for comparison
- preferably an already aggregated evaluator-vs-human summary

### Current files

Evaluator outputs:

- `/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Evaluation_Pipeline/outputs/jsonl/auto_manifest_lens_scores_all6__qwen35_08b_layer_only.jsonl`
- `/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Evaluation_Pipeline/outputs/jsonl/auto_manifest_lens_scores_all6__qwen35_08b_lora_layer.jsonl`
- `/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Evaluation_Pipeline/outputs/jsonl/auto_manifest_lens_scores_all6__qwen35_2b_layer_only.jsonl`
- `/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Evaluation_Pipeline/outputs/jsonl/auto_manifest_lens_scores_all6__qwen35_2b_lora_layer.jsonl`

Supporting human-alignment summary:

- `/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Evaluation_Pipeline/outputs/summaries/metrics_vs_human_summary.json`

Relevant analysis script:

- `/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Evaluation_Pipeline/scripts/analyze_metrics_vs_human.py`

### Why these files are needed

This section should show:

- MIE is better aligned with human preference than third-party metrics
- the evaluator can generalize across the gold benchmark
- the benchmark is not only diagnostic, but also useful for training better evaluators

### Why the current files are sufficient

These files are sufficient for a strong `5.2` subsection using the currently available 4 evaluator variants.
They already support:

- overall pairwise accuracy vs human preference
- seen vs unseen comparison
- category-level alignment
- evaluator family comparison across `0.8B` and `2B`
- `layer_only` vs `lora_layer`

### Important limitation

The current `outputs/jsonl` directory contains only **4 evaluator outputs**, not the full 6-model set.

Currently available:

- `qwen35_08b_layer_only`
- `qwen35_08b_lora_layer`
- `qwen35_2b_layer_only`
- `qwen35_2b_lora_layer`

Not yet available in this directory:

- `qwen35_4b_layer_only`
- `qwen35_4b_lora_layer`

### Writing recommendation

For now, write this section as:

- evaluator results for the currently available `0.8B` and `2B` MIE variants

This is already enough to support a credible section.

## Section 4.2.3

### 5.3 Breakdown Analysis

### Required data

- same evaluator jsonl outputs as `5.2`
- same human annotation files as `4.2`
- aggregated evaluator-vs-human analysis
- optional figures and summaries

### Current files

Primary evaluator outputs:

- all files in `/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Evaluation_Pipeline/outputs/jsonl`

Human annotations:

- `/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Evaluation_Pipeline/data/human_annotations/round10_annotations_latest.csv`
- `/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Evaluation_Pipeline/data/human_annotations/export_round1260_20260503.csv`

Helpful summary/figure files:

- `/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Evaluation_Pipeline/outputs/summaries/metrics_vs_human_summary.json`
- `/Users/bytedance/Downloads/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Evaluation_Pipeline/outputs/figures/metrics_vs_human_alignment.png`

### Why these files are needed

This section should explain *why* MIE works better, not just *that* it works better.
Typical breakdowns include:

- seen vs unseen generators
- category-level performance
- parameter scaling from `0.8B -> 2B`
- `layer_only` vs `lora_layer`

### Why the current files are sufficient

Even without 4B, the currently available 4 evaluator files are already enough for:

- seen vs unseen analysis
- category-level analysis
- `0.8B vs 2B` scaling trend
- `layer_only vs lora_layer`

### Important limitation

If you want to claim a full cross-scale trend across `0.8B`, `2B`, and `4B`, then the current files are **not yet sufficient**.
If you only need:

- small-to-medium scale trend
- LoRA vs layer comparison
- category-level and seen/unseen breakdown

then the current files are sufficient.

## Bottom Line

### Sections already fully supportable now

- `4.1 MIB-Silver`
- `4.2 MIB-Gold`
- `5.1 Existing Metrics Fail on MIB-Gold`

### Sections supportable now with current evaluator scope

- `5.2 MIE Aligns Better with Human Preference`
- `5.3 Breakdown Analysis`

as long as the paper clearly states that the currently available evaluator comparison is based on the exported `0.8B` and `2B` variants.

### Sections not yet supportable if you insist on full 6-model evaluator claims

- any subsection that explicitly claims full evaluator coverage including the 4B checkpoints

## Practical Writing Advice

If you want to move fast now, the safest paper framing is:

- `4.1`: validate silver supervision
- `4.2`: validate gold benchmark and explain what human labels reveal
- `5.1`: show existing metrics fail on the benchmark
- `5.2`: show currently available MIE variants outperform baselines in human alignment
- `5.3`: explain the gain through seen/unseen, category, and small-to-medium-scale breakdowns

This gives you a complete and internally consistent story **without waiting for 4B**.
