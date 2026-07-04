# NeurIPS Datasets & Benchmarks Initial Submission Checklist

This note summarizes what should be uploaded for the **first submission** of this project to the **NeurIPS Datasets & Benchmarks Track**, based on the public 2025 D&B track call, hosting guidelines, and FAQ.

Primary references:

- [NeurIPS 2025 Datasets & Benchmarks Call for Papers](https://nips.cc/Conferences/2025/CallForDatasetsBenchmarks)
- [NeurIPS 2025 Data Hosting Guidelines](https://neurips.cc/Conferences/2025/DataHostingGuidelines)
- [NeurIPS 2025 Datasets & Benchmarks FAQ](https://nips.cc/Conferences/2025/DatasetsBenchmarks-FAQ)

## Short Answer

For the first submission, do **not** think in terms of "upload every raw score we ever produced".

Instead, think in terms of:

- the **dataset/benchmark artifacts that are claimed in the paper**
- the **metadata and documentation needed for review**
- the **code required to reproduce the benchmark evaluation**
- the **summary results needed to verify the paper's claims**

For this paper, the safest initial submission package is:

- the public/reviewable portion of `MIB-Gold`
- the public/reviewable portion of `MIB-Silver`
- benchmark metadata and label schema
- benchmark/evaluation code
- baseline result summaries
- current MIE result summaries

## What NeurIPS D&B Cares About at First Submission

The main D&B expectations are:

- dataset and benchmark **must be accessible at submission time**
- code must be **accessible and executable at submission time**
- dataset/code are **not supplementary-only** in this track
- dataset access must not require a private email request to the authors
- dataset should be hosted via a proper data hosting platform, ideally:
  - Dataverse
  - Kaggle
  - Hugging Face
  - OpenML
- a **Croissant metadata file** is expected for datasets
- if the dataset is not public yet, a **reviewer-accessible private preview URL** is acceptable
- if accepted, the dataset is expected to become public by the camera-ready deadline

## What This Means for This Project

Your paper is not only a benchmark paper, but a **benchmark + dataset resource + evaluator proof** paper.

That means the first submission package should cover:

- `MIB-Gold` as the human benchmark
- `MIB-Silver` as the large-scale supervision resource
- the benchmark/evaluator code needed to understand and verify the paper
- enough result summaries to validate the main claims in Sections 4 and 5

## Recommended Initial Upload Package

## 1. Benchmark Dataset Package

This is the most important part.

### Must include

- benchmark metadata file(s)
- prompt / pair / sample identifiers
- split definitions
- generator identity fields
- difficulty tags
- label schema documentation
- benchmark examples

### For this project, the submission package should include

#### MIB-Gold

- all public/reviewable benchmark pair metadata
- all human labels used for evaluation
- the split identifiers needed for:
  - `v10`
  - `v13`
  - seen vs unseen generator analysis
- the benchmark control tags:
  - `level`
  - `class_tag`
  - `ratio_type`
- clear definitions for:
  - `preference`
  - `existence`
  - `appearance`
  - `interaction`

#### MIB-Silver

- all public/reviewable silver supervision metadata
- silver preference and diagnostic labels
- the SOP/judge metadata needed to understand how labels were produced
- enough documentation to explain:
  - two-LLM judging
  - agreement filtering or consensus logic
  - how silver labels become training supervision

### Strong recommendation

If the paper claims the `60K silver dataset` as a contribution, then the initial submission should make the **claimed public portion** of that resource available for review.

Do **not** write the paper as if the 60K resource is a major contribution while only exposing a tiny sample unless you explicitly describe the public sample vs private holdout split.

## 2. Metadata and Documentation Package

This is what prevents reviewer confusion.

### Must include

- README / dataset card
- file schema documentation
- split documentation
- label definition document
- filtering and aggregation rules
- licensing / usage notes
- ethical or release constraints if any

### For this project, document at least

- what a benchmark item is
- what a pair is
- what a `base_id` / `combo_id` means
- how `v10` and `v13` differ
- what is meant by seen vs unseen generators
- how human disagreement is handled
- how `prompt_ilogical` is used
- how image-level labels are derived from A/B annotations
- how silver labels are generated and validated

### Croissant

For NeurIPS D&B, a Croissant metadata file is expected for the dataset submission.

So the initial package should ideally include:

- dataset hosting URL
- generated Croissant file
- if multiple datasets are submitted:
  - one Croissant file per dataset, or
  - a zip of multiple Croissant files

In your case, the natural split is:

- one metadata artifact for `MIB-Gold`
- one metadata artifact for `MIB-Silver`

## 3. Code Package

This is also required at submission time.

### Must include

- benchmark/evaluation code
- analysis code for main tables
- instructions to run the core evaluation

### For this project, the minimum useful code package is

- benchmark analysis code for:
  - `4.1 MIB-Silver`
  - `4.2 MIB-Gold`
  - `5.1 Existing Metrics`
  - `5.2 MIE`
  - `5.3 Breakdown`
- evaluator analysis code
- scripts that generate the summary tables used in the paper
- enough README instructions so a reviewer can understand how the results are produced

### Good current candidates from this repo

- `Evaluation_Pipeline/paper_data/build_paper_data.py`
- `Evaluation_Pipeline/paper_data/generate_section4_5_analysis.py`
- `Evaluation_Pipeline/scripts/analyze_metrics_vs_human.py`
- `Evaluation_Pipeline/scripts/analyze_metrics_jsonl.py`
- if you want to expose evaluator inference:
  - `Evaluation_Pipeline/scripts/export_lens_scores.py`
  - `Evaluation_Pipeline/scripts/run_export_lens_scores.sh`

If you think the full evaluator export path is too heavy for initial review, it is acceptable to prioritize:

- analysis/reproduction code for the paper results
- plus clear description of the evaluator inference pipeline

## 4. Results Package

NeurIPS D&B reviewers do not need every raw intermediate score inside the PDF, but they do need enough structured results to verify your story.

### Must include somewhere in the submission package

- benchmark summary statistics
- baseline result summaries
- MIE result summaries
- enough breakdowns to support the main conclusions

### The safest project-specific package is

- `paper_data/section_4_1_1_mib_silver/*`
- `paper_data/section_4_1_2_mib_gold/*`
- `paper_data/section_4_2_1_existing_metrics/*`
- `paper_data/section_4_2_2_mie_alignment/*`
- `paper_data/section_4_2_3_breakdown/*`
- `paper_data/paper_section4&5_analysis.md`

These are especially useful because they are already paper-facing and reviewer-readable.

## What You Do NOT Need to Upload in Full for First Submission

You do **not** need to put every internal artifact into the first submission package.

These can remain outside the first review package unless needed:

- every intermediate cache file
- every temporary plotting artifact
- every experimental log
- every per-image evaluator raw score not referenced by the paper
- every checkpoint
- every training run directory

### Important nuance

If a raw score file is required to support a central paper claim, then it should be represented at least through:

- a reproducible summary table
- a script that regenerates the summary
- or a direct reviewable result file

So the goal is not "upload everything", but "upload everything needed for reviewable reproducibility".

## Recommended Size / Scope for Initial Submission

The safest review-time package is:

### Public/reviewable benchmark artifacts

- full `MIB-Gold`
- full public/reviewable `MIB-Silver`

### Review-time summary artifacts

- the section-level files already built under `paper_data`

### Review-time code artifacts

- benchmark analysis code
- summary generation code
- minimal evaluator result reproduction code

### Review-time documentation

- dataset card / README
- schema doc
- split doc
- label doc
- Croissant file(s)

## Recommended Project-Specific First Submission Bundle

If you want a very concrete answer, this is the recommended first upload bundle for your project:

### A. Dataset hosting side

- hosted `MIB-Gold` dataset
- hosted `MIB-Silver` dataset
- reviewer-accessible URL(s)
- Croissant file for gold
- Croissant file for silver

### B. Code hosting side

- benchmark analysis code
- summary generation code
- instructions to regenerate Section 4 and 5 tables

### C. Paper companion artifacts

- `paper_data/section_4_1_1_mib_silver`
- `paper_data/section_4_1_2_mib_gold`
- `paper_data/section_4_2_1_existing_metrics`
- `paper_data/section_4_2_2_mie_alignment`
- `paper_data/section_4_2_3_breakdown`
- `paper_data/paper_section4&5_analysis.md`
- `paper_data/images`

## Current Best Practical Strategy

For the first submission, the most defensible strategy is:

- upload the benchmark/dataset resources claimed in the paper
- upload the metadata/schema needed to understand them
- upload reviewer-accessible code
- upload section-level result summaries
- do **not** try to dump every raw artifact

## One-Sentence Bottom Line

For the initial NeurIPS Datasets & Benchmarks submission, you should upload:

- the reviewable public portion of `MIB-Gold`
- the reviewable public portion of `MIB-Silver`
- dataset metadata and Croissant files
- benchmark/evaluation code
- the section-level result summaries needed to verify the paper

You do **not** need to upload every raw score and every training artifact as long as the paper's claims are reviewable and reproducible.
