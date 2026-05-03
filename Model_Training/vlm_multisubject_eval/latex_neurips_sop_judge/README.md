# SOP NeurIPS Draft

This is a minimal LaTeX project under `vlm_multisubject_eval` for writing a NeurIPS-style note on SOP-guided LLM-as-judge evaluation.

## Template choice

- `main.tex` is written to prefer the latest publicly available NeurIPS style naming convention, `neurips_2025.sty`.
- If `neurips_2025.sty` is not present in this folder yet, the document falls back to a plain `article` layout so the text remains editable and compilable.
- To switch to the official NeurIPS appearance, place the official `neurips_2025.sty` file in this directory and compile again.

## Files

- `main.tex`: entry point
- `sections/sop_alignment_subsection.tex`: the requested subsection draft
- `refs.bib`: bibliography placeholder

## Build

```bash
cd /Users/bytedance/Documents/multi_subject_generation/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Model_Training/vlm_multisubject_eval/latex_neurips_sop_judge
pdflatex main.tex
```
