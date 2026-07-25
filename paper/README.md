# MIDC Paper — AAAI 2026 Anonymous Submission

## Status: READY TO SUBMIT

- **6 pages** (AAAI 2026 limit: 7 pages main + unlimited references)
- Compiles clean: `pdflatex` + `bibtex` + `pdflatex` ×2 → **0 errors, 0 undefined refs/citations, 0 bibtex warnings**
- All numbers verified against committed `records.jsonl` data (see "Number verification" below)
- Self-contained: no secrets, no absolute local paths, no tokens

## Contents

```
paper/
├── main.tex        # Full paper source (AAAI 2026 template)
├── main.pdf        # Compiled PDF (6 pages)
├── refs.bib        # References (all cited keys resolved)
├── aaai2026.sty    # AAAI 2026 official style (from AAAI author kit)
├── aaai2026.bst    # AAAI 2026 bibliography style
├── figures/        # 9 qualitative images (oneshot/bon/ours × 3 tasks), all referenced
└── README.md       # This file
```

## How to compile

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```
Requires a TeX distribution with `pdflatex`/`bibtex` (e.g., TeX Live). The AAAI 2026 `.sty`/`.bst` are included in this folder.

## Paper structure

- **Title**: MIDC: Calibrated Test-Time Correction for Multi-Subject Identity Collapse
- **Abstract**: routing-led narrative; 3-seed headline numbers; human-eval corroboration; compute efficiency
- **Figure 1 (teaser)**: 2-panel one-shot (collapsed) vs MIDC (recovered), 8-entity FLUX.2
- §1 Introduction — interaction-induced identity collapse; 4 contributions
- §2 Related Work — subject-driven, multi-identity, training-free, decomposed eval
- §3 Method — decomposed verifier, calibrated deficit routing, dual-signal diagnosis, action portfolio, guarded acceptance, algorithm
- **Figure 2 (pipeline)**: TikZ method diagram
- §4 Experiments
  - §4.1 Setup
  - §4.2 Main Results: OmniGen2 — **Table 1** (SCR + DINO, 3 seeds, hard_4 + easy_2)
  - §4.3 FLUX.2 Scaling — **Table 2** (6/8-entity, 3 seeds) + **Table 4** (compute/generations per task) + **Figure 3** (3×3 qualitative grid)
  - §4.4 Ablation — **Table 3** (6 variants, calibrated routing is the star at +10.3%)
  - §4.5 Human Evaluation — **Table 5** (3 labelers, 4 dimensions; Q1 existence p=4e-7)
- §5 Discussion & Limitations — circularity, when MIDC fails, modest gaps, single-base SOTA, dual-signal caveat
- §6 Conclusion

## Narrative (per AAAI AC analysis)

The headline is **calibrated deficit routing** (ablation +10.3%, the strongest component). Dual-signal diagnosis is framed as a *supporting* mechanism (ablation +1.3%), not the headline, to avoid the "core contribution not validated" rejection risk. Human eval (Q1 existence, 84.6% preference, p=4e-7) independently corroborates the automatic SCR result and is highlighted in abstract + discussion.

## Number verification (all recomputed from committed data)

Every number in Tables 1–5 was recomputed from the committed `records.jsonl` files and matches `round2/REPORT_seed012.md` and `round2/REPORT_round3_final.md`:

- **Table 1** (OmniGen2 main, 3 seeds, n=500): SCR + DINO for one_shot / best_of_n / UMO / MIDC on hard_4 and easy_2. Verified.
- **Table 2** (FLUX.2-klein-9B, 3 seeds, n=100/cell): SCR + DINO at 6 and 8 entities for one_shot / best_of_8 / MIDC. 6-entity DINO = 0.393/0.408/0.428 (recomputed; was a hallucination in an earlier draft, now fixed).
- **Table 3** (ablation, 2 seeds, n=100): 6 variants; calibrated routing +10.3% is the largest delta.
- **Table 4** (compute): generator calls per task — one_shot 1.0, UMO 1.0, best_of_8 8.0, MIDC 4.6 (OmniGen2) / 4.3 (FLUX.2 8-entity). Recomputed from `gen_calls` field.
- **Table 5** (human eval): from `round2/human_eval/HUMAN_EVAL/aggregate_result_3labeler.json`. MIDC vs UMO Q1: 84.6% win [0.75,0.94], n=52, Fleiss κ=0.32, binomial p=4e-7. Q4: 58.9%, n=73, p=0.16 (directional). Q2/Q3: negative κ (no claim).

Source records: `round2/results_r2/merged/`, `round2/results_flux2/`, `round2/results_ablation/`, `round2/human_eval/HUMAN_EVAL/` (all committed to repo).

## Repo hygiene

- `paper/` contains no secrets, no absolute local paths, no tokens.
- `.gitignore` excludes `.env`, `*token*`, `*credentials*`, `.venvs/`, `models/`, `external/`, logs, caches, and generated results.
- The `round2/*.sh` scripts contain `/workspace/` server-path defaults (overridable via env vars); these are reproducibility defaults, not secrets.
- `MIBE_Core/` (the separate MIBE paper) is unrelated to this MIDC submission.

## Remaining for camera-ready (not submission)

1. Fill in author block (currently "Anonymous Submission") and affiliations at camera-ready.
2. (Optional) Replace the TikZ pipeline with a polished graphic.
3. (Optional) Add full-resolution qualitative images in a supplement.
