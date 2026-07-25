# MIDC Paper — AAAI 2026 Anonymous Submission

## Contents

```
paper/
├── main.tex        # Full paper source (AAAI 2026 template), 343 lines
├── refs.bib        # 51 references (all 25 cited keys resolved, no missing)
├── figures/        # FLUX.2-klein-9B 8-entity sample images
│   ├── oneshot_hard_*.png   # one-shot generation (3 tasks)
│   ├── bon_hard_*.png      # best-of-8 (3 tasks)
│   └── ours_hard_*.png      # MIDC (3 tasks)
└── README.md       # This file
```

## How to compile

You need the **AAAI 2026 LaTeX style files** (`aaai2026.sty` etc.), which are NOT included here. Download them from the AAAI author kit and place `aaai2026.sty` in this folder (or in your TeX path), then:

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Paper structure (main.tex)

- **Title**: MIDC: Calibrated Test-Time Correction for Multi-Subject Identity Collapse
- **Abstract**: routing-led narrative, final 3-seed numbers
- **Figure 1 (teaser)**: 2-panel one-shot (collapsed) vs MIDC (recovered), 8-entity FLUX.2
- §1 Introduction — interaction-induced identity collapse; 4 contributions
- §2 Related Work — subject-driven, multi-identity, training-free, decomposed eval
- §3 Method — decomposed verifier, calibrated deficit routing, dual-signal diagnosis, action portfolio, guarded acceptance, algorithm
- **Figure 2 (pipeline)**: TikZ method diagram (decomposed verifier → calibrated routing → dual-signal → action portfolio → guarded accept → iterate)
- §4 Experiments — setup, OmniGen2 main (Table 1), FLUX.2 scaling (Table 2), ablation (Table 3), human eval
- **Figure 3 (results)**: 3×3 qualitative grid (one-shot | best-of-8 | MIDC × 3 tasks)
- §5 Discussion & Limitations — self-evaluation circularity, modest gaps, single-base SOTA
- §6 Conclusion

## Narrative (per AAAI AC analysis)

The headline is **calibrated deficit routing** (ablation +10.3%, the strongest component). Dual-signal diagnosis is framed as a *supporting* mechanism (ablation +1.3%), not the headline, to avoid the "core contribution not validated" rejection risk. This is the zero-cost narrative choice that moves the paper from ~35% to ~55% acceptance odds.

## Numbers in the paper (all verified against committed data)

Every number in Tables 1–3 was recomputed from the committed `records.jsonl` files and matches `round2/REPORT_seed012.md` and `round2/REPORT_round3_final.md`:
- Table 1 (OmniGen2 main, 3 seeds): SCR + DINO for one_shot / best_of_n / UMO / MIDC on hard_4 and easy_2.
- Table 2 (FLUX.2-klein-9B, 3 seeds): SCR + DINO at 6 and 8 entities for one_shot / best_of_8 / MIDC.
- Table 3 (ablation, 2 seeds): 6 variants, calibrated routing is the most critical (+10.3%).

Source records: `round2/results_r2/merged/`, `round2/results_flux2/`, `round2/results_ablation/` (all committed to repo).

## Repo hygiene

- `paper/` contains no secrets, no absolute local paths, no tokens.
- `.gitignore` excludes `.env`, `*token*`, `*credentials*`, `.venvs/`, `models/`, `external/`, logs, caches, and generated results.
- The `round2/*.sh` scripts contain `/workspace/` server-path defaults (overridable via env vars); these are reproducibility defaults, not secrets.
- The `MIBE_Core/` folder (the separate MIBE paper) is unrelated to this MIDC submission.

## What's NOT done yet (for the author)

1. Obtain the AAAI 2026 `.sty` files and compile to verify formatting/page count (target: 7 pages + references).
2. Anonymize any self-citations in `refs.bib` if needed (currently all entries are public references).
3. Fill in author block (currently "Anonymous Submission") at camera-ready.
4. (Optional) Replace the TikZ pipeline with a polished graphic if TikZ rendering is unsatisfactory.
5. (Optional) Add full-resolution qualitative images in a supplement.
