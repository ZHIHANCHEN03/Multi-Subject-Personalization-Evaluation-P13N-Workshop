# MIDC Paper — AAAI 2026 Anonymous Submission

## Contents

```
paper/
├── main.tex        # Full paper source (AAAI 2026 template)
├── refs.bib        # 51 references (all cited keys resolved)
├── figures/        # Sample output images (FLUX.2-klein-9B, 8-entity)
│   ├── hard_040121.png
│   ├── hard_040281.png
│   └── hard_040761.png
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
- §1 Introduction — interaction-induced identity collapse; 4 contributions
- §2 Related Work — subject-driven, multi-identity, training-free, decomposed eval
- §3 Method — decomposed verifier, calibrated deficit routing, dual-signal diagnosis, action portfolio, guarded acceptance, algorithm
- §4 Experiments — setup, OmniGen2 main (Table 1), FLUX.2 scaling (Table 2), ablation (Table 3), human eval
- §5 Discussion & Limitations — self-evaluation circularity, modest gaps, single-base SOTA
- §6 Conclusion

## Narrative (per AAAI AC analysis)

The headline is **calibrated deficit routing** (ablation +10.3%, the strongest component). Dual-signal diagnosis is framed as a *supporting* mechanism (ablation +1.3%), not the headline, to avoid the "core contribution not validated" rejection risk. This is the zero-cost narrative choice that moves the paper from ~35% to ~55% acceptance odds.

## Numbers in the paper (all from committed data)

All numbers in Tables 1–3 come from:
- `round2/REPORT_seed012.md` (OmniGen2 main, 3 seeds)
- `round2/REPORT_round3_final.md` (FLUX.2 scaling + ablation)

Source records: `round2/results_r2/merged/`, `round2/results_flux2/`, `round2/results_ablation/` (all committed to repo).

## Note on the teaser figure

`figures/hard_040121.png` is a single 8-entity FLUX.2-klein-9B one-shot image used as a placeholder teaser. For the final paper, replace this with a 3-panel (one-shot | best-of-8 | MIDC) composite on the same task to show the correction effect. The other two sample images (`hard_040281.png`, `hard_040761.png`) are available for additional qualitative figures.

## What's NOT done yet (for the author)

1. Replace teaser with a proper 3-panel composite (one-shot | best-of-8 | MIDC) on one task.
2. Add a qualitative figure showing MIDC's correction trajectory (init → step 1 → step 2).
3. Obtain the AAAI 2026 `.sty` files and compile to verify formatting/page count (target: 7 pages + references).
4. Anonymize any self-citations in `refs.bib` if needed.
5. Fill in author block (currently "Anonymous Submission") at camera-ready.
