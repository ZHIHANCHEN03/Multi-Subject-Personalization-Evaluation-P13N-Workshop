# Code and Data Package — MIDC (Anonymous Submission #27384)

This package supports the paper "Calibrated Test-Time Correction for
Multi-Subject Identity Collapse" (MIDC). It contains the inference runners for
all compared methods, the committed per-task records behind every reported
number, and the analysis scripts that turn records into the paper's tables.
Everything is anonymized.

## Layout

```
round1/            Inference runners and shared infrastructure (GPU, generation side)
  p1_ours_v2.py      MIDC runner (calibrated routing, dual-signal diagnosis,
                     propose-verify loop, guarded acceptance, selective gate)
  p2_oneshot.py      one_shot baseline
  p3_bestofn.py      best_of_n baseline (N=8, verifier-total selection)
  p4_umo.py          UMO baseline (official inference config, context-refiner
                     LoRA rank 512 on OmniGen2)
  common.py          Shared: manifest loading, MIE critic client, SCR/DINO scoring
  actions.py         Calibrated deficit routing + the three actions
                     (reference reorder, prompt rewrite, refset manipulation)
  generators.py      Generator wrappers (OmniGen2, FLUX.2-klein-9B)
  external_generators.py  UMO/FreeGraFTor adapters
  mie_server.py      Persistent MIE verifier server (Qwen-based, frozen)
  calibrate_mie.py   Verifier calibration (deficit normalization stats)
  requirements.txt   Python dependencies

round2/            Analysis and records (CPU, no GPU needed for verification)
  verify_paper_numbers.py  Recomputes every number quoted in the paper from the
                           committed records and asserts equality (start here)
  analyze.py               Aggregate records -> main-result tables, bootstrap
                           95% CIs, paired two-sided tests
  b1_reanalysis.py         Selective-correction gating analysis
  blur_counterfactual.py   Blur-matched counterfactual analysis
  score_*.py               Per-split scoring entry points
  plot_*.py                Figure generation scripts
  results_*/               Committed per-task records (JSONL):
                             results_r2        OmniGen2, 375 tasks x 3 seeds x 4 methods
                             results_flux2     FLUX.2-klein-9B, 250 tasks x 3 seeds x 3 methods
                             results_ablation  MIDC ablations (no-gate, raw routing, etc.)
                             results_blur_cf   blur-matched counterfactual records
                             results_clip      CLIP-similarity scoring records
                             human eval records (n=142 / n=130 comparisons)

MIE_Inference/     The frozen decomposed verifier V (existence/appearance/interaction)
  mie_loader.py      load_runtime() + score(); verifies weights actually applied
  verify_umo_lora.py UMO adapter activity check (main paper Sec. 4.1 footnote;
                     supplementary Sec. A)
  score_batch.py     Batch scoring over a manifest
```

## Reproduce the paper's numbers (no GPU required)

Every statistic quoted in the paper (Tables 1-4, all in-text numbers) is
computed from the committed JSONL records under `round2/results_*/`:

```
cd round2
python verify_paper_numbers.py   # asserts every quoted number matches records
python analyze.py                # prints main tables with CIs and paired tests
```

Scoring convention (identical to the paper): per-subject DINOv2 CLS similarity
over Grounding-DINO crops against the task's reference images; SCR counts a
subject as missed when its best-crop similarity is below delta=0.50.

## Re-run generation (GPU required)

Generation used seeds {0, 1, 2} throughout. Required assets, loaded from paths
set via environment variables (`MIDC_MODELS`, `MIDC_ROUND1`, `UMO_ROOT`,
`MIE_CKPT`; see `round1/setup_round1.sh`):

- OmniGen2 (base generator) and FLUX.2-klein-9B (step-distilled transfer base)
- UMO context-refiner adapter (UMO_OmniGen2.safetensors, rank 512)
- MIE verifier checkpoint (the decomposed evaluator of the companion
  benchmark, Anonymous 2026, cited in the paper)
- MIBE task manifests and reference images (companion benchmark)
- DINOv2 and Grounding-DINO (headline-metric scoring)

Example: `cd round1 && bash run_round1.sh` runs one_shot / best_of_n / UMO /
MIDC over a manifest shard. `MIE_Inference/verify_umo_lora.py --data
<manifest.jsonl>` reproduces the adapter-activity check.

No part of this package trains any model; MIDC is training-free.
