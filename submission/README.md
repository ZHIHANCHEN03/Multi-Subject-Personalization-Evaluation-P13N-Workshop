# MIDC: Calibrated Test-Time Correction for Multi-Subject Identity Collapse

This repository contains the code, data, and paper source for **MIDC**, a
training-free, inference-time correction paradigm that diagnoses and repairs
*interaction-induced multi-subject identity collapse* using a decomposed
verifier as a structured controller.

> **Status**: AAAI-27 submission (full paper due 2026-07-28, supplementary +
> code due 2026-07-31). The paper lives in [`paper/`](paper/) and is built with
> the official AAAI-27 `aaai2027.sty` author kit.

## What is MIDC?

Multi-subject personalized image generation suffers from *identity collapse*:
as the number of subjects grows and their interactions become entangled,
generators drop, merge, or swap identities. MIDC is **training-free** and
**model-agnostic**: it treats the generator as a black box and uses a
*decomposed* verifier (scoring existence / appearance / interaction) as a
diagnostic signal to route correction to the most deficient facet, with a
dual-signal diagnosis (facet deficit + subject-level identity similarity)
localizing the collapsed subject. Correction is a propose–verify loop with
guarded acceptance. See [`paper/main.tex`](paper/main.tex) for the full method.

## Repository layout

| Path | Contents |
|---|---|
| [`paper/`](paper/) | AAAI-27 paper source (`main.tex`, `refs.bib`, `aaai2027.sty`, figures) — see [`paper/README.md`](paper/README.md) |
| [`round1/`](round1/) | Core MIDC pipeline: `p1_ours_v2.py` (main method), `p2_oneshot.py`, `p3_bestofn.py`, `p4_umo.py`, `external_generators.py` (OmniGen2 / FLUX.2 / UMO LoRA adapters), `common.py` |
| [`round1_1/`](round1_1/) | Ablation sweep configs (`sweep.py`) |
| [`round2/`](round2/) | Round-2/3 experiment scripts, manifests, and results — see [`round2/README.md`](round2/README.md) |
| [`round2/results_r2/`](round2/results_r2/) | **Raw per-task records** (`records.jsonl`) for the 500-task OmniGen2 main experiment (all methods, 3 seeds) — the basis for Table 1 |
| [`round2/results_flux2/`](round2/results_flux2/) | Raw records for the FLUX.2-klein-9B 6/8-entity scaling study — the basis for Table 2 and the gating analysis (§4.5, Figure 5) |
| [`round2/verify_paper_numbers.py`](round2/verify_paper_numbers.py) | Recomputes every value in Tables 1, 2, 3, 5, the §4.3 generator calls, and the §4.5 gating split from the committed records and diffs them against `paper/main.tex`; exits non-zero on any disagreement. Table 4 (paired bootstrap) is reproduced by [`round2/analyze.py`](round2/analyze.py) |
| [`ABSTRACT.md`](ABSTRACT.md) | Final title + abstract (synced with `paper/main.tex`) — **must be mirrored to OpenReview before 7/28** |
| `../meta/PLAN.md` | Project plan and claim boundaries (working doc, outside `submission/`) |

> **Companion MIBE data (outside `submission/`):** the MIE verifier training data
> (`prompt/train_60k_v13_2.jsonl`, 42 MB) and the full MIBE evaluator/benchmark
> (`MIBE_Core/`, 258 MB) belong to the companion MIBE paper (cited as
> `anon2025mibe`, under review separately). They live in the repo's sibling
> [`companion/`](../companion/) folder, deliberately kept out of `submission/`
> so the submission export stays lean. Experiment scripts reference the training
> data via the `DATA_SRC` env var (default `prompt/train_60k_v13_2.jsonl`,
> overridable); point `DATA_SRC` at `../companion/prompt/train_60k_v13_2.jsonl`
> to reproduce hard-case selection. The data will be released with the MIBE
> paper upon its acceptance.

## Reproducing the main results

```bash
# OmniGen2, 500 tasks, 3 seeds (Table 1)
cd round2 && bash run_shard.sh

# FLUX.2-klein-9B, 6/8 entities, 3 seeds (Table 2)
cd round2 && bash run_flux2_scaling.sh

# Ablations (Table 3)
cd round1_1 && python sweep.py
```

Each run writes per-task `records.jsonl` with SCR, DINO similarity, verifier
scores, `accepted_steps`, `gen_calls`, and a `step_log` for full
reproducibility. Aggregate with `round2/analyze.py`.

## Key results (3 seeds, 95% bootstrap CI)

OmniGen2, hard 4-entity slice (n=250 tasks × 3 seeds). SCR lower is better,
DINO identity similarity higher is better. These match Table 1 of the paper;
regenerate with `python3 round2/verify_paper_numbers.py`.

| Method | SCR | DINO |
|---|---|---|
| one-shot | 0.536 | 0.455 |
| best-of-8 | 0.492 | 0.492 |
| UMO (retrained SOTA) | 0.531 | 0.455 |
| **MIDC (ours)** | **0.470** | **0.509** |

MIDC triggers on only 28% of 8-entity tasks (the harder ones) and cuts SCR by
23.8% on that subset (0.688 → 0.524), while cleanly declining to act on the
remaining 72% (no measurable regression vs one-shot) — see §4.5 and Figure 5
of the paper.

## Notes

- The MIE verifier checkpoint is not committed (large binary); it is provided
  in the supplementary Code & Data package. Training data
  (`train_60k_v13_2.jsonl`) lives in [`../companion/prompt/`](../companion/prompt/)
  — see the note above.
- All headline metrics (SCR, DINO) use an **independent** DINOv2-based judge,
  not the verifier used inside MIDC, to avoid self-evaluation circularity.
- `HF_TOKEN` is required to access gated FLUX.2-klein-9B weights.
