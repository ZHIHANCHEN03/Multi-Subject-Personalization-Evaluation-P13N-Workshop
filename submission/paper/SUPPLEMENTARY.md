# Supplementary materials — MIDC

What accompanies the submission and how a reviewer can check each reported number.
Everything referenced here is in the supplementary archive; no claim in the paper
depends on material that is not.

## 1. Reproducing every number in the paper

One script recomputes Tables 1–4 and the gating figure from the committed per-task
records and diffs the result against the values in `main.tex`. No GPU, no model
weights, no network:

```bash
cd round2
python3 verify_paper_numbers.py          # exits non-zero on any mismatch
```

It currently reports *all paper numbers reproduce from the committed records*. This
is the fastest way to audit the submission, and it is why the raw records are
included rather than summary tables alone.

The three late additions are checkable the same way, from their own record files:

| Claim | Recompute from |
|---|---|
| Sharpness-matched blur control (SCR $+0.013$, DINO $-0.008$, $n{=}150$) | `round2/results_blur_cf/blur_cf.jsonl` |
| CLIP-T / CLIP-I ($n{=}500$ per method, seed 0) | `round2/results_clip/clip_*_s0.jsonl` |
| Human-eval robustness ($82.9\%$ at $\ge2$ labelers, $87.0\%$ unanimous) | `round2/human_eval/HUMAN_EVAL/` + `aggregate_human_eval.py` |

## 2. Code

```
round1/                     core library and the MIDC loop
  common.py                 task loading, DinoScorer (Grounding-DINO + DINOv2), SCR,
                            the dataset driver, the MIE subprocess client
  p1_ours_v2.py             MIDC: calibrated routing, dual-signal diagnosis,
                            action portfolio, guarded acceptance
  p2_oneshot.py             one_shot baseline
  p3_bestofn.py             best_of_n baseline
  p4_umo.py                 UMO baseline (released LoRA, its official config)
  actions.py                CalibratedRouter / RawRouter, prompt rewrite, refset ops
  generators.py             OmniGen2 and FLUX.2 adapters
  external_generators.py    UMO and FreeGraftor wrappers
  mie_server.py             persistent MIE verifier, separate venv (see §4)
  calibrate_mie.py          freezes the per-facet calibration statistics
  requirements.txt          generation-side dependencies

round2/                     experiments and analysis
  run_round2_main.sh        500-task x 3-seed sharded run
  run_shard.sh              one shard of the four methods on one GPU
  run_flux2_scaling.sh      FLUX.2 6/8-subject scaling
  run_ablation.sh           the six ablation variants
  merge_shards.py           merge per-shard records
  analyze.py                means, bootstrap CIs, paired significance
  verify_paper_numbers.py   recompute every table, diff against main.tex
  b1_reanalysis.py          the gating (triggered vs no-op) analysis
  blur_counterfactual.py    the sharpness-matched control of Sec. 4.2
  score_clip.py             CLIP-T / CLIP-I scoring
  aggregate_human_eval.py   human-eval win rates, CIs, Fleiss kappa
  export_human_eval.py      builds the blinded A/B pairs and the key
  analyze_umo_vs_oneshot.py the UMO-vs-one_shot ballot analysis
  score_precomputed.py      score externally generated images by task_id
  plot_pipeline.py          Figure 2
  plot_umo_qualitative.py   Figure 3
  plot_gating.py            Figure 5
```

Every generated figure is built from the records by a `plot_*.py` script, so no
figure carries a number that is not in the data.

## 3. Data

- **Task manifests** — prompts, reference filenames and entity metadata for each
  split, under `round2/results_r2/manifests/` (500-task OmniGen2) and
  `round2/results_flux2/manifests/` (6/8-entity). These are the exact runner inputs.
- **Raw per-task records** — `records.jsonl` under `round2/results_r2/merged/`
  (4 methods x 3 seeds), `round2/results_flux2/` (3 methods x 2 entity counts x
  3 seeds) and `round2/results_ablation/` (6 variants x 2 seeds). One line per task
  carrying `scr`, `dino_sims`, `dino_mean`, `gen_calls`, `accepted_steps` and the
  full `step_log` of routing decisions, which is what makes the gating analysis and
  Figure 2 auditable.
- **Human evaluation** — `round2/human_eval/HUMAN_EVAL/` holds the three labelers'
  raw ballots, the blinding key (`key.json`, required to decode LEFT/RIGHT) and the
  aggregate; `round2/human_eval/UMO_VS_ONESHOT/` holds the second study. Including
  the key is what makes Table 4 independently checkable rather than a quoted
  summary.
- **Reference images** — from MIB-Gold [anon2025mibe], under review separately. The
  manifests carry the filenames; the images are released with that benchmark.

## 4. Environments

MIE needs Unsloth with a Qwen vision stack whose `torch`/`transformers` pins
conflict with OmniGen2 and FLUX.2, so the verifier runs in its own virtualenv and is
driven over a JSON-lines pipe (`round1/mie_server.py`). Two consequences worth
knowing when re-running it: `unsloth` must be imported before `transformers`, and
stdout is the protocol channel, so all diagnostics go to stderr.
`round1/requirements.txt` lists the generation side. Hardware and library versions
for every reported run are in Sec. 4.1 of the paper.

## 5. MIE verifier checkpoint

The decomposed verifier is a Qwen3.5-4B LoRA from MIBE [anon2025mibe]. It is used
only for in-loop routing and acceptance: **no reported metric depends on it.** SCR,
DINO, CLIP-I and CLIP-T are computed entirely by code included here, and the human
studies do not involve it. Reviewers can therefore reproduce every number in the
paper without the checkpoint; it is needed only to re-run the correction loop, and
is released with MIBE.

## 6. Deliberately not included

- **Generated images** (~8 GB across all cells). The records carry every per-subject
  similarity and score computed from them, which is what the tables require.
- **The MIE checkpoint**, per §5.
- **Reference images**, per §3.
