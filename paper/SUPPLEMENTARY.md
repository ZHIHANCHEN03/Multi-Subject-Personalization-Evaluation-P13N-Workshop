# Supplementary Materials — Availability

This document lists the artifacts accompanying the MIDC submission and how reviewers can access them.

## 1. Code

The full implementation of MIDC (MIE-guided Inference-time Diagnosis and Correction), the
calibrated routing loop, the guarded acceptance criterion, all baselines
(`one_shot`, `best_of_n`, `UMO`), the FLUX.2-klein adapter, the multi-seed sharded
runner, the merge/analysis scripts, and the statistical tests (paired bootstrap,
win/tie/loss) is included in the anonymized supplementary code archive and will be
released publicly upon acceptance under an MIT license.

Top-level layout of the repo:

```
round1/                      # core library: common.py, external_generators.py, MIE verifier, MIDC loop
round2/                      # experiment runners (Round 2 = OmniGen2 main, Round 3 = FLUX.2 scaling)
  p2_oneshot.py  p3_bestofn.py  p4_umo.py  p5_midc.py
  run_shard.sh   merge_shards.py   analyze.py
  run_flux2_scaling.sh  calibrate_flux2.sh  score_mie_precomputed.py
  results_r2/                # raw per-task records.jsonl (committed)
round2/README_round3.md      # Round 3 (FLUX.2 scaling) documentation
paper/                       # LaTeX source, figures, refs.bib, ReproducibilityChecklist
```

## 2. Data

- **MIB-Gold hard splits** (the 500-task OmniGen2 split and the 6/8-entity FLUX.2 splits)
  are derived from MIBE [anon2025mibe], which is currently under review. The task
  manifests (prompts + reference image paths + entity metadata) are included in the
  supplementary archive under `round2/manifests/`. The underlying reference images
  will be released under a CC-BY-4.0 license upon publication of MIBE.
- **Raw per-task results** for every reported number in Table 1, Table 2, and the
  gating analysis (Table 3) are committed under `round2/results_r2/` as
  `records.jsonl` files, one per (method, seed, shard). These are the exact artifacts
  used by `round2/analyze.py` to produce the tables.

## 3. MIE Verifier Checkpoint

The MIE verifier is a LoRA-fine-tuned Qwen2-VL-2B model. The checkpoint
(`mie_verifier_lora.safetensors`, ~50 MB) is **not** bundled in the supplementary
archive to keep the submission size within the AAAI limit, but it will be released
publicly upon acceptance via the companion MIBE paper's release. For reviewer
convenience during the review period, the checkpoint is available upon request to
the area chair (the release is gated only to preserve double-blind review; the
checkpoint itself contains no identifying information).

The verifier can be reproduced from scratch by running
`round1/train_mie_verifier.py` on the MIBE Silver Set (60K pairs); training takes
~2 hours on a single A100. Full training hyperparameters are listed in
`round1/train_mie_verifier.py` and in the Reproducibility Checklist.

## 4. Pre-trained Base Models

All base models are publicly available and used under their respective licenses:

| Model | Source | License |
|---|---|---|
| OmniGen2 [wu2025omnigen2] | HuggingFace `OmniGen2/OmniGen2` | Apache 2.0 |
| FLUX.2-klein-9B | HuggingFace `black-forest-labs/FLUX.2-klein-9B` | FLUX.2 Non-Commercial |
| UMO LoRA [cheng2025umo] | project release `UMO_OmniGen2.safetensors` | MIT |
| DINOv2 [oquab2023dinov2] | HuggingFace `facebook/dinov2-large` | Apache 2.0 |
| Qwen2-VL-2B | HuggingFace `Qwen/Qwen2-VL-2B` | Tongyi Qianwen |

## 5. Computing Infrastructure

All experiments were run on a single A100 80GB GPU (OmniGen2 main experiments) and
4× A100 80GB GPUs (FLUX.2 scaling, 6/8-entity splits). Software: Python 3.10,
PyTorch 2.3, diffusers 0.30, peft 0.11, transformers 4.44, safetensors 0.4.
OS: Ubuntu 22.04. Full version pins are in `requirements.txt`.

## 6. UMO LoRA Loading Verification

Because UMO [cheng2025umo] is applied to OmniGen2 via a context-refiner LoRA, a
natural concern is whether the LoRA actually loads and takes effect, or silently
degrades to base OmniGen2 (which would invalidate the UMO baseline). We verified
correct loading with two independent checks:

**Key-match check.** We instantiate the OmniGen2 transformer, inject a PEFT LoRA
adapter with the exact configuration used by our runner (`r=512`,
`target_modules=["to_k","to_q","to_v","to_out.0"]`,
`init_lora_weights="gaussian"`), and load
`UMO_OmniGen2.safetensors` via `load_state_dict(strict=False)`. Result:
`unexpected_keys = 0` — all 304 UMO LoRA keys (288 main transformer-block keys +
16 context-refiner keys) match the adapter's parameter names and load
successfully. The `missing_keys` are only the transformer-block LoRA weights that
UMO did not train (which remain at the gaussian init, i.e. near-identity) — this
is the expected behavior, not a failure. Script: `misc/umo_keycheck.py`.

**Pixel-diff check.** As a stronger, end-to-end test, we generate the same task
with the same seed and `image_guidance_scale=2.0` using (a) base OmniGen2 (no
LoRA) and (b) UMO (LoRA fused via `fuse_lora` then `unload_lora`). The mean
absolute pixel difference between the two outputs is **10.44** (on 0–255 scale,
512×512, 5 inference steps), far above the ~0.5 threshold for "no effect" and the
~2.0 threshold for "minor effect." This confirms the UMO LoRA has a substantial,
observable effect on generation — UMO is a genuine retrained model, not a silent
no-op over base OmniGen2. Script: `misc/umo_pixeldiff_fast.py`.

Together these two checks rule out the "silent failure" hypothesis: the UMO
baseline numbers in Table 1 reflect the real UMO model.

## 7. Reproducibility Statement

Every reported number is the mean over 3 random seeds (0, 1, 2). Per-task raw
records, paired bootstrap CIs, and win/tie/loss counts are all derivable from the
committed `records.jsonl` files via `round2/analyze.py`. Estimated wall-clock cost
to reproduce all tables: ~120 A100-hours.
