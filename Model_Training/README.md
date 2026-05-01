# Model_Training Public Release

This directory contains the public `data_v2` training and evaluation pipeline for the `LENS` metric.

## Overview

This release keeps the current paper-facing `v2` workflow only:

- `scripts/build_v2_dataset.py`
  - Builds `train_v2.json`, `val_v2.json`, and `test_v2.json`
  - Merges prompt metadata, reference images, generated images, and LLM labels
- `scripts/train.py`
  - Trains LENS in `layer_only` or `lora_layer` mode
- `scripts/evaluate_pipeline.py`
  - Loads a saved checkpoint and evaluates LENS against CLIP and DINO baselines
- `run_a100_pipeline.sh`
  - End-to-end RunPod / A100 entrypoint
  - Installs dependencies, builds manifests, trains, evaluates, and writes outputs

Legacy `data_v1`, local-only utilities, cached virtual environments, and old checkpoints have been removed from this public release.

## Directory Layout

The public repository keeps only the files needed for the current `v2` pipeline:

```text
Model_Training/
  data_v2/
    60k_LLM_Result/
    prompt/
    refs/
  docs/
  lens/
  scripts/
  README.md
  run_a100_pipeline.sh
```

## Expected Data Layout

Reference images stay inside the repository:

```text
Model_Training/data_v2/refs/
```

Generated images are expected on the training machine:

```text
/workspace/data/A
/workspace/data/B
```

By default, the public release assumes:

- A images: `/workspace/data/A/<task_id>.png`
- B images: `/workspace/data/B/<task_id>.jpg`

The builder also supports extension fallback for:

- `.png`
- `.jpg`
- `.jpeg`
- `.webp`

If an image pair is missing, that sample is dropped during manifest building.

## Label Processing Rules

The `v2` builder uses the following rules:

- Reads prompts from `data_v2/prompt/train_60k_v13_2.jsonl`
- Reads labels from:
  - `data_v2/60k_LLM_Result/2_5_merged_sorted.jsonl`
  - `data_v2/60k_LLM_Result/3_1_merged_sorted.jsonl`
- Drops a sample if the two label sources disagree on `winner / preference`
- Averages `category_scores_A` and `category_scores_B` across label sources
- Drops a sample if generated images are missing
- Drops a sample if required reference images are missing
- Splits train / val / test by `seed_id` groups to reduce leakage across prompt families

## Environment Assumptions

This pipeline is designed for:

- Linux / RunPod
- NVIDIA A100 80GB
- CUDA training with Unsloth

The main script creates a local virtual environment under:

```text
Model_Training/.venv-a100-unsloth
```

Hugging Face cache stays on the local overlay disk:

```text
/root/huggingface_cache
```

Logs and checkpoints are written to:

```text
/workspace/Model_Training_runs
```

## Build V2 Manifests

```bash
cd Model_Training
python scripts/build_v2_dataset.py \
  --image_a_root /workspace/data/A \
  --image_b_root /workspace/data/B
```

This writes:

- `data_v2/train_v2.json`
- `data_v2/val_v2.json`
- `data_v2/test_v2.json`

Useful builder output includes:

- total prompt count
- usable sample count
- dropped count for missing labels
- dropped count for preference conflicts
- dropped count for missing generated images
- dropped count for missing reference images

## Recommended Training Entry

Recommended first run:

```bash
cd Model_Training
MODEL_NAME=unsloth/Qwen3.5-0.8B \
RUN_LAYER_ONLY=1 \
RUN_LORA_LAYER=0 \
bash run_a100_pipeline.sh
```

This script will:

1. Create or reuse the virtual environment
2. Install pinned dependencies
3. Build `train_v2.json`, `val_v2.json`, and `test_v2.json`
4. Train the selected mode
5. Save checkpoints
6. Run evaluation

## Training Modes

The public release supports:

- `layer_only`
  - Unfreezes the last transformer layers
  - Recommended default
- `lora_layer`
  - Uses LoRA adapters and also unfreezes the last transformer layers

`run_a100_pipeline.sh` defaults to:

- `DATA_VERSION=v2`
- `RUN_LAYER_ONLY=1`
- `RUN_LORA_LAYER=0`

## Paper-Friendly Default Protocol

The current defaults are set to be more reproducible and more reviewer-friendly:

- `seed = 3407`
- `lr = 2e-5`
- `weight_decay = 0.01`
- `warmup_ratio = 0.03`
- `alpha = 1.0`
- `beta = 0.5`
- `adaptive_beta = false`
- `target effective batch size = 16`
- `target optimizer updates = 6000`
- `auto_scale_min_epochs = 2`
- `auto_scale_max_epochs = 4`

These defaults are intended to keep training budgets comparable across model sizes and modes while staying reasonably strong in practice.

## Output Layout

Each run writes to:

```text
/workspace/Model_Training_runs/<data_version>/<model_name>/<timestamp>/
```

Inside each run directory:

- `logs/`
- `outputs/`

Typical checkpoint directories look like:

```text
/workspace/Model_Training_runs/v2/unsloth_Qwen3.5-0.8B/<timestamp>/outputs/unsloth_Qwen3.5-0.8B-layer_only-best
/workspace/Model_Training_runs/v2/unsloth_Qwen3.5-0.8B/<timestamp>/outputs/unsloth_Qwen3.5-0.8B-layer_only-epoch1
```

Each checkpoint contains some or all of:

- `lens_heads.pt`
- `lens_config.json`
- `trainable_backbone.pt`
- `lora_adapter/`

## Evaluate A Saved Checkpoint

Example:

```bash
cd Model_Training
python scripts/evaluate_pipeline.py \
  --model_name unsloth/Qwen3.5-0.8B \
  --mode layer_only \
  --test_path ./data_v2/test_v2.json \
  --checkpoint_dir /workspace/Model_Training_runs/v2/unsloth_Qwen3.5-0.8B/<timestamp>/outputs/unsloth_Qwen3.5-0.8B-layer_only-best \
  --outputs_dir /workspace/Model_Training_runs/v2/unsloth_Qwen3.5-0.8B/<timestamp>/outputs
```

If `--checkpoint_dir` is omitted, the evaluator will:

- prefer `<safe_model_name>-<mode>-best`
- otherwise fall back to the latest saved epoch checkpoint

## Multi-Run Examples

Run `0.8B + layer_only`:

```bash
cd Model_Training
MODEL_NAME=unsloth/Qwen3.5-0.8B RUN_LAYER_ONLY=1 RUN_LORA_LAYER=0 bash run_a100_pipeline.sh
```

Run `0.8B + lora_layer`:

```bash
cd Model_Training
MODEL_NAME=unsloth/Qwen3.5-0.8B RUN_LAYER_ONLY=0 RUN_LORA_LAYER=1 bash run_a100_pipeline.sh
```

Run `2B + layer_only`:

```bash
cd Model_Training
MODEL_NAME=unsloth/Qwen3.5-2B RUN_LAYER_ONLY=1 RUN_LORA_LAYER=0 bash run_a100_pipeline.sh
```

Run `2B + lora_layer`:

```bash
cd Model_Training
MODEL_NAME=unsloth/Qwen3.5-2B RUN_LAYER_ONLY=0 RUN_LORA_LAYER=1 bash run_a100_pipeline.sh
```

Run `4B + layer_only`:

```bash
cd Model_Training
MODEL_NAME=unsloth/Qwen3.5-4B RUN_LAYER_ONLY=1 RUN_LORA_LAYER=0 bash run_a100_pipeline.sh
```

Run `4B + lora_layer`:

```bash
cd Model_Training
MODEL_NAME=unsloth/Qwen3.5-4B RUN_LAYER_ONLY=0 RUN_LORA_LAYER=1 bash run_a100_pipeline.sh
```

Run `9B + layer_only`:

```bash
cd Model_Training
MODEL_NAME=unsloth/Qwen3.5-9B RUN_LAYER_ONLY=1 RUN_LORA_LAYER=0 bash run_a100_pipeline.sh
```

Run `9B + lora_layer`:

```bash
cd Model_Training
MODEL_NAME=unsloth/Qwen3.5-9B RUN_LAYER_ONLY=0 RUN_LORA_LAYER=1 bash run_a100_pipeline.sh
```

## Sanity Checks Before Training

Recommended checks on the training machine:

```bash
find /workspace/data/A -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.webp' \) ! -name '._*' | wc -l
find /workspace/data/B -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.webp' \) ! -name '._*' | wc -l
```

If your data transfer came from macOS, clean AppleDouble sidecar files:

```bash
find /workspace/data/A /workspace/data/B -type f -name '._*' -delete
```

## Notes

- This public release assumes `data_v2` only.
- Generated images are not stored inside the repository.
- Hugging Face cache stays on `/root` by default.
- Logs and checkpoints stay on `/workspace`.
- Reference images remain inside `data_v2/refs`.
