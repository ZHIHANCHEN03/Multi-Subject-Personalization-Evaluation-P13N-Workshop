# MIE Metric Model Training Pipeline

This repository contains the training and evaluation pipeline for the MIE metric.

## 1. Environment Setup

This pipeline is designed for Linux / NVIDIA A100 (80GB) environments. It uses an isolated virtual environment and installs pinned dependencies automatically via the provided shell script.

## 2. Data Preparation

Due to size constraints and to ensure privacy, the dataset is not included in this repository. Please prepare your data according to the following structure before running the pipeline.

### 2.1 Directory Structure

Create a `data_v2` directory in the root of the project with the following layout:

```text
Model_Training_Paper_Coding/
  ├── data_v2/
  │   ├── prompt/
  │   │   └── train_60k_v13_2.jsonl
  │   ├── 60k_LLM_Result/
  │   │   ├── 2_5_merged_sorted.jsonl
  │   │   └── 3_1_merged_sorted.jsonl
  │   └── refs/
  │       ├── subject1.jpg
  │       ├── subject2.jpg
  │       └── ...
  ├── mie/
  ├── scripts/
  ├── run_a100_pipeline.sh
  └── README.md
```

### 2.2 Image Data

The generated images should be placed on the training machine (default paths are under `/workspace`):

- **Image A (Strong model / high quality)**: `/workspace/data/A/<task_id>.png`
- **Image B (Weak model / low quality)**: `/workspace/data/B/<task_id>.jpg`

*Note: The script supports fallback extensions (`.jpg`, `.jpeg`, `.webp`, `.png`).*

### 2.3 Data Formats

**Prompt JSONL (`data_v2/prompt/train_60k_v13_2.jsonl`)**:
Each line should be a valid JSON object containing:
```json
{
  "id": 12345,
  "prompt_en": "A photo of subject1 and subject2...",
  "people_names": ["subject1"],
  "object_names": ["subject2"],
  "total_entities": 2,
  "level": 2,
  "class_tag": "tag",
  "ratio_type": "1:1",
  "n_humans": 1,
  "n_objects": 1
}
```

**Label JSONL (`data_v2/60k_LLM_Result/*.jsonl`)**:
Each line should be a valid JSON object representing the teacher VLM's evaluation:
```json
{
  "task_id": "12345",
  "winner": "A",
  "a_existence": 1.0,
  "a_appearance": 1.0,
  "a_interaction": 1.0,
  "b_existence": 0.0,
  "b_appearance": 0.0,
  "b_interaction": 0.0,
  "prompt": "A photo of subject1 and subject2...",
  "subject_count": 2,
  "metadata": {}
}
```

## 3. Running the Pipeline

You can run the end-to-end pipeline using the provided bash script. It will automatically build the dataset manifests, train the model, and run evaluation.

```bash
# First run: build the dataset manifests and start training
bash run_a100_pipeline.sh
```

If you already have the dataset manifests (`train_v2.json`, `val_v2.json`, `test_v2.json`) built and want to skip the build step for subsequent runs:

```bash
SKIP_BUILD=1 MODEL_NAME=unsloth/Qwen3.5-0.8B RUN_LAYER_ONLY=1 bash run_a100_pipeline.sh
```

### Training Configuration
The default script configuration uses reproducible and reviewer-friendly settings:
- **Seed**: 3407
- **Learning Rate**: 2e-5
- **Target Effective Batch Size**: 16
- **Max Optimizer Steps**: 600

Outputs, logs, and model checkpoints will be saved in `/workspace/Model_Training_runs/`.
