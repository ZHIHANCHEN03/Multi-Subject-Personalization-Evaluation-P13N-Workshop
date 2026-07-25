# MIE Metric Evaluation Pipeline

This repository contains the evaluation pipeline for the MIE (Multi-subject Image Evaluation) metric. It is designed to export and analyze the scores from a trained MIE model checkpoint against image pairs.

## 1. Environment Setup

This pipeline is designed to run on Linux environments (e.g., RunPod / NVIDIA A100) and relies on the Unsloth framework for efficient inference.

You can use the provided bash script to automatically set up the environment and run the evaluation:

```bash
bash scripts/run_export_mie_scores.sh /path/to/your/manifest.jsonl
```

The script will:
1. Bootstrap an isolated Python virtual environment (`.venv-eval-export`).
2. Install pinned versions of PyTorch, Transformers, Peft, and Unsloth.
3. Automatically execute the `export_mie_scores.py` script.

## 2. Data Preparation

Due to size and privacy constraints, the generated images and reference images are not included in this repository. You must provide a manifest file (JSONL) that describes the image pairs to be evaluated.

### Manifest Format (`manifest.jsonl`)

Each line in the manifest should be a valid JSON object containing the following structure:

```json
{
  "task_id": "unique_task_identifier",
  "prompt": "A photo of subject A and subject B...",
  "subject_count": 2,
  "subject_refs": [
    {
      "id": "subject_A",
      "image_path": "/absolute/or/relative/path/to/ref_A.jpg"
    },
    {
      "id": "subject_B",
      "image_path": "/absolute/or/relative/path/to/ref_B.jpg"
    }
  ],
  "image_A_path": "/absolute/path/to/generated_image_model1.jpg",
  "image_B_path": "/absolute/path/to/generated_image_model2.png",
  "metadata": {
    "model_A": "model_1_name",
    "model_B": "model_2_name",
    "level": 2,
    "class_tag": "example_tag",
    "ratio_type": "1:1"
  }
}
```

## 3. Exporting Scores

To export MIE scores using a trained checkpoint:

```bash
# Using the wrapper script (Recommended)
bash scripts/run_export_mie_scores.sh /path/to/your/manifest.jsonl

# Or running Python directly (if environment is already set up)
python scripts/export_mie_scores.py \
  --model_name unsloth/Qwen3.5-0.8B \
  --mode layer_only \
  --checkpoint_dir /path/to/your/saved/checkpoint \
  --manifest_path /path/to/your/manifest.jsonl \
  --output_dir ./outputs
```

The script will output a new JSONL file containing the MIE evaluation scores (e.g., `outputs/manifest_mie_scores.jsonl`).

## 4. Analyzing the Results

Once the scores are exported, you can analyze them using the provided scripts:

```bash
# Analyze basic metrics distribution
python scripts/analyze_metrics_jsonl.py \
  --jsonl_path outputs/manifest_mie_scores.jsonl \
  --output_dir outputs/summaries

# Compare against human annotations (if available)
python scripts/analyze_metrics_vs_human.py \
  --scores_dir outputs/ \
  --human_csv /path/to/human_annotations.csv \
  --output_dir outputs/summaries
```

*Note: For plotting figures and generating alignment comparisons, you can use the other included scripts (`generate_metrics_alignment_figure.py` and `generate_pipeline_figure.py`), which will save charts into the `outputs/figures` directory.*
