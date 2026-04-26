#!/bin/bash

# Partial batch evaluation script for Gemini models

# Configuration
TRAIN_DATA="/Users/bytedance/Documents/multi_subject_generation/data/train_60k_v13_2.jsonl"
RESULTS_DIR="/Users/bytedance/Documents/multi_subject_generation/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Model_Training/vlm_multisubject_eval/results"

# Model versions to evaluate
MODELS=("gemini-2.5-flash" "gemini-3.1-flash-lite-preview")

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Run evaluation for each model with limited samples
for model in "${MODELS[@]}"; do
    echo "Running evaluation for model: $model"
    timestamp=$(date +"%Y%m%d_%H%M%S")
    output_file="$RESULTS_DIR/gemini_${model}_${timestamp}_partial.jsonl"

    python3 run_vlm_eval.py \
        --provider gemini \
        --gemini-model "$model" \
        --dataset "$TRAIN_DATA" \
        --output "$output_file" \
        --api-mode batch \
        --wait-for-batch \
        --batch-poll-seconds 60 \
        --max-samples 100  # Limit to 100 samples for testing

echo "Partial evaluation completed for model: $model"
echo "Results saved to: $output_file"
done

echo "All partial evaluations completed!"