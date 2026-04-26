#!/bin/bash

# Full batch evaluation script for Gemini models

# Configuration
TRAIN_DATA="/Users/bytedance/Documents/multi_subject_generation/data/train_60k_v13_2.jsonl"
RESULTS_DIR="/Users/bytedance/Documents/multi_subject_generation/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Model_Training/vlm_multisubject_eval/results"

# Model versions to evaluate
MODELS=("gemini-2.5-flash" "gemini-3.1-flash-lite-preview")

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Function to download Google Drive files
drive_download() {
    local file_id="$1"
    local output_path="$2"
    echo "Downloading $file_id to $output_path..."
    gdown "https://drive.google.com/uc?id=$file_id" -O "$output_path"
}

# Function to download from Runpodunpod_download() {
    local remote_path="$1"
    local local_path="$2"
    echo "Downloading from Runpod: $remote_path to $local_path..."
    scp -P 11878 -i ~/.ssh/id_ed25519_3 root@64.247.206.95:"$remote_path" "$local_path"
}

# Download A model data from Google Drive
echo "Downloading A model data..."
A_DIR="/Users/bytedance/Documents/multi_subject_generation/data/A"
mkdir -p "$A_DIR"

# Example download for part1_1_10000/batch_01_0001_0100/1.png
# You'll need to update this with the actual file IDs or use gdown to download folders
# drive_download "FILE_ID" "$A_DIR/1.png"

# Alternative: Download entire folder using gdown
# gdown "https://drive.google.com/drive/folders/1nFtOyJOcuAryz6rGpyyduMnX693xW4j4" -O "$A_DIR" --folder

# Download B model data from Runpod
echo "Downloading B model data..."
B_DIR="/Users/bytedance/Documents/multi_subject_generation/data/B"
mkdir -p "$B_DIR"

# Download outputs_v13_2_part_*
for part in {1..4}; do
    runpod_download "/workspace/MOSAIC/outputs_v13_2_part_${part}" "$B_DIR/outputs_v13_2_part_${part}"
done

# Download rerun_outputs_part_*
for part in {1..4}; do
    runpod_download "/workspace/MOSAIC/rerun_outputs_part_${part}" "$B_DIR/rerun_outputs_part_${part}"
done

# Run evaluation for each model
for model in "${MODELS[@]}"; do
    echo "Running evaluation for model: $model"
    timestamp=$(date +"%Y%m%d_%H%M%S")
    output_file="$RESULTS_DIR/gemini_${model}_${timestamp}.jsonl"

    python3 run_vlm_eval.py \
        --provider gemini \
        --gemini-model "$model" \
        --dataset "$TRAIN_DATA" \
        --output "$output_file" \
        --api-mode batch \
        --wait-for-batch \
        --batch-poll-seconds 60

echo "Evaluation completed for model: $model"
echo "Results saved to: $output_file"
done

echo "All evaluations completed!"