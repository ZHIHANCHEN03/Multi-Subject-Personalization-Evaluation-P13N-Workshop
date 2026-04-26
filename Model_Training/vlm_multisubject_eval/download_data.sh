#!/bin/bash

# Data download script for A and B models

# Configuration
A_DIR="/Users/bytedance/Documents/multi_subject_generation/data/A"
B_DIR="/Users/bytedance/Documents/multi_subject_generation/data/B"

# Create directories if they don't exist
mkdir -p "$A_DIR"
mkdir -p "$B_DIR"

echo "Starting data download..."

# Download A model data from Google Drive
echo "Downloading A model data from Google Drive..."
# Install gdown if not available
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install gdown
fi

# Download the entire folder
gdown "https://drive.google.com/drive/folders/1nFtOyJOcuAryz6rGpyyduMnX693xW4j4" -O "$A_DIR" --folder

# Download B model data from Runpod
echo "Downloading B model data from Runpod..."

# Function to download from Runpod
runpod_download() {
    local remote_path="$1"
    local local_path="$2"
    echo "Downloading $remote_path to $local_path..."
    scp -r -P 11878 -i ~/.ssh/id_ed25519_3 root@64.247.206.95:"$remote_path" "$local_path"
}

# Download outputs_v13_2_part_*
for part in {1..4}; do
    runpod_download "/workspace/MOSAIC/outputs_v13_2_part_${part}" "$B_DIR/"
done

# Download rerun_outputs_part_*
for part in {1..4}; do
    runpod_download "/workspace/MOSAIC/rerun_outputs_part_${part}" "$B_DIR/"
done

echo "Data download completed!"
echo "A model data saved to: $A_DIR"
echo "B model data saved to: $B_DIR"