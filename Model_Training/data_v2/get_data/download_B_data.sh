#!/bin/bash

set -euo pipefail

# Download B model data from Runpod and flatten everything into LOCAL_B_DIR.

# Configuration
REMOTE_HOST="root@64.247.206.95"
REMOTE_PORT="11878"
REMOTE_KEY="$HOME/.ssh/id_ed25519_3"
LOCAL_B_DIR="/Users/bytedance/Documents/multi_subject_generation/data/B"

mkdir -p "$LOCAL_B_DIR"

STAGING_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGING_DIR"' EXIT

echo "Starting download of B model data from Runpod..."
echo "Local flat target directory: $LOCAL_B_DIR"
echo "Temporary staging directory: $STAGING_DIR"

download_and_flatten_directory() {
    local remote_path="$1"
    local staging_subdir="$2"

    echo "Downloading: $remote_path"

    mkdir -p "$staging_subdir"
    rsync -avz \
        -e "ssh -p $REMOTE_PORT -i $REMOTE_KEY" \
        "$REMOTE_HOST:$remote_path/" \
        "$staging_subdir/"

    find "$staging_subdir" -type f \
        \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.webp" \) \
        -print0 | while IFS= read -r -d '' file_path; do
            file_name="$(basename "$file_path")"
            cp -f "$file_path" "$LOCAL_B_DIR/$file_name"
        done
}

echo "Downloading outputs_v13_2_part_XX directories..."
for idx in $(seq 1 16); do
    part="$(printf "%02d" "$idx")"
    remote_dir="/workspace/MOSAIC/outputs_v13_2_part_${part}"
    download_and_flatten_directory "$remote_dir" "$STAGING_DIR/outputs_v13_2_part_${part}"
done

echo "Downloading rerun_outputs_part_XX directories..."
for idx in $(seq 1 16); do
    part="$(printf "%02d" "$idx")"
    remote_dir="/workspace/MOSAIC/rerun_outputs_part_${part}"
    download_and_flatten_directory "$remote_dir" "$STAGING_DIR/rerun_outputs_part_${part}"
done

echo "Download completed!"
echo "All B model data has been flattened into: $LOCAL_B_DIR"

echo "Verifying download..."
local_file_count=$(find "$LOCAL_B_DIR" -maxdepth 1 -type f | wc -l)
echo "Total files in flat target directory: $local_file_count"

png_count=$(find "$LOCAL_B_DIR" -maxdepth 1 -iname "*.png" | wc -l)
jpg_count=$(find "$LOCAL_B_DIR" -maxdepth 1 \( -iname "*.jpg" -o -iname "*.jpeg" \) | wc -l)
webp_count=$(find "$LOCAL_B_DIR" -maxdepth 1 -iname "*.webp" | wc -l)
echo "PNG files: $png_count"
echo "JPG/JPEG files: $jpg_count"
echo "WEBP files: $webp_count"
