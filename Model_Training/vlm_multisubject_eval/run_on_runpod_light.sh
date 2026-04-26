#!/bin/bash

# Lightweight script to run evaluation on existing Runpod environment

# Configuration
LOCAL_SCRIPT_DIR="/Users/bytedance/Documents/multi_subject_generation/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Model_Training/vlm_multisubject_eval"
REMOTE_HOST="root@64.247.206.95"
REMOTE_PORT="11878"
REMOTE_KEY="~/.ssh/id_ed25519_3"
REMOTE_WORKSPACE="/workspace/MOSAIC"

# Upload evaluation scripts
echo "Uploading evaluation scripts..."
scp -P $REMOTE_PORT -i $REMOTE_KEY $LOCAL_SCRIPT_DIR/run_vlm_eval.py $REMOTE_HOST:$REMOTE_WORKSPACE/
scp -P $REMOTE_PORT -i $REMOTE_KEY $LOCAL_SCRIPT_DIR/run_vlm_eval.py $REMOTE_HOST:$REMOTE_WORKSPACE/

# Run evaluation on Runpod
echo "Starting evaluation on Runpod..."
ssh -p $REMOTE_PORT -i $REMOTE_KEY $REMOTE_HOST << 'EOF'

cd $REMOTE_WORKSPACE

# Set API key
export GEMINI_API_KEY="your-api-key-here"

# Run evaluation for both models
for model in "gemini-2.5-flash" "gemini-3.1-flash-lite-preview"; do
    echo "Running evaluation for model: $model"
    timestamp=$(date +"%Y%m%d_%H%M%S")
    output_file="results/gemini_${model}_${timestamp}.jsonl"

    python3 run_vlm_eval.py \
        --provider gemini \
        --gemini-model "$model" \
        --dataset "/workspace/MOSAIC/train_60k_v13_2.jsonl" \
        --output "$output_file" \
        --api-mode batch \
        --wait-for-batch \
        --batch-poll-seconds 60 \
        --base-dir "/workspace/MOSAIC"

echo "Evaluation completed for model: $model"
done

EOF

echo "All evaluations completed on Runpod!"

# Download results
echo "Downloading results from Runpod..."
mkdir -p $LOCAL_SCRIPT_DIR/runpod_results
scp -P $REMOTE_PORT -i $REMOTE_KEY $REMOTE_HOST:$REMOTE_WORKSPACE/results/*.jsonl $LOCAL_SCRIPT_DIR/runpod_results/

echo "Results downloaded to: $LOCAL_SCRIPT_DIR/runpod_results/"