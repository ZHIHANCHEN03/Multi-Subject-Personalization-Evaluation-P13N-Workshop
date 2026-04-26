#!/bin/bash

# Remote launch script for Runpod evaluation

# Configuration
REMOTE_HOST="root@64.247.206.95"
REMOTE_PORT="11878"
REMOTE_KEY="~/.ssh/id_ed25519_3"
REMOTE_WORKSPACE="/workspace/MOSAIC"

# Create remote script
cat << 'EOF' > remote_eval.sh
#!/bin/bash

cd /workspace/MOSAIC

# Set API key
export GEMINI_API_KEY="your-api-key-here"

# Create results directory
mkdir -p results

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

echo "All evaluations completed!"
EOF

# Upload script and run evaluation
echo "Uploading script to Runpod..."
scp -P $REMOTE_PORT -i $REMOTE_KEY remote_eval.sh $REMOTE_HOST:$REMOTE_WORKSPACE/
scp -P $REMOTE_PORT -i $REMOTE_KEY run_vlm_eval.py $REMOTE_HOST:$REMOTE_WORKSPACE/

# Run evaluation in background
echo "Starting evaluation on Runpod..."
ssh -p $REMOTE_PORT -i $REMOTE_KEY $REMOTE_HOST "cd $REMOTE_WORKSPACE && chmod +x remote_eval.sh && nohup ./remote_eval.sh > eval_output.log 2>&1 &"

# Clean up
rm remote_eval.sh

echo "Evaluation started on Runpod in background!"
echo "Check progress with: ssh -p $REMOTE_PORT -i $REMOTE_KEY $REMOTE_HOST 'tail -f $REMOTE_WORKSPACE/eval_output.log'"
echo "Results will be in: $REMOTE_WORKSPACE/results/"