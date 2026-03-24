#!/bin/bash
# GCP Cloud Runner for WorldModelLens
# Usage: ./run_gcp.sh [PROJECT] [REGION] [MACHINE_TYPE] [n_gpus]

set -e

PROJECT="${1:-my-project}"
REGION="${2:-us-central1}"
MACHINE_TYPE="${3:-a2-highgpu-1g}"
N_GPUS="${4:-1}"

echo "Starting WorldModelLens on GCP..."
echo "Project: $PROJECT"
echo "Region: $REGION"
echo "Machine: $MACHINE_TYPE"

# Create instance
gcloud compute instances create wml-runner \
    --project="$PROJECT" \
    --zone="$REGION-a" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="type=nvidia-tesla-a100,count=$N_GPUS" \
    --image-family=tf-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --scopes=cloud-storage-read-write,storage-full

# Wait for instance
echo "Waiting for instance to be ready..."
gcloud compute ssh wml-runner --project="$PROJECT" --zone="$REGION-a" --command="pip install world-model-lens"

# Run analysis
echo "Running analysis..."
gcloud compute ssh wml-runner --project="$PROJECT" --zone="$REGION-a" --command="
    python -c \"
    from world_model_lens import HookedWorldModel
    print('WorldModelLens ready')
    \"
"

echo "Done! Connect with: gcloud compute ssh wml-runner --zone=$REGION-a"
