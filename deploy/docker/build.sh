#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Building World Model Lens Docker Images..."
echo "=========================================="

echo ""
echo "1. Building CPU image..."
docker build \
    -f deploy/docker/Dockerfile.cpu \
    -t world-model-lens:cpu \
    .

echo ""
echo "2. Building GPU image..."
docker build \
    -f deploy/docker/Dockerfile.gpu \
    -t world-model-lens:gpu \
    .

echo ""
echo "3. Building A100-optimized image..."
docker build \
    -f deploy/docker/Dockerfile.a100 \
    -t world-model-lens:a100 \
    .

echo ""
echo "=========================================="
echo "Build complete! Images created:"
echo "  - world-model-lens:cpu"
echo "  - world-model-lens:gpu"
echo "  - world-model-lens:a100"
echo ""
echo "To run the API server:"
echo "  docker run -p 8000:8000 world-model-lens:cpu"
