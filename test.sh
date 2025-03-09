#!/bin/bash

# Set Docker image, Hugging Face cache, and output directory
IMAGE_NAME="ghcr.io/codysnider/flux"
TAG="latest"
HF_CACHE="/data/models"
OUTPUT_DIR="$(pwd)/output"
OUTPUT_FILE="$OUTPUT_DIR/test_image.png"

# Required commands
REQUIRED_CMDS=("docker" "file" "identify" "bc")

# Check for required dependencies
for cmd in "${REQUIRED_CMDS[@]}"; do
    if ! command -v $cmd &> /dev/null; then
        echo "Error: '$cmd' is required but not installed."
        exit 1
    fi
done

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Define test input
PROMPT="A potato farmer holding a sign that says 'Flux on a potato'"

# Run inference inside the container with HF cache volume
echo "Running inference..."
START_TIME=$(date +%s.%N)
docker run --rm --gpus all \
    -v "$HF_CACHE:/app/models" \
    -e HF_TOKEN="$HF_TOKEN" \
    -v "$OUTPUT_DIR:/app/output" \
    $IMAGE_NAME:$TAG \
    --prompt "$PROMPT" --output "/app/output/test_image.png"
END_TIME=$(date +%s.%N)

# Calculate execution time
EXECUTION_TIME=$(echo "$END_TIME - $START_TIME" | bc)
echo "Time to generate: ${EXECUTION_TIME} seconds"

# Check if output file exists
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Test Failed: Output file was not created."
    exit 1
fi

# Validate image file format
if ! file "$OUTPUT_FILE" | grep -q "PNG image data"; then
    echo "Test Failed: Output file is not a valid PNG image."
    rm -f "$OUTPUT_FILE"
    exit 1
fi

# Validate 512x512 image dimensions
DIMENSIONS=$(identify -format "%wx%h" "$OUTPUT_FILE")
if [ "$DIMENSIONS" != "512x512" ]; then
    echo "Test Failed: 512x512 image has incorrect dimensions ($DIMENSIONS instead of 512x512)."
    exit 1
fi

echo "Validation passed: 512x512 PNG file was successfully generated."

# Clean up files
rm -f "$OUTPUT_FILE"
rmdir output
exit 0
