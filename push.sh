#!/bin/bash

# GitHub Container Registry configuration
GHCR_USERNAME="codysnider"
IMAGE_NAME="ghcr.io/codysnider/flux"
TAG="latest"

# Run the test script
# bash test.sh

# Check if the test passed
if [ $? -eq 0 ]; then
    echo "Tests passed. Pushing to GitHub Container Registry..."

    # Authenticate with GitHub Packages
    echo "Logging into GitHub Container Registry..."
    echo $GITHUB_TOKEN | docker login ghcr.io -u $GHCR_USERNAME --password-stdin

    # Push image to GHCR
    docker push $IMAGE_NAME:$TAG

    echo "Image pushed successfully: $IMAGE_NAME:$TAG"
else
    echo "Tests failed. Not pushing the image."
    exit 1
fi
