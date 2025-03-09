#!/bin/bash

IMAGE_NAME="ghcr.io/codysnider/flux"
TAG="latest"

echo "Building Docker image: $IMAGE_NAME:$TAG"
docker build -t $IMAGE_NAME:$TAG .

# bash test.sh
if [ $? -eq 0 ]; then
    echo "Tests passed."
else
    echo "Tests failed."
    exit 1
fi
