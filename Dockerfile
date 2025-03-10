# Use NVIDIA CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    HF_HOME="/app/models"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY main.py .

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

LABEL org.opencontainers.image.source https://github.com/codysnider/flux

# Default entrypoint, allows CLI arguments
ENTRYPOINT ["/app/entrypoint.sh"]
