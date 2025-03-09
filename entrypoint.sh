#!/bin/bash
set -e

echo "Starting Flux Dev inference..."
python3 -u main.py "$@"
