#!/bin/bash
set -e
CONFIG=${1:-configs/baseline.yaml}

echo "=== Training with $CONFIG ==="
python scripts/train.py

echo "✅ Training complete"
