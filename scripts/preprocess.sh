#!/bin/bash
set -e
echo "=== [1/2] Preprocessing ==="
python -m src.data.preprocessing

echo "=== [2/2] Windowing ==="
python -m src.data.windowing

echo "✅ Data pipeline complete"
