#!/bin/bash
# ==============================================================================
# NASA C-MAPSS Dataset Downloader
# ==============================================================================
# Downloads the NASA C-MAPSS Turbofan Engine Degradation Dataset via Kaggle API.
#
# Prerequisites:
#   1. Kaggle account: https://www.kaggle.com/
#   2. API token: https://www.kaggle.com/settings → Create New Token
#   3. Place downloaded kaggle.json in ~/.kaggle/
#   4. chmod 600 ~/.kaggle/kaggle.json  (Linux/Mac only)
#
# Usage:
#   bash scripts/download_data.sh
# ==============================================================================

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'  # No Color

DATA_DIR="data/raw"

echo -e "${GREEN}=== NASA C-MAPSS Dataset Downloader ===${NC}"

# Check if Kaggle CLI is installed
if ! command -v kaggle &> /dev/null; then
    echo -e "${RED}Error: kaggle CLI not found.${NC}"
    echo "Install with: pip install kaggle"
    exit 1
fi

# Check if Kaggle API token exists
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo -e "${RED}Error: Kaggle API token not found at ~/.kaggle/kaggle.json${NC}"
    echo "1. Visit https://www.kaggle.com/settings"
    echo "2. Click 'Create New Token'"
    echo "3. Move downloaded kaggle.json to ~/.kaggle/"
    exit 1
fi

# Create data directory
mkdir -p "$DATA_DIR"

# Download dataset
echo -e "${YELLOW}Downloading NASA C-MAPSS dataset...${NC}"
kaggle datasets download -d behrad3d/nasa-cmaps -p "$DATA_DIR" --unzip

# Verify downloaded files
echo -e "${GREEN}=== Verifying downloaded files ===${NC}"
EXPECTED_FILES=(
    "train_FD001.txt"
    "test_FD001.txt"
    "RUL_FD001.txt"
)

for file in "${EXPECTED_FILES[@]}"; do
    # File might be in subdirectory after unzip
    if find "$DATA_DIR" -name "$file" | grep -q .; then
        path=$(find "$DATA_DIR" -name "$file" | head -1)
        size=$(wc -l < "$path")
        echo -e "${GREEN}✓${NC} $file ($size lines) found at $path"
    else
        echo -e "${RED}✗${NC} $file NOT FOUND"
    fi
done

echo -e "${GREEN}=== Download complete ===${NC}"
echo "Next step: Run notebooks/01_EDA.ipynb to explore the data"