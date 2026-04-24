#!/usr/bin/env bash

set -euo pipefail

DATASET="${1:-rd_dataset_formatted.json}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mapfile -t models < <(
  cd "$SCRIPT_DIR" && python - <<'PY'
import sys
sys.path.append("../../common_utils")
from models import OPEN_ITA_MODELS
for model_name in OPEN_ITA_MODELS.keys():
    print(model_name)
PY
)

for model in "${models[@]}"; do
  echo "Running model: $model on dataset: $DATASET"
  cd "$SCRIPT_DIR" && python get_answers_rd_local.py --model "$model" --data "$DATASET"
done
