#!/usr/bin/env bash

set -euo pipefail

DATASET="${1:-rd_dataset_formatted.json}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mapfile -t models < <(
  cd "$SCRIPT_DIR" && python - <<'PY'
import sys
sys.path.append("../../common_utils")
from models import API_MODELS
for model_name in API_MODELS.keys():
    print(model_name)
PY
)

for model in "${models[@]}"; do
  echo "Running model: $model on dataset: $DATASET"
  cd "$SCRIPT_DIR" && python get_answers_rd_api.py --model "$model" --data "$DATASET"
done
